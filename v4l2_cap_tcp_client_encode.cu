#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "DS_timer.h"
#include "DS_definitions.h"

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <linux/videodev2.h>
}

#define WIDTH 1920
#define HEIGHT 1080
#define BUFFER_SIZE WIDTH *HEIGHT

#define VIDEO_DEV "/dev/video0"
#define FB_DEV "/dev/fb0"
#define SERVER_PORT 8080
#define SERVER_IP "127.0.0.1"
// get base type (uint8 or float) from vector

typedef unsigned char uchar;
static int cond = 1;
/*
 * framebuffer에 비디오 프레임을 그리기 위해 포인터에 대한 변수 선언
 */
/*
 * 프레임 버퍼에 대한 다양한 화면 속성 정보를 담고있다.
 * xres, yres, xres_virtual, yres_virtual
 * bits_per_pixel: 한 픽셀을 표현하는 데 사용되는 비트수(e.g. 16, 24, 32 비트)
 */
static struct fb_var_screeninfo vinfo;
/*
 * 카메라로부터 온 비디오 프레임 YUYV 포맷을 프레임 버퍼에서 출력하기 위해 RGB 16bit로 변환하는 함수
 * */
static void yuyv2rgba(uchar *yuyv, uchar4 *fbPtr, int width, int height);
/*
 * 프레임 버퍼를 초기화하는 함수
 * */
static int init_framebuffer(uchar4 **fbPtr, int *size);
/*
 * 디버깅을 위해 프레임 버퍼에 출력된 데이터를 bmp 파일로 저장
 * */
static inline int clip(int value, int min, int max);
static void sigHandler(int signo);
cudaError_t process_frame_cuda(uint8_t *I420, uchar4 *rgba, DS_timer &timer);
void saveImage(const std::string &filename, uchar4 *data, int width, int height);
template <typename T, bool formatYV12>
static cudaError_t launch420ToRGB(uint8_t *srcDev, T *dstDev, size_t width, size_t height, cudaStream_t stream);

static inline __device__ float clamp(float x)
{
    return fminf(fmaxf(x, 0.0f), 255.0f);
}
inline __device__ __host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

static inline __device__ float3 YUV2RGB(float Y, float U, float V)
{
    U -= 128.0f;
    V -= 128.0f;

#if 1
    return make_float3(clamp(Y + 1.402f * V),
                       clamp(Y - 0.344f * U - 0.714f * V),
                       clamp(Y + 1.772f * U));
#else
    return make_float3(clamp(Y + 1.140f * V),
                       clamp(Y - 0.395f * U - 0.581f * V),
                       clamp(Y + 2.3032f * U));
#endif
}

template <typename T, bool formatYV12>
__global__ void I420ToRGB(uint8_t *srcImage, int srcPitch,
                          T *dstImage, int dstPitch,
                          int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width)
        return;

    if (y >= height)
        return;

    const int x2 = x / 2;
    const int y2 = y / 2;

    const int srcPitch2 = srcPitch / 2;
    const int planeSize = srcPitch * height;

    // get the YUV plane offsets
    uint8_t *y_plane = srcImage;
    uint8_t *u_plane;
    uint8_t *v_plane;

    if (formatYV12)
    {
        v_plane = y_plane + planeSize;
        u_plane = v_plane + (planeSize / 4); // size of U & V planes is 25% of Y plane
    }
    else
    {
        u_plane = y_plane + planeSize; // in I420, order of U & V planes is reversed
        v_plane = u_plane + (planeSize / 4);
    }

    // read YUV pixel
    const float Y = y_plane[y * srcPitch + x];
    const float U = u_plane[y2 * srcPitch2 + x2];
    const float V = v_plane[y2 * srcPitch2 + x2];

    const float3 RGB = YUV2RGB(Y, U, V);

    dstImage[y * width + x] = make_uchar4(RGB.x, RGB.y, RGB.z, 255);
}

int main(int argc, char **argv)
{
    signal(SIGINT, sigHandler);
    int fb_fd;
    int fbSize; // YUYV는 픽셀당 2바이트
    uchar4 *rgbBuffer, *fbPtr;

    DS_timer timer(5);
    timer.setTimerName(0, (char *)"CUDA Total");
    timer.setTimerName(1, (char *)"Computation(Kernel)");
    timer.setTimerName(2, (char *)"Data Trans. : Host -> Device");
    timer.setTimerName(3, (char *)"Data Trans. : Device -> Host");
    timer.setTimerName(4, (char *)"Host Performance");
    timer.initTimers();

    int sock;
    struct sockaddr_in server_addr;
    ssize_t bytes_received = 0;

    // 소켓 생성
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        perror("socket() failed");
        close(sock);
        return -1;
    }

    // 서버 주소 설정
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    server_addr.sin_port = htons(SERVER_PORT);

    // 서버에 연결
    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        perror("connect() failed");
        close(sock);
        return -1;
    }

    printf("Connected to server...\n");

    // FFmpeg 디코더 초기화
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec)
    {
        fprintf(stderr, "Codec not found\n");
        return -1;
    }

    AVCodecContext *dec_ctx = avcodec_alloc_context3(codec);
    if (!dec_ctx)
    {
        fprintf(stderr, "Could not allocate video codec context\n");
        return -1;
    }
    dec_ctx->time_base = (AVRational){1, 120};
    dec_ctx->framerate = (AVRational){120, 1};

    if (avcodec_open2(dec_ctx, codec, NULL) < 0)
    {
        fprintf(stderr, "Could not open codec\n");
        return -1;
    }

    AVPacket *pkt = av_packet_alloc();
    if (!pkt)
    {
        fprintf(stderr, "Could not allocate packet\n");
        return -1;
    }

    AVFrame *frame = av_frame_alloc();
    if (!frame)
    {
        fprintf(stderr, "Could not allocate frame\n");
        return -1;
    }

    // 프레임버퍼 초기화
    fb_fd = init_framebuffer(&fbPtr, &fbSize);
    if (fb_fd < 0)
    {
        fprintf(stderr, "Failed to initialize framebuffer\n");
        return -1;
    }

    // 영상을 저장할 메모리 할당
    rgbBuffer = (uchar4 *)malloc(fbSize);
    printf("fbSize: %d\n", fbSize);
    printf("output size: %zu\n", BUFFER_SIZE * sizeof(uchar4));

    if (!rgbBuffer)
    {
        perror("Failed to allocate buffers");
        close(fb_fd);
        return -1;
    }

    // .h264 파일로 저장할 파일 포인터 열기
    FILE *h264_file = fopen("client_received.h264", "wb");
    if (!h264_file)
    {
        perror("Failed to open output .h264 file");
        close(sock);
        return -1;
    }

    uint8_t *packet_data = (uint8_t *)malloc(BUFFER_SIZE * (3 / 2) * sizeof(uint8_t));

    while (cond)
    {
        if (cond >= 5)
        {
            break;
        }

        // recv()로 패킷을 수신
        bytes_received = recv(sock, packet_data, BUFFER_SIZE * (3 / 2) * sizeof(uint8_t), 0);
        if (bytes_received < 0)
        {
            perror("recv failed");
            break;
        }
        else if (bytes_received == 0)
        {
            printf("Server disconnected\n");
            break;
        }

        printf("\rReceived %zd bytes\n", bytes_received);
        fflush(stdout);

        // NAL 유닛인지 확인하고 기록
        if (packet_data[0] == 0x00 && packet_data[1] == 0x00 && packet_data[2] == 0x00 && packet_data[3] == 0x01)
        {
            // fwrite(packet_data, 1, bytes_received, h264_file);
            pkt->size = bytes_received;
            pkt->data = packet_data;
        }
        // fflush(h264_file); // 데이터를 즉시 파일로 기록

        cudaError_t cudaStatus = process_frame_cuda(packet_data, rgbBuffer, timer);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }
        saveImage("outputC.jpg", rgbBuffer, WIDTH, HEIGHT);

        // 프레임버퍼에 RGB 데이터를 복사
        // memcpy(fbPtr, rgbBuffer, fb_height * fb_width * 2);
        cond++;
    }

    printf("\nGood Bye!!!\n");
    // 파일디스크립터 정리
    munmap(fbPtr, fbSize);
    free(rgbBuffer);
    free(packet_data);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&dec_ctx);
    close(fb_fd);

    return 0;
}
// YUYV 데이터를 rgba로 변환하는 함수
void yuyv2rgba(uchar *yuyv, unsigned short *fbPtr, int width, int height)
{
    uchar *in = (uchar *)yuyv;
    unsigned short pixel;
    int istride = width * 2; /* 이미지의 폭을 넘어가면 다음 라인으로 내려가도록 설정 */
    int x, y, j;
    int y0, u, y1, v, r, g, b;
    long loc = 0;

    for (y = 0; y < height; ++y)
    {
        // j는 한 라인의 바이트 , x는 한 라인의 픽셀 수
        for (j = 0, x = 0; j < vinfo.xres * 2; j += 4, x += 2)
        {
            if (j >= width * 2)
            { /* 현재의 화면에서 이미지를 넘어서는 빈 공간을 처리 */
                loc++;
                loc++;
                continue;
            }
            /* YUYV 성분을 분리 */
            y0 = in[j];
            u = in[j + 1] - 128;
            y1 = in[j + 2];
            v = in[j + 3] - 128;

            /* YUV를 RGB로 전환: Y0 + U + V */
            r = clip((298 * y0 + 409 * v + 128) >> 8, 0, 255);
            g = clip((298 * y0 - 100 * u - 208 * v + 128) >> 8, 0, 255);
            b = clip((298 * y0 + 516 * u + 128) >> 8, 0, 255);
            pixel = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3); /* 16비트 컬러로 전환 */
            fbPtr[loc++] = pixel;

            /* YUV를 RGB로 전환 : Y1 + U + V */
            r = clip((298 * y1 + 409 * v + 128) >> 8, 0, 255);
            g = clip((298 * y1 - 100 * u - 208 * v + 128) >> 8, 0, 255);
            b = clip((298 * y1 + 516 * u + 128) >> 8, 0, 255);
            pixel = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3); /* 16비트 컬러로 전환*/
            fbPtr[loc++] = pixel;
        }
        in += istride;
    }
}

// 프레임버퍼를 설정하는 함수
int init_framebuffer(uchar4 **fbPtr, int *size)
{
    // frame buffer 장치 열기
    int fd = open(FB_DEV, O_RDWR);
    if (fd < 0)
    {
        perror("Failed to open framebuffer device");
        return -1;
    }

    // ioctl은 디바이스와 소통하기 위한 시스템 호출로 하드웨어 장치의 설정과 상태를 제어할 때 사용.
    // 파일디스크립터를 통해 장치에 명령을 전달하고 특정 명령에 대한 처리를 요청
    // 아래의 함수는 프레임버퍼 장치에 FBIOGET_VSCREENINFO 명령을 보내 화면 정보를 가져오라는 요청을 하는 것
    // 성공적으로 호출되면 프레임버퍼의 화면 설정 정보가 vinfo라는 fb_var_screeninfo 구조체에 저장됨
    if (ioctl(fd, FBIOGET_VSCREENINFO, &vinfo))
    {
        perror("Error reading variable information");
        close(fd);
        return -1;
    }
    /*
    printf("vinfo.yres: %d\n", vinfo.yres);
    printf("vinfo.yres_virtual: %d\n", vinfo.yres_virtual);
    */

    // mmap을 통해 메모리를 매핑하여 물리 메모리와 프레임 버퍼를 연결, 매핑된 메모리 주소가 반환되며, 이주소를 사용해 프레임버퍼에 데이터를 직접 쓸 수 있다.
    *size = vinfo.yres_virtual * vinfo.xres_virtual * vinfo.bits_per_pixel / 8;

    *fbPtr = (uchar4 *)mmap(0, *size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (*fbPtr == MAP_FAILED)
    {
        perror("Failed to mmap framebuffer");
        close(fd);
        return -1;
    }

    return fd;
}

static inline int clip(int value, int min, int max)
{
    return (value > max ? max : (value < min ? min : value));
}

static void sigHandler(int signo)
{
    cond = 0;
}

// // YUV420P 데이터를 rgba로 변환하는 함수
// void yuv420p_to_rgba(unsigned char *yuv420p_data[3], uchar4 *rgba_data, int width, int height, int fb_width)
// {
//     int y, x, y_stride, uv_stride;
//     int r, g, b;
//     int y_value, u_value, v_value;

//     y_stride = width;
//     uv_stride = width / 2;
//     int start_x = 400;
//     int start_y = 250;

//     for (y = 0; y < height; y++)
//     {
//         for (x = 0; x < width; x++)
//         {
//             int y_index = y * y_stride + x;
//             int uv_index = (y / 2) * uv_stride + (x / 2);

//             y_value = yuv420p_data[0][y_index];
//             u_value = yuv420p_data[1][uv_index] - 128;
//             v_value = yuv420p_data[2][uv_index] - 128;

//             // YUV -> RGB 변환 공식
//             r = clip((298 * y_value + 409 * v_value + 128) >> 8, 0, 255);
//             g = clip((298 * y_value - 100 * u_value - 208 * v_value + 128) >> 8, 0, 255);
//             b = clip((298 * y_value + 516 * u_value + 128) >> 8, 0, 255);

//             // RGB888 -> rgba로 변환
//             rgba_data[(start_y + y) * fb_width + (start_x + x)] = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
//         }
//     }
// }

// CUDA 처리 함수
cudaError_t process_frame_cuda(uint8_t *I420, uchar4 *rgba, DS_timer &timer)
{

    // CUDA 디바이스 메모리 포인터들
    uint8_t *d_I420 = NULL;
    uchar4 *d_rgba = NULL;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // GPU 메모리 할당
    cudaStatus = cudaMalloc(&d_I420, BUFFER_SIZE * (3 / 2) * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_I420: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_rgba, BUFFER_SIZE * sizeof(uchar4)); // RGBA 버퍼
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_rgba: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // 입력 데이터를 GPU로 복사
    timer.onTimer(2);
    cudaStatus = cudaMemcpy(d_I420, I420, BUFFER_SIZE * (3 / 2) * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy yuyv to device yuyv: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    timer.offTimer(2);

    // 커널 실행
    timer.onTimer(1);
    cudaStatus = launch420ToRGB<uchar4, false>(d_I420, d_rgba, WIDTH, HEIGHT, 0);
    timer.offTimer(1);

    // 커널 실행 오류 체크
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // 결과를 CPU로 복사
    timer.onTimer(3);
    cudaStatus = cudaMemcpy(rgba, d_rgba, BUFFER_SIZE * sizeof(uchar4), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy rgba to host: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    timer.offTimer(3);

Error:
    if (d_I420)
    {
        cudaFree(d_I420);
    }

    if (d_rgba)
    {
        cudaFree(d_rgba);
    }
    return cudaStatus;
}

template <typename T, bool formatYV12>
cudaError_t launch420ToRGB(uint8_t *srcDev, T *dstDev, size_t width, size_t height, cudaStream_t stream)
{
    if (!srcDev || !dstDev)
        return cudaErrorInvalidDevicePointer;

    if (width == 0 || height == 0)
        return cudaErrorInvalidValue;

    const int srcPitch = width * sizeof(uint8_t);
    const int dstPitch = width * sizeof(T);

    const dim3 blockDim(8, 8);
    // const dim3 gridDim((width+(2*blockDim.x-1))/(2*blockDim.x), (height+(blockDim.y-1))/blockDim.y, 1);
    const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y), 1);

    // Check for any errors before launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Pre-kernel launch error: %s\n", cudaGetErrorString(err));
        return err;
    }

    I420ToRGB<T, formatYV12><<<gridDim, blockDim, 0, stream>>>(srcDev, srcPitch, dstDev, dstPitch, width, height);
    cudaDeviceSynchronize();

    // Check for any errors after launching the kernel
    if (err != cudaSuccess)
    {
        printf("Post-kernel launch error: %s\n", cudaGetErrorString(err));
        return err;
    }

    return cudaGetLastError();
}

void saveImage(const std::string &filename, uchar4 *data, int width, int height)
{
    // uchar4 데이터를 OpenCV Mat으로 변환
    cv::Mat image(height, width, CV_8UC4, data);

    // 이미지 파일로 저장
    cv::imwrite(filename, image);
}
