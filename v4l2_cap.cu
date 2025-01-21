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
#include <linux/videodev2.h>
#include <time.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cudaMath.h"
#include <opencv2/opencv.hpp>

#define WIDTH 1920
#define HEIGHT 1080
#define BUFFER_SIZE WIDTH *HEIGHT
#define BUFFER_COUNT 4 // 버퍼의 개수
#define VIDEO_DEV "/dev/video0"
#define FB_DEV "/dev/fb0"

typedef unsigned char uchar;

cudaError_t process_frame_cuda(const uchar *yuyv, uchar4 *rgba);
cudaError_t launchYUYV(uchar2 *input, size_t inputPitch, uchar4 *output, size_t outputPitch, size_t width, size_t height);

struct buffer
{
    void *start;
    size_t length;
};

struct __align__(8) uchar8
{
    uint8_t a0, a1, a2, a3, a4, a5, a6, a7;
};

static __host__ __device__ __forceinline__ uchar8 make_uchar8(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4, uint8_t a5, uint8_t a6, uint8_t a7)
{
    uchar8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
    return val;
}

static int cond = 1;

/**
 * 프레임 버퍼에 대한 다양한 화면 속성 정보를 담고있다.
 * xres, yres, xres_virtual, yres_virtual
 * bits_per_pixel: 한 픽셀을 표현하는 데 사용되는 비트수(e.g. 16, 24, 32 비트)
 */
static struct fb_var_screeninfo vinfo;

static void sigHandler(int signo)
{
    cond = 0;
}

// YUYV를 RGBA로 변환하는 CUDA 커널
__global__ void yuyvToRgba(uchar4 *src, int srcAlignedWidth, uchar8 *dst, int dstAlignedWidth, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= srcAlignedWidth || y >= height)
        return;

    const uchar4 macroPx = src[y * srcAlignedWidth + x];

    const float y0 = macroPx.x;
    const float y1 = macroPx.z;
    const float u = macroPx.y - 128.0f;
    const float v = macroPx.w - 128.0f;

    const float4 px0 = make_float4(y0 + 1.7790f * u,
                                   y0 - 0.3455f * u - 0.7169f * v,
                                   y0 + 1.4065f * v, 255.0f);

    const float4 px1 = make_float4(y1 + 1.7790f * u,
                                   y1 - 0.3455f * u - 0.7169f * v,
                                   y1 + 1.4065f * v, 255.0f);

    dst[y * dstAlignedWidth + x] = make_uchar8(clamp(px0.x, 0.0f, 255.0f),
                                               clamp(px0.y, 0.0f, 255.0f),
                                               clamp(px0.z, 0.0f, 255.0f),
                                               clamp(px0.w, 0.0f, 255.0f),
                                               clamp(px1.x, 0.0f, 255.0f),
                                               clamp(px1.y, 0.0f, 255.0f),
                                               clamp(px1.z, 0.0f, 255.0f),
                                               clamp(px1.w, 0.0f, 255.0f));
}

void saveImage(const std::string &filename, uchar4 *data, int width, int height)
{
    // uchar4 데이터를 OpenCV Mat으로 변환
    cv::Mat image(height, width, CV_8UC4, data);

    // 이미지 파일로 저장
    cv::imwrite(filename, image);
}

// CUDA 처리 함수
cudaError_t process_frame_cuda(uchar *yuyv, uchar4 *rgba)
{

    // CUDA 디바이스 메모리 포인터들
    uchar2 *d_yuyv = NULL; // 디바이스 YUYV 버퍼
    uchar4 *d_rgba = NULL; // 디바이스 RGBA 버퍼

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // GPU 메모리 할당
    cudaStatus = cudaMalloc(&d_yuyv, BUFFER_SIZE * sizeof(uchar2)); // YUYV 버퍼
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_yuyv: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_rgba, BUFFER_SIZE * sizeof(uchar4)); // RGBA 버퍼
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_rgba: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // 입력 데이터를 GPU로 복사
    cudaStatus = cudaMemcpy(d_yuyv, reinterpret_cast<uchar2 *>(yuyv), BUFFER_SIZE * sizeof(uchar2), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy yuyv to device yuyv: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // 커널 실행
    cudaStatus = launchYUYV(d_yuyv, WIDTH * sizeof(uchar2), d_rgba, WIDTH * sizeof(uchar4), WIDTH, HEIGHT);

    // 커널 실행 오류 체크
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // 결과를 CPU로 복사
    cudaStatus = cudaMemcpy(rgba, d_rgba, BUFFER_SIZE * sizeof(uchar4), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy rgba to host: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

Error:
    cudaFree(d_yuyv);
    cudaFree(d_rgba);

    return cudaStatus;
}

cudaError_t launchYUYV(uchar2 *input, size_t inputPitch, uchar4 *output, size_t outputPitch, size_t width, size_t height)
{
    if (!input || !inputPitch || !output || !outputPitch || !width || !height)
        return cudaErrorInvalidValue;

    const dim3 block(8, 8);
    int numWidth = (width / 2 + block.x - 1) / block.x;
    int numHeight = (height + block.y - 1) / block.y;
    const dim3 grid(numWidth, numHeight);

    const int srcAlignedWidth = inputPitch / sizeof(uchar4);  // normally would be uchar2, but we're doubling up pixels
    const int dstAlignedWidth = outputPitch / sizeof(uchar8); // normally would be uchar4 ^^^

    yuyvToRgba<<<grid, block>>>((uchar4 *)input, srcAlignedWidth, (uchar8 *)output, dstAlignedWidth, width, height);

    return cudaGetLastError();
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

    printf("vinfo.bits_per_pixel: %d\n", vinfo.bits_per_pixel);
    *size = vinfo.yres_virtual * vinfo.xres_virtual * vinfo.bits_per_pixel / 8;
    // mmap을 통해 메모리를 매핑하여 물리 메모리와 프레임 버퍼를 연결, 매핑된 메모리 주소가 반환되며, 이주소를 사용해 프레임버퍼에 데이터를 직접 쓸 수 있다.
    *fbPtr = (uchar4 *)mmap(0, *size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (*fbPtr == MAP_FAILED)
    {
        perror("Failed to mmap framebuffer");
        close(fd);
        return -1;
    }

    return fd;
}

// V4L2 설정
static int init_v4l2(int *fd, struct buffer *buffers)
{
    // 장치의 해상도와 포맷 설정하는 구조체
    struct v4l2_format format;
    // v4l2 장치에서 버퍼를 요청할 때 사용하는 구조체
    // 메모리 매핑 방식으로 비디오 프레임을 처리할 버퍼를 할당하고 설정할 때 사용
    struct v4l2_requestbuffers reqbuf;

    // v4l2에서 비디오 프레임 데이터를 처리하기 위해 사용하는 버퍼에 대한 정보를 담고 있다.
    // ioctl함수에서 VIDIOC_QUERYBUF, VIDIOC_QBUF, VIDIOC_DQBUF와 같은 명령어와 함께 사용된다.
    // 이 버퍼는 비디오 데이터를 큐에 넣고 큐에서 빼는 작업을 위해 사용됨.
    struct v4l2_buffer buf;
    int i;

    // 카메라 장치 열기
    *fd = open(VIDEO_DEV, O_RDWR);
    if (*fd < 0)
    {
        perror("Failed to open video device");
        return -1;
    }

    // 포맷 설정 (YUYV)
    memset(&format, 0, sizeof(format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = WIDTH;
    format.fmt.pix.height = HEIGHT;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    format.fmt.pix.field = V4L2_FIELD_NONE;

    // 카메라 장치의 포맷을 설정
    if (ioctl(*fd, VIDIOC_S_FMT, &format) < 0)
    {
        perror("Failed to set format");
        close(*fd);
        return -1;
    }
    printf("영상의 해상도 : %d x %d\n", format.fmt.pix.width, format.fmt.pix.height);

    // 버퍼 요청
    memset(&reqbuf, 0, sizeof(reqbuf));
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    // 버퍼의 메모리를 설정함. 아래 처럼 설정하게 되면 MMAP을 통해 커널 메모리를 사용자 공간으로 매핑하는 것
    reqbuf.memory = V4L2_MEMORY_MMAP;
    reqbuf.count = BUFFER_COUNT;

    // VIDIOC_REQBUF는 버퍼를 할당해달라는 시스템 콜
    if (ioctl(*fd, VIDIOC_REQBUFS, &reqbuf) < 0)
    {
        perror("Failed to request buffers");
        close(*fd);
        return -1;
    }

    // 버퍼 매핑
    for (i = 0; i < BUFFER_COUNT; i++)
    {
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        // 드라이버에서 비디오 장치의 버퍼에 대한 정보를 조회하는 명령 -> VIDIOC_REQBUFS로 버퍼를 요청한 후, 각 버퍼에 대해 VIDIOC_QUERYBUF를 사용하여 해당 버퍼의 상태를 확인하고 메모리를 매핑하는데 필요한 정보를 가져옴
        if (ioctl(*fd, VIDIOC_QUERYBUF, &buf) < 0)
        {
            perror("Failed to query buffer");
            close(*fd);
            return -1;
        }

        // buffers는 우리가 정의한 구조체 변수로 화면에 보여줄 버퍼를 다루게되고
        // buf는 v4l2에서 다루는 버퍼로 커널 영역에 존재한다.
        // 카메라로부터 비디오 프레임을 캡쳐하면 buf에 저장되는데, 그것을 유저 영역에서 접근할 수 없기 때문에 mmap을 통해서 buffers라는 구조체로 buf에 접근할 수 있도록 설정해주는 것!
        buffers[i].length = buf.length;
        // start 인자에 mmap을 할당한다.
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED)
        {
            perror("Failed to mmap buffer");
            close(*fd);
            return -1;
        }

        // 큐에 버퍼를 넣음
        if (ioctl(*fd, VIDIOC_QBUF, &buf) < 0)
        {
            perror("Failed to queue buffer");
            close(*fd);
            return -1;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    // 비디오 스트리밍을 시작하는 시스템 콜 -> 이 시스템 콜을 호출하면 큐에있는 버퍼에 비디오 프레임이 저장되게 되는데 이것을 사용하기 위해 나중에 디큐를 해주는 것!
    if (ioctl(*fd, VIDIOC_STREAMON, &type) < 0)
    {
        perror("Failed to start capturing");
        close(*fd);
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    uchar4 *rgbBuffer, *fbPtr;
    int cam_fd, fb_fd;
    int fbSize; // YUYV는 픽셀당 2바이트
    struct buffer buffers[BUFFER_COUNT];
    struct v4l2_buffer buf; // V4L2에서 사용할 메모리 버퍼

    signal(SIGINT, sigHandler);

    // V4L2 초기화
    if (init_v4l2(&cam_fd, buffers) < 0)
    {
        fprintf(stderr, "V4L2 initialization failed\n");
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
    printf("fbSize: %d\n", fbSize);
    rgbBuffer = (uchar4 *)malloc(fbSize);
    if (!rgbBuffer)
    {
        perror("Failed to allocate buffers");
        close(fb_fd);
        return -1;
    }

    // FPS 계산을 위한 변수
    struct timespec start_time, end_time;
    long frame_count = 0;
    int fps = 0;

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    printf("vinfo.xres: %d\n", vinfo.xres);
    printf("vinfo.yres: %d\n", vinfo.yres);
    printf("width: %d\n", WIDTH);
    printf("height %d\n", HEIGHT);

    // V4L2를 이용한 영상의 캡쳐 및 표시
    while (cond)
    {
        if (cond >= 10)
        {
            break;
        }

        // 버퍼 초기화
        memset(&buf, 0, sizeof(buf));

        // MMAP 기반으로 영상 캡쳐
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        // STREAMON 명령으로 캡쳐한 비디오 프레임을 디큐한다.
        if (ioctl(cam_fd, VIDIOC_DQBUF, &buf) < 0)
        {
            perror("Failed to dequeue buffer");
            break;
        }

        // init_v4l2함수에서 캡쳐한 비디오 프레임이 mmap(buffers[buf.index]).start에 저장되어있는데 이것을 인자로 넣어준다.
        cudaError_t cudaStatus = process_frame_cuda((uchar *)buffers[buf.index].start, rgbBuffer);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        frame_count++;
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        if (elapsed_time >= 1.0)
        {
            fps = frame_count / elapsed_time;
            frame_count = 0;
            clock_gettime(CLOCK_MONOTONIC, &start_time);

            // 콘솔에 FPS 출력
            printf("FPS: %d\n", fps);
        }

        // memcpy(fbPtr, rgbBuffer, fbSize);
        saveImage("output.jpg", rgbBuffer, WIDTH, HEIGHT);

        // 버퍼를 다시 큐에 넣기
        if (ioctl(cam_fd, VIDIOC_QBUF, &buf) < 0)
        {
            perror("Failed to queue buffer");
            break;
        }

        cond++;
    }

    printf("\nGood Bye!!!\n");

    // 캡쳐 종료
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(cam_fd, VIDIOC_STREAMOFF, &type);

    // 메모리 정리
    for (int i = 0; i < BUFFER_COUNT; i++)
    {
        munmap(buffers[i].start, buffers[i].length);
    }
    munmap(fbPtr, fbSize);
    free(rgbBuffer);

    // 파일디스크립터 정리
    close(cam_fd);
    close(fb_fd);

    return 0;
}
