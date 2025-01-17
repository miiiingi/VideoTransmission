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

#define WIDTH 1920
#define HEIGHT 1080
#define BUFFER_SIZE WIDTH *HEIGHT
#define BUFFER_COUNT 4 // 버퍼의 개수
#define VIDEO_DEV "/dev/video0"
#define FB_DEV "/dev/fb0"

typedef unsigned char uchar;

// v4l2_cap.c
cudaError_t process_frame_cuda(const uchar *yuyv, uchar *rgba);

struct buffer
{
    void *start;
    size_t length;
};

static int cond = 1;
// 프레임 버퍼에 대한 다양한 화면 속성 정보를 담고있다.
// xres, yres, xres_virtual, yres_virtual
// bits_per_pixel: 한 픽셀을 표현하는 데 사용되는 비트수(e.g. 16, 24, 32 비트)
static struct fb_var_screeninfo vinfo;

static void sigHandler(int signo)
{
    cond = 0;
}

static inline int clip(int value, int min, int max)
{
    return (value > max ? max : (value < min ? min : value));
}

// CUDA 초기화 함수
#include <cuda_runtime.h>
#include <stdio.h>

// YUV를 RGB로 변환하는 디바이스 함수
__device__ void yuv2rgb_device(uchar y, uchar u, uchar v, int *r, int *g, int *b)
{
    int c = y - 16;
    int d = u - 128;
    int e = v - 128;

    *r = (298 * c + 409 * e + 128) >> 8;
    *g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    *b = (298 * c + 516 * d + 128) >> 8;

    *r = min(max(*r, 0), 255);
    *g = min(max(*g, 0), 255);
    *b = min(max(*b, 0), 255);

    // R과 B 교환
    int temp = *r;
    *r = *b;
    *b = temp;
}

// YUYV를 RGBA로 변환하는 CUDA 커널
__global__ void yuyv2rgba_kernel(const uchar *yuyv, uchar *rgba, int width, int height)
{
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // YUYV는 2픽셀 단위로 처리
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    uchar y0 = yuyv[y * width * 2 + x * 2];
    uchar u = yuyv[y * width * 2 + x * 2 + 1];
    uchar y1 = yuyv[y * width * 2 + x * 2 + 2];
    uchar v = yuyv[y * width * 2 + x * 2 + 3];

    int r0, g0, b0, r1, g1, b1;
    yuv2rgb_device(y0, u, v, &r0, &g0, &b0);
    yuv2rgb_device(y1, u, v, &r1, &g1, &b1);

    // 첫 번째 픽셀
    rgba[y * width * 4 + x * 4] = r0;
    rgba[y * width * 4 + x * 4 + 1] = g0;
    rgba[y * width * 4 + x * 4 + 2] = b0;
    rgba[y * width * 4 + x * 4 + 3] = 255;

    // 두 번째 픽셀
    rgba[y * width * 4 + (x + 1) * 4] = r1;
    rgba[y * width * 4 + (x + 1) * 4 + 1] = g1;
    rgba[y * width * 4 + (x + 1) * 4 + 2] = b1;
    rgba[y * width * 4 + (x + 1) * 4 + 3] = 255;
}

void gpuInfo()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %lu bytes\n", prop.totalGlobalMem);
        printf("  Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max thread dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0],
               prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        printf("  Clock rate: %d kHz\n", prop.clockRate);
    }
}

// CUDA 처리 함수
cudaError_t process_frame_cuda(const uchar *yuyv, uchar *rgba)
{

    // CUDA 디바이스 메모리 포인터들
    uchar *d_yuyv = NULL; // 디바이스 YUYV 버퍼
    uchar *d_rgba = NULL; // 디바이스 RGBA 버퍼

    // 커널 실행 설정
    dim3 block_size(32, 32);
    // 정수로 나눠주기 위해 아래와 같이 연산
    dim3 grid_size((WIDTH + block_size.x * 2 - 1) / block_size.x * 2, (HEIGHT + block_size.y - 1) / block_size.y);

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // GPU 메모리 할당
    cudaStatus = cudaMalloc(&d_yuyv, BUFFER_SIZE * 2 * sizeof(uchar)); // YUYV 버퍼
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_yuyv: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc(&d_rgba, BUFFER_SIZE * 4 * sizeof(uchar)); // RGBA 버퍼
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_rgba: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // 입력 데이터를 GPU로 복사
    cudaStatus = cudaMemcpy(d_yuyv, yuyv, BUFFER_SIZE * 2 * sizeof(uchar), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy yuyv to device yuyv: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // 커널 실행
    yuyv2rgba_kernel<<<grid_size, block_size>>>(d_yuyv, d_rgba, WIDTH, HEIGHT);

    // 커널 실행 오류 체크
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // 결과를 CPU로 복사
    cudaStatus = cudaMemcpy(rgba, d_rgba, BUFFER_SIZE * 4 * sizeof(uchar), cudaMemcpyDeviceToHost);
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

// 프레임버퍼를 설정하는 함수
int init_framebuffer(uchar **fbPtr, int *size)
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
    *fbPtr = (uchar *)mmap(0, *size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
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
    uchar *rgbBuffer, *fbPtr;
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
    rgbBuffer = (uchar *)malloc(fbSize);
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

    gpuInfo();
    // V4L2를 이용한 영상의 캡쳐 및 표시
    while (cond)
    {
        if (cond >= 100)
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
        cudaError_t cudaStatus = process_frame_cuda((const uchar *)buffers[buf.index].start, rgbBuffer);
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

        // 프레임버퍼에 RGB565 데이터 쓰기
        memcpy(fbPtr, rgbBuffer, fbSize);

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
