#include <cuda_runtime.h>
#include <stdio.h>

// CUDA 디바이스 메모리 포인터들
static unsigned char *d_yuyv = NULL;    // 디바이스 YUYV 버퍼
static unsigned char *d_rgba = NULL;    // 디바이스 RGBA 버퍼

// YUV를 RGB로 변환하는 디바이스 함수
__device__ void yuv2rgb_device(unsigned char y, unsigned char u, unsigned char v, int *r, int *g, int *b) {
    int c = y - 16;
    int d = u - 128;
    int e = v - 128;

    *r = (298 * c + 409 * e + 128) >> 8;
    *g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    *b = (298 * c + 516 * d + 128) >> 8;

    // 클리핑
    *r = (*r < 0) ? 0 : (*r > 255) ? 255 : *r;
    *g = (*g < 0) ? 0 : (*g > 255) ? 255 : *g;
    *b = (*b < 0) ? 0 : (*b > 255) ? 255 : *b;

    // R과 B 교환
    int temp = *r;
    *r = *b;
    *b = temp;
}

// YUYV를 RGBA로 변환하는 CUDA 커널
__global__ void yuyv2rgba_kernel(const unsigned char *yuyv, unsigned char *rgba, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // 2픽셀씩 처리 (YUYV 포맷 특성상)
    x *= 2;
    if (x < width - 1) {
        unsigned char y0 = yuyv[y * width * 2 + x * 2];
        unsigned char u  = yuyv[y * width * 2 + x * 2 + 1];
        unsigned char y1 = yuyv[y * width * 2 + x * 2 + 2];
        unsigned char v  = yuyv[y * width * 2 + x * 2 + 3];

        int r0, g0, b0, r1, g1, b1;
        yuv2rgb_device(y0, u, v, &r0, &g0, &b0);
        yuv2rgb_device(y1, u, v, &r1, &g1, &b1);

        // 첫 번째 픽셀
        rgba[y * width * 4 + x * 4]     = r0;
        rgba[y * width * 4 + x * 4 + 1] = g0;
        rgba[y * width * 4 + x * 4 + 2] = b0;
        rgba[y * width * 4 + x * 4 + 3] = 255;

        // 두 번째 픽셀
        rgba[y * width * 4 + (x + 1) * 4]     = r1;
        rgba[y * width * 4 + (x + 1) * 4 + 1] = g1;
        rgba[y * width * 4 + (x + 1) * 4 + 2] = b1;
        rgba[y * width * 4 + (x + 1) * 4 + 3] = 255;
    }
}

// CUDA 초기화 함수
int init_cuda(int width, int height) {
    cudaError_t err;

    // GPU 메모리 할당
    err = cudaMalloc(&d_yuyv, width * height * 2);  // YUYV 버퍼
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_yuyv: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc(&d_rgba, width * height * 4);  // RGBA 버퍼
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_rgba: %s\n", cudaGetErrorString(err));
        cudaFree(d_yuyv);
        return -1;
    }

    return 0;
}

// CUDA 정리 함수
void cleanup_cuda(void) {
    if (d_yuyv) cudaFree(d_yuyv);
    if (d_rgba) cudaFree(d_rgba);
    d_yuyv = NULL;
    d_rgba = NULL;
}

// CUDA 처리 함수
int process_frame_cuda(const unsigned char *yuyv, unsigned char *rgba, int width, int height) {
    cudaError_t err;

    // 입력 데이터를 GPU로 복사
    err = cudaMemcpy(d_yuyv, yuyv, width * height * 2, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy yuyv to device: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 커널 실행 설정
    dim3 block_size(16, 16);
    dim3 grid_size((width / 2 + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    // 커널 실행
    yuyv2rgba_kernel<<<grid_size, block_size>>>(d_yuyv, d_rgba, width, height);

    // 커널 실행 오류 체크
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 결과를 CPU로 복사
    err = cudaMemcpy(rgba, d_rgba, width * height * 4, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy rgba to host: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}