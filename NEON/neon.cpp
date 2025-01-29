#include "neon.h"
#include <arm_neon.h>
#include <cstring>

NEON::NEON(int width, int height)
{
    this->width = width;
    this->height = height;
}
NEON::~NEON()
{
}

void NEON::yuyvToRgbaHostSIMD(uchar *yuyv, uchar *rgba)
{
    // unsigned char *aligned_buffer = NULL;

    // // posix_memalign으로 32바이트 정렬된 메모리 할당
    // if (posix_memalign((void **)&aligned_buffer, 32, size) != 0)
    // {
    //     return NULL;
    // }

    // // v4l2 버퍼의 데이터를 정렬된 버퍼로 복사
    // memcpy(aligned_buffer, v4l2_buffer, size);

    int istride = width * 2;
    int row_size = width * 4; // RGBA stride

    // NEON constants for YUV to RGB conversion
    const int16x8_t const_298 = vdupq_n_s16(298);
    const int16x8_t const_409 = vdupq_n_s16(409);
    const int16x8_t const_100 = vdupq_n_s16(100);
    const int16x8_t const_208 = vdupq_n_s16(208);
    const int16x8_t const_516 = vdupq_n_s16(516);
    const int16x8_t const_128 = vdupq_n_s16(128);

    for (int y = 0; y < height; y++)
    {
        uint8_t *in_line = yuyv + y * istride;
        uint8_t *out_line = rgba + y * row_size;

        for (int x = 0; x < width; x += 8)
        { // Process 8 pixels (4 YUYV pairs) at a time
            // Load 8 pixels of YUYV (32 bytes)
            uint8x8x4_t yuyv_data = vld4_u8(in_line);

            // Convert to signed 16-bit
            int16x8_t y0 = vreinterpretq_s16_u16(vmovl_u8(yuyv_data.val[0])); // Y0
            int16x8_t u = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(yuyv_data.val[1])), // U
                const_128);
            int16x8_t y1 = vreinterpretq_s16_u16(vmovl_u8(yuyv_data.val[2])); // Y1
            int16x8_t v = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(yuyv_data.val[3])), // V
                const_128);

            // Calculate RGB components for Y0
            int16x8_t r0 = vshrq_n_s16(vaddq_s16(
                                           vaddq_s16(vmulq_s16(const_298, y0), vmulq_s16(const_516, u)),
                                           const_128),
                                       8);

            int16x8_t g0 = vshrq_n_s16(vaddq_s16(
                                           vsubq_s16(
                                               vsubq_s16(vmulq_s16(const_298, y0), vmulq_s16(const_100, u)),
                                               vmulq_s16(const_208, v)),
                                           const_128),
                                       8);

            int16x8_t b0 = vshrq_n_s16(vaddq_s16(
                                           vaddq_s16(vmulq_s16(const_298, y0), vmulq_s16(const_409, v)),
                                           const_128),
                                       8);

            // Calculate RGB components for Y1
            int16x8_t r1 = vshrq_n_s16(vaddq_s16(
                                           vaddq_s16(vmulq_s16(const_298, y1), vmulq_s16(const_516, u)),
                                           const_128),
                                       8);

            int16x8_t g1 = vshrq_n_s16(vaddq_s16(
                                           vsubq_s16(
                                               vsubq_s16(vmulq_s16(const_298, y1), vmulq_s16(const_100, u)),
                                               vmulq_s16(const_208, v)),
                                           const_128),
                                       8);

            int16x8_t b1 = vshrq_n_s16(vaddq_s16(
                                           vaddq_s16(vmulq_s16(const_298, y1), vmulq_s16(const_409, v)),
                                           const_128),
                                       8);

            // Narrow and combine results
            uint8x8x4_t rgba0, rgba1;
            rgba0.val[0] = vqmovun_s16(r0); // R
            rgba0.val[1] = vqmovun_s16(g0); // G
            rgba0.val[2] = vqmovun_s16(b0); // B
            rgba0.val[3] = vdup_n_u8(255);  // A

            rgba1.val[0] = vqmovun_s16(r1); // R
            rgba1.val[1] = vqmovun_s16(g1); // G
            rgba1.val[2] = vqmovun_s16(b1); // B
            rgba1.val[3] = vdup_n_u8(255);  // A

            // Store results
            vst4_u8(out_line, rgba0);
            vst4_u8(out_line + 32, rgba1);

            in_line += 32;  // Move to next 8 YUYV pixels
            out_line += 64; // Move to next 8 RGBA pixels
        }
    }
}