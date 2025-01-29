#include "neon.h"
#include <immintrin.h>
#include <cstring>
#include <cstdint>

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
    int istride = width * 2;
    int row_size = width * 4;
    
    // YUV to RGB conversion constants based on the CUDA implementation
    // Scaled by 1024 to maintain precision in integer arithmetic
    __m256i const_298 = _mm256_set1_epi16(298);  // 1.0 scaling factor
    __m256i const_409 = _mm256_set1_epi16(409);  // 1.4065 * 298
    __m256i const_100 = _mm256_set1_epi16(100);  // 0.3455 * 298
    __m256i const_208 = _mm256_set1_epi16(208);  // 0.7169 * 298
    __m256i const_516 = _mm256_set1_epi16(516);  // 1.7790 * 298
    __m256i const_128 = _mm256_set1_epi16(128);
    __m256i const_255 = _mm256_set1_epi16(255);
    __m256i const_0 = _mm256_set1_epi16(0);

    for (int y = 0; y < height; y++)
    {
        uint8_t *in_line = yuyv + y * istride;
        uint8_t *out_line = rgba + y * row_size;

        for (int x = 0; x < width; x += 8) // Process 8 pixels at a time
        {
            // Load 16 bytes of YUYV data (8 pixels)
            __m256i yuyv_data = _mm256_loadu_si256((__m256i *)in_line);

            // Separate Y, U, V components
            __m256i y_values = _mm256_and_si256(yuyv_data, _mm256_set1_epi16(0x00FF));
            __m256i uv_values = _mm256_srli_epi16(yuyv_data, 8);
            __m256i u_values = _mm256_and_si256(uv_values, _mm256_set1_epi16(0x00FF));
            __m256i v_values = _mm256_srli_epi16(uv_values, 8);

            // Center U and V around zero
            u_values = _mm256_sub_epi16(u_values, const_128);
            v_values = _mm256_sub_epi16(v_values, const_128);

            // Calculate R = Y + 1.4065V
            __m256i r = _mm256_add_epi16(
                _mm256_mullo_epi16(const_298, y_values),
                _mm256_mullo_epi16(const_409, v_values)
            );
            r = _mm256_srai_epi16(r, 8);

            // Calculate G = Y - 0.3455U - 0.7169V
            __m256i g = _mm256_sub_epi16(
                _mm256_mullo_epi16(const_298, y_values),
                _mm256_add_epi16(
                    _mm256_mullo_epi16(const_100, u_values),
                    _mm256_mullo_epi16(const_208, v_values)
                )
            );
            g = _mm256_srai_epi16(g, 8);

            // Calculate B = Y + 1.7790U
            __m256i b = _mm256_add_epi16(
                _mm256_mullo_epi16(const_298, y_values),
                _mm256_mullo_epi16(const_516, u_values)
            );
            b = _mm256_srai_epi16(b, 8);

            // Clamp values between 0 and 255
            r = _mm256_min_epi16(_mm256_max_epi16(r, const_0), const_255);
            g = _mm256_min_epi16(_mm256_max_epi16(g, const_0), const_255);
            b = _mm256_min_epi16(_mm256_max_epi16(b, const_0), const_255);

            // Pack RGB components with 255 alpha
            __m256i rgba_lo = _mm256_unpacklo_epi8(
                r,
                _mm256_unpacklo_epi8(g, b)
            );
            __m256i rgba_hi = _mm256_unpackhi_epi8(
                r,
                _mm256_unpackhi_epi8(g, b)
            );

            // Add alpha channel (255)
            rgba_lo = _mm256_or_si256(rgba_lo, _mm256_slli_epi32(const_255, 24));
            rgba_hi = _mm256_or_si256(rgba_hi, _mm256_slli_epi32(const_255, 24));

            // Store results
            _mm256_storeu_si256((__m256i *)out_line, rgba_lo);
            _mm256_storeu_si256((__m256i *)(out_line + 32), rgba_hi);

            in_line += 16;  // Move to next 8 YUYV pixels
            out_line += 32; // Move to next 8 RGBA pixels
        }
    }
}