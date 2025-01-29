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

    // Constants for YUV to RGB conversion
    __m256i const_298 = _mm256_set1_epi16(298);
    __m256i const_409 = _mm256_set1_epi16(409);
    __m256i const_100 = _mm256_set1_epi16(100);
    __m256i const_208 = _mm256_set1_epi16(208);
    __m256i const_516 = _mm256_set1_epi16(516);
    __m256i const_128 = _mm256_set1_epi16(128);

    for (int y = 0; y < height; y++)
    {
        uint8_t *in_line = yuyv + y * istride;
        uint8_t *out_line = rgba + y * row_size;

        for (int x = 0; x < width; x += 16)
        {
            // Load 16 YUYV bytes (8 pixels)
            __m256i yuyv_data = _mm256_loadu_si256((__m256i *)in_line);

            // Unpack Y, U, V components
            __m256i y_values = _mm256_and_si256(yuyv_data, _mm256_set1_epi16(0x00FF)); // Extract Y
            __m256i uv_packed = _mm256_srli_epi16(yuyv_data, 8);                      // Extract interleaved U, V
            __m256i u_values = _mm256_and_si256(uv_packed, _mm256_set1_epi16(0x00FF));
            __m256i v_values = _mm256_srli_epi16(uv_packed, 8);

            // Center UV values around zero
            u_values = _mm256_sub_epi16(u_values, const_128);
            v_values = _mm256_sub_epi16(v_values, const_128);

            // Calculate RGB for each Y
            __m256i r = _mm256_srai_epi16(
                _mm256_add_epi16(
                    _mm256_mullo_epi16(const_298, y_values),
                    _mm256_mullo_epi16(const_409, v_values)),
                8);

            __m256i g = _mm256_srai_epi16(
                _mm256_sub_epi16(
                    _mm256_sub_epi16(
                        _mm256_mullo_epi16(const_298, y_values),
                        _mm256_mullo_epi16(const_100, u_values)),
                    _mm256_mullo_epi16(const_208, v_values)),
                8);

            __m256i b = _mm256_srai_epi16(
                _mm256_add_epi16(
                    _mm256_mullo_epi16(const_298, y_values),
                    _mm256_mullo_epi16(const_516, u_values)),
                8);

            // Pack and saturate
            __m256i r_clamped = _mm256_max_epi16(_mm256_min_epi16(r, _mm256_set1_epi16(255)), _mm256_set1_epi16(0));
            __m256i g_clamped = _mm256_max_epi16(_mm256_min_epi16(g, _mm256_set1_epi16(255)), _mm256_set1_epi16(0));
            __m256i b_clamped = _mm256_max_epi16(_mm256_min_epi16(b, _mm256_set1_epi16(255)), _mm256_set1_epi16(0));

            // Interleave RGBA components
            __m256i rgba_packed = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_slli_epi16(r_clamped, 0),
                    _mm256_slli_epi16(g_clamped, 8)),
                _mm256_slli_epi16(b_clamped, 16));

            // Store packed RGBA results
            _mm256_storeu_si256((__m256i *)out_line, rgba_packed);

            in_line += 32;
            out_line += 64;
        }
    }
}
