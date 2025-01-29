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
    
    // Constants matching NEON implementation
    __m256i const_16 = _mm256_set1_epi8(16);                    // Y shift
    __m256i const_128_16 = _mm256_set1_epi16(128);             // UV shift
    __m256i const_298 = _mm256_set1_epi32(298);                // Y multiplier
    __m256i const_409 = _mm256_set1_epi32(409);                // V multiplier for R
    __m256i const_208 = _mm256_set1_epi32(208);                // V multiplier for G
    __m256i const_100 = _mm256_set1_epi32(100);                // U multiplier for G
    __m256i const_516 = _mm256_set1_epi32(516);                // U multiplier for B
    __m256i const_128_32 = _mm256_set1_epi32(128);             // Rounding
    __m256i const_255 = _mm256_set1_epi8(255);                 // Alpha value

    for (int y = 0; y < height; y += 2)
    {
        uint8_t *in_line = yuyv + y * istride;
        uint8_t *out_line = rgba + y * row_size;

        for (int x = 0; x < width; x += 8)
        {
            // Load Y values for two rows
            __m256i y_values_row1 = _mm256_loadu_si256((__m256i *)in_line);
            __m256i y_values_row2 = _mm256_loadu_si256((__m256i *)(in_line + istride));
            
            // Extract Y values (every other byte)
            __m256i y0 = _mm256_and_si256(y_values_row1, _mm256_set1_epi16(0x00FF));
            __m256i y1 = _mm256_and_si256(y_values_row2, _mm256_set1_epi16(0x00FF));
            
            // Shift Y range
            y0 = _mm256_subs_epu8(y0, const_16);
            y1 = _mm256_subs_epu8(y1, const_16);
            
            // Convert Y to 32-bit and multiply by 298
            __m256i y0_32_low = _mm256_mullo_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(y0, 0)), const_298);
            __m256i y0_32_high = _mm256_mullo_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(y0, 1)), const_298);
            __m256i y1_32_low = _mm256_mullo_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(y1, 0)), const_298);
            __m256i y1_32_high = _mm256_mullo_epi32(
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(y1, 1)), const_298);

            // Load and process UV values
            __m256i uv_values = _mm256_loadu_si256((__m256i *)(in_line + 1));
            __m256i u_values = _mm256_and_si256(_mm256_srli_epi16(uv_values, 8), _mm256_set1_epi16(0x00FF));
            __m256i v_values = _mm256_and_si256(uv_values, _mm256_set1_epi16(0x00FF));
            
            // Center UV around zero
            u_values = _mm256_sub_epi16(u_values, const_128_16);
            v_values = _mm256_sub_epi16(v_values, const_128_16);

            // Convert UV to 32-bit
            __m256i u_32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u_values, 0));
            __m256i v_32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_values, 0));

            // Calculate color components
            __m256i r_temp = _mm256_add_epi32(const_128_32, _mm256_mullo_epi32(v_32, const_409));
            __m256i g_temp = _mm256_sub_epi32(
                _mm256_sub_epi32(const_128_32, _mm256_mullo_epi32(u_32, const_100)),
                _mm256_mullo_epi32(v_32, const_208));
            __m256i b_temp = _mm256_add_epi32(const_128_32, _mm256_mullo_epi32(u_32, const_516));

            // Process first row
            __m256i r_row1 = _mm256_srai_epi32(_mm256_add_epi32(y0_32_low, r_temp), 8);
            __m256i g_row1 = _mm256_srai_epi32(_mm256_add_epi32(y0_32_low, g_temp), 8);
            __m256i b_row1 = _mm256_srai_epi32(_mm256_add_epi32(y0_32_low, b_temp), 8);

            // Pack and store first row
            __m256i rgb_row1 = _mm256_packus_epi32(
                _mm256_packus_epi32(r_row1, g_row1),
                _mm256_packus_epi32(b_row1, _mm256_set1_epi32(255)));
            _mm256_storeu_si256((__m256i *)out_line, rgb_row1);

            // Process second row
            __m256i r_row2 = _mm256_srai_epi32(_mm256_add_epi32(y1_32_low, r_temp), 8);
            __m256i g_row2 = _mm256_srai_epi32(_mm256_add_epi32(y1_32_low, g_temp), 8);
            __m256i b_row2 = _mm256_srai_epi32(_mm256_add_epi32(y1_32_low, b_temp), 8);

            // Pack and store second row
            __m256i rgb_row2 = _mm256_packus_epi32(
                _mm256_packus_epi32(r_row2, g_row2),
                _mm256_packus_epi32(b_row2, _mm256_set1_epi32(255)));
            _mm256_storeu_si256((__m256i *)(out_line + row_size), rgb_row2);

            in_line += 16;
            out_line += 32;
        }
    }
}