#pragma once

#if __cplusplus < 202002L
#error "C++20 or later is required"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <execution>
#include <forward_list>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace cv
{

namespace utils
{
template <typename T> static inline constexpr T saturate_cast(T value) noexcept
{
    if (value > std::numeric_limits<T>::max())
    {
        return std::numeric_limits<T>::max();
    }
    else if (value < std::numeric_limits<T>::min())
    {
        return std::numeric_limits<T>::min();
    }
    return static_cast<T>(value);
}

} // namespace utils

namespace traits
{

template <typename T, typename P> static constexpr bool compare_kernel()
{
    return T::data_parallel > P::data_parallel;
}

template <size_t... I, typename... Types>
static constexpr bool compare_kernels_adjacent(std::index_sequence<I...>, std::tuple<Types...>)
{
    return (compare_kernel<std::tuple_element_t<I, std::tuple<Types...>>,
                           std::tuple_element_t<I + 1, std::tuple<Types...>>>() &&
            ...);
}

template <typename... Kernels> static constexpr bool is_kernels_convergent()
{
    if constexpr (sizeof...(Kernels) == 1)
    {
        return std::tuple_element_t<0, std::tuple<Kernels...>>::data_parallel == 1;
    }
    return compare_kernels_adjacent(std::make_index_sequence<sizeof...(Kernels) - 1>{}, std::tuple<Kernels...>{});
}

template <typename... Kernels> static constexpr size_t cal_kernels_product()
{
    size_t product = 1;
    ((product *= Kernels::data_parallel), ...);
    return product;
}

} // namespace traits

template <typename KernelType> class Kernel
{
  public:
    static constexpr std::string_view name = KernelType::name;
    static constexpr size_t data_parallel = KernelType::data_parallel;

    template <typename... Args> static inline constexpr void operator()(size_t index, Args &&...args) noexcept
    {
        return KernelType::operator()(index, std::forward<Args>(args)...);
    }
};

template <typename BackendType> class Backend
{
  public:
    static constexpr std::string_view name = BackendType::name;

    template <typename... KernelTypes, typename... Args>
    static inline constexpr bool operator()(size_t size, Args &&...args)
    {
        return BackendType::template operator()<KernelTypes...>(size, std::forward<Args>(args)...);
    }
};

namespace kernels
{

template <typename ArtihmeticType = float> class RGB2HSVPIL final : public Kernel<RGB2HSVPIL<ArtihmeticType>>
{
  public:
    static constexpr std::string_view name = "RGB2HSVPIL";
    static constexpr size_t data_parallel = 1;

    static inline constexpr void operator()(size_t index, const std::span<const uint8_t> &in_r,
                                            const std::span<const uint8_t> &in_g, const std::span<const uint8_t> &in_b,
                                            const std::span<uint8_t> &out_h, const std::span<uint8_t> &out_s,
                                            const std::span<uint8_t> &out_v) noexcept
    {
        using T = ArtihmeticType;

        const uint8_t r = in_r[index];
        const uint8_t g = in_g[index];
        const uint8_t b = in_b[index];

        const uint8_t maxc = std::max(r, std::max(g, b));
        const uint8_t minc = std::min(r, std::min(g, b));

        T h, s, rc, gc, bc, cr;
        uint8_t uh, us, uv;

        uv = maxc;

        if (minc == maxc)
        {
            uh = 0;
            us = 0;
        }
        else
        {
            cr = static_cast<T>(maxc) - static_cast<T>(minc);
            s = cr / static_cast<T>(maxc);

            rc = static_cast<T>(maxc - r) / cr;
            gc = static_cast<T>(maxc - g) / cr;
            bc = static_cast<T>(maxc - b) / cr;

            if (r == maxc)
            {
                h = bc - gc;
            }
            else if (g == maxc)
            {
                h = static_cast<T>(2.0) + rc - bc;
            }
            else
            {
                h = static_cast<T>(4.0) + gc - rc;
            }

            h = std::fmod((h / static_cast<T>(6.0)) + static_cast<T>(1.0), static_cast<T>(1.0));

            uh = std::clamp(static_cast<uint8_t>(h * static_cast<T>(255.0)), static_cast<uint8_t>(0),
                            static_cast<uint8_t>(255));
            us = std::clamp(static_cast<uint8_t>(s * static_cast<T>(255.0)), static_cast<uint8_t>(0),
                            static_cast<uint8_t>(255));
        }

        out_h[index] = uh;
        out_s[index] = us;
        out_v[index] = uv;
    }
};

template <size_t HRangeMax = 180> class RGB2HSVCV final : public Kernel<RGB2HSVCV<HRangeMax>>
{
  public:
    static constexpr std::string_view name = "RGB2HSVCV";
    static constexpr size_t data_parallel = 1;

    static inline constexpr void operator()(size_t index, const std::span<const uint8_t> &in_r,
                                            const std::span<const uint8_t> &in_g, const std::span<const uint8_t> &in_b,
                                            const std::span<uint8_t> &out_h, const std::span<uint8_t> &out_s,
                                            const std::span<uint8_t> &out_v) noexcept
    {
        const uint8_t r = in_r[index];
        const uint8_t g = in_g[index];
        const uint8_t b = in_b[index];

        int h, s, v = b;
        int vmin = b, diff;
        int vr, vg;

        v = std::max(static_cast<uint8_t>(v), g);
        v = std::max(static_cast<uint8_t>(v), r);
        vmin = std::min(static_cast<uint8_t>(vmin), g);
        vmin = std::min(static_cast<uint8_t>(vmin), r);

        diff = v - vmin;
        vr = v == r ? -1 : 0;
        vg = v == g ? -1 : 0;

        s = diff * s_div_table[v] >> hsv_shift;
        h = (vr & (g - b)) + (~vr & ((vg & (b - r + (2 * diff))) + ((~vg) & (r - g + (4 * diff)))));
        h = (h * h_div_table[diff] * h_scale + (1 << (hsv_shift + 6))) >> (7 + hsv_shift);
        h += h < 0 ? h_range_max : 0;

        out_h[index] = static_cast<uint8_t>(h);
        out_s[index] = static_cast<uint8_t>(s);
        out_v[index] = static_cast<uint8_t>(v);
    }

  private:
    static constexpr int h_range_max = HRangeMax;
    static_assert(h_range_max == 180 || h_range_max == 256);
    static constexpr int h_scale = h_range_max == 180 ? 15 : 21;
    static constexpr int hsv_shift = 12;

    static constexpr auto h_div_table{[]() noexcept {
        std::array<int, 256> table;
        table[0] = 0;
        for (size_t i = 1; i < table.size(); ++i)
        {
            table[i] = utils::saturate_cast<int>((h_range_max << hsv_shift) / (6. * i));
        }
        return table;
    }()};
    static constexpr auto s_div_table{[]() noexcept {
        std::array<int, 256> table;
        table[0] = 0;
        for (size_t i = 1; i < table.size(); ++i)
        {
            table[i] = utils::saturate_cast<int>((255 << hsv_shift) / (1. * i));
        }
        return table;
    }()};
};

#if defined(__ARM_NEON) && (defined(__ARM_64BIT_STATE) || defined(_M_ARM64))

template <size_t HRangeMax = 180> class RGB2HSVCVNeonAAarch64 final : public Kernel<RGB2HSVCVNeonAAarch64<HRangeMax>>
{
  public:
    static constexpr std::string_view name = "RGB2HSVCVNeonAAarch64";
    static constexpr size_t data_parallel = 16;

    static inline constexpr void operator()(size_t index, const std::span<const uint8_t> &in_r,
                                            const std::span<const uint8_t> &in_g, const std::span<const uint8_t> &in_b,
                                            const std::span<uint8_t> &out_h, const std::span<uint8_t> &out_s,
                                            const std::span<uint8_t> &out_v) noexcept
    {
        const uint8x16_t r = vld1q_u8(&in_r[index]);
        const uint8x16_t g = vld1q_u8(&in_g[index]);
        const uint8x16_t b = vld1q_u8(&in_b[index]);

        uint8x16_t h, s, v;
        uint8x16_t vmin;

        v = vmaxq_u8(b, vmaxq_u8(g, r));
        vmin = vminq_u8(b, vminq_u8(g, r));

        uint8x16_t diff, vr, vg;
        diff = vsubq_u8(v, vmin);
        uint8x16_t v255 = vdupq_n_u8(255), vz = vdupq_n_u8(0);
        vr = vbslq_u8(vceqq_u8(v, r), v255, vz);
        vg = vbslq_u8(vceqq_u8(v, g), v255, vz);

        int32x4_t sdiv0, sdiv1, sdiv2, sdiv3;
        uint16x8_t vd0, vd1;
        v_expand(v, vd0, vd1);
        int32x4_t vq0, vq1, vq2, vq3;
        v_expand(vd0, vq0, vq1);
        v_expand(vd1, vq2, vq3);
        {
            alignas(32) int32_t storevq[16];
            vst1q_s32(storevq, vq0);
            vst1q_s32(storevq + 4, vq1);
            vst1q_s32(storevq + 8, vq2);
            vst1q_s32(storevq + 12, vq3);

            v_lut(sdiv0, s_div_table, storevq);
            v_lut(sdiv1, s_div_table, storevq + 4);
            v_lut(sdiv2, s_div_table, storevq + 8);
            v_lut(sdiv3, s_div_table, storevq + 12);
        }

        int32x4_t hdiv0, hdiv1, hdiv2, hdiv3;
        uint16x8_t diffd0, diffd1;
        v_expand(diff, diffd0, diffd1);
        int32x4_t diffq0, diffq1, diffq2, diffq3;
        v_expand(diffd0, diffq0, diffq1);
        v_expand(diffd1, diffq2, diffq3);
        {
            alignas(32) int32_t storediffq[16];
            vst1q_s32(storediffq, diffq0);
            vst1q_s32(storediffq + 4, diffq1);
            vst1q_s32(storediffq + 8, diffq2);
            vst1q_s32(storediffq + 12, diffq3);

            v_lut(hdiv0, h_div_table, storediffq);
            v_lut(hdiv1, h_div_table, storediffq + 4);
            v_lut(hdiv2, h_div_table, storediffq + 8);
            v_lut(hdiv3, h_div_table, storediffq + 12);
        }

        int32x4_t sq0, sq1, sq2, sq3;
        int32x4_t vdescale = vdupq_n_s32(1 << (hsv_shift - 1));
        v_shr<hsv_shift>(sq0, vaddq_s32(vmulq_s32(diffq0, sdiv0), vdescale));
        v_shr<hsv_shift>(sq1, vaddq_s32(vmulq_s32(diffq1, sdiv1), vdescale));
        v_shr<hsv_shift>(sq2, vaddq_s32(vmulq_s32(diffq2, sdiv2), vdescale));
        v_shr<hsv_shift>(sq3, vaddq_s32(vmulq_s32(diffq3, sdiv3), vdescale));
        int16x8_t sd0, sd1;
        v_pack(sd0, sq0, sq1);
        v_pack(sd1, sq2, sq3);
        v_pack(s, sd0, sd1);

        uint16x8_t bdu0, bdu1, gdu0, gdu1, rdu0, rdu1;
        v_expand(b, bdu0, bdu1);
        v_expand(g, gdu0, gdu1);
        v_expand(r, rdu0, rdu1);
        int16x8_t bd0, bd1, gd0, gd1, rd0, rd1;
        bd0 = vreinterpretq_s16_u16(bdu0);
        bd1 = vreinterpretq_s16_u16(bdu1);
        gd0 = vreinterpretq_s16_u16(gdu0);
        gd1 = vreinterpretq_s16_u16(gdu1);
        rd0 = vreinterpretq_s16_u16(rdu0);
        rd1 = vreinterpretq_s16_u16(rdu1);

        int16x8_t vrd0, vrd1, vgd0, vgd1;
        v_expand(vr, vrd0, vrd1);
        v_expand(vg, vgd0, vgd1);
        int16x8_t diffsd0, diffsd1;
        diffsd0 = vreinterpretq_s16_u16(diffd0);
        diffsd1 = vreinterpretq_s16_u16(diffd1);

        int16x8_t hd0, hd1;
        int16x8_t gb = vsubq_s16(gd0, bd0);
        int16x8_t br = vaddq_s16(vsubq_s16(bd0, rd0), vshlq_n_s16(diffsd0, 1));
        int16x8_t rg = vaddq_s16(vsubq_s16(rd0, gd0), vshlq_n_s16(diffsd0, 2));
        hd0 = vaddq_s16(vandq_s16(vrd0, gb),
                        vandq_s16(vmvnq_s16(vrd0), vaddq_s16(vandq_s16(vgd0, br), vandq_s16(vmvnq_s16(vgd0), rg))));
        gb = vsubq_s16(gd1, bd1);
        br = vaddq_s16(vsubq_s16(bd1, rd1), vshlq_n_s16(diffsd1, 1));
        rg = vaddq_s16(vsubq_s16(rd1, gd1), vshlq_n_s16(diffsd1, 2));
        hd1 = vaddq_s16(vandq_s16(vrd1, gb),
                        vandq_s16(vmvnq_s16(vrd1), vaddq_s16(vandq_s16(vgd1, br), vandq_s16(vmvnq_s16(vgd1), rg))));

        int32x4_t hq0, hq1, hq2, hq3;
        v_expand(hd0, hq0, hq1);
        v_expand(hd1, hq2, hq3);
        v_shr<hsv_shift>(hq0, vaddq_s32(vmulq_s32(hq0, hdiv0), vdescale));
        v_shr<hsv_shift>(hq1, vaddq_s32(vmulq_s32(hq1, hdiv1), vdescale));
        v_shr<hsv_shift>(hq2, vaddq_s32(vmulq_s32(hq2, hdiv2), vdescale));
        v_shr<hsv_shift>(hq3, vaddq_s32(vmulq_s32(hq3, hdiv3), vdescale));

        v_pack(hd0, hq0, hq1);
        v_pack(hd1, hq2, hq3);
        int16x8_t vhr = vdupq_n_s16(h_range_max);
        int16x8_t vzd = vdupq_n_s16(0);
        hd0 = vaddq_s16(hd0, vbslq_s16(vcltq_s16(hd0, vzd), vhr, vzd));
        hd1 = vaddq_s16(hd1, vbslq_s16(vcltq_s16(hd1, vzd), vhr, vzd));
        v_pack(h, hd0, hd1);

        vst1q_u8(&out_h[index], h);
        vst1q_u8(&out_s[index], s);
        vst1q_u8(&out_v[index], v);
    }

  private:
    static inline constexpr void v_expand(const uint8x16_t &src, uint16x8_t &low, uint16x8_t &high) noexcept
    {
        low = vmovl_u8(vget_low_u8(src));
        high = vmovl_u8(vget_high_u8(src));
    }

    static inline constexpr void v_expand(const uint8x16_t &src, int16x8_t &low, int16x8_t &high) noexcept
    {
        low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src)));
        high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src)));
    }

    static inline constexpr void v_expand(const uint16x8_t &src, int32x4_t &low, int32x4_t &high) noexcept
    {
        low = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(src)));
        high = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(src)));
    }

    static inline constexpr void v_expand(const int16x8_t &src, int32x4_t &low, int32x4_t &high) noexcept
    {
        low = vmovl_s16(vget_low_s16(src));
        high = vmovl_s16(vget_high_s16(src));
    }

    static inline constexpr void v_lut(int32x4_t &store, const std::array<int, 256> &tab, const int *idx) noexcept
    {
        alignas(16) int32_t elems[4]{tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]};
        store = vld1q_s32(elems);
    }

    template <size_t N> static inline constexpr void v_shr(int32x4_t &store, const int32x4_t &val) noexcept
    {
        store = vshrq_n_s32(val, N);
    }

    static inline constexpr void v_pack(int16x8_t &store, const int32x4_t &low, const int32x4_t &high) noexcept
    {
        store = vcombine_s16(vqmovn_s32(low), vqmovn_s32(high));
    }

    static inline constexpr void v_pack(uint8x16_t &store, const int16x8_t &low, const int16x8_t &high) noexcept
    {
        store = vcombine_u8(vqmovun_s16(low), vqmovun_s16(high));
    }

  private:
    static constexpr int h_range_max = HRangeMax;
    static_assert(h_range_max == 180 || h_range_max == 256);
    static constexpr int h_scale = h_range_max == 180 ? 15 : 21;
    static constexpr int hsv_shift = 12;

    static constexpr auto h_div_table{[]() noexcept {
        std::array<int, 256> table;
        table[0] = 0;
        for (size_t i = 1; i < table.size(); ++i)
        {
            table[i] = utils::saturate_cast<int>((h_range_max << hsv_shift) / (6. * i));
        }
        return table;
    }()};
    static constexpr auto s_div_table{[]() noexcept {
        std::array<int, 256> table;
        table[0] = 0;
        for (size_t i = 1; i < table.size(); ++i)
        {
            table[i] = utils::saturate_cast<int>((255 << hsv_shift) / (1. * i));
        }
        return table;
    }()};
};

#endif

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX__) || defined(__AVX2__))

template <size_t HRangeMax = 180> class RGB2HSVCVAVX2 final : public Kernel<RGB2HSVCVAVX2<HRangeMax>>
{
  public:
    static constexpr std::string_view name = "RGB2HSVCVAVX2";
    static constexpr size_t data_parallel = 32;

    static inline constexpr void operator()(size_t index, const std::span<const uint8_t> &in_r,
                                            const std::span<const uint8_t> &in_g, const std::span<const uint8_t> &in_b,
                                            const std::span<uint8_t> &out_h, const std::span<uint8_t> &out_s,
                                            const std::span<uint8_t> &out_v) noexcept
    {
        const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&in_r[index]));
        const __m256i g = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&in_g[index]));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&in_b[index]));

        __m256i h, s, v;
        __m256i vmin;

        v = _mm256_max_epu8(b, _mm256_max_epu8(g, r));
        vmin = _mm256_min_epu8(b, _mm256_min_epu8(g, r));

        __m256i diff, vr, vg;
        diff = _mm256_subs_epu8(v, vmin);
        __m256i v255 = _mm256_set1_epi8(255), vz = _mm256_setzero_si256();
        vr = _mm256_blendv_epi8(v255, vz, _mm256_cmpeq_epi8(v, r));
        vg = _mm256_blendv_epi8(v255, vz, _mm256_cmpeq_epi8(v, g));

        __m256i sdiv0, sdiv1, sdiv2, sdiv3;
        __m256i vd0, vd1;
        v_expand_u8x32_u16x16(v, vd0, vd1);
        __m256i vq0, vq1, vq2, vq3;
        v_expand_u16x16_i32x8(vd0, vq0, vq1);
        v_expand_u16x16_i32x8(vd1, vq2, vq3);
        {
            alignas(32) int32_t storevq[32];
            _mm256_store_si256(reinterpret_cast<__m256i *>(storevq), vq0);
            _mm256_store_si256(reinterpret_cast<__m256i *>(storevq + 8), vq1);
            _mm256_store_si256(reinterpret_cast<__m256i *>(storevq + 16), vq2);
            _mm256_store_si256(reinterpret_cast<__m256i *>(storevq + 24), vq3);

            v_lut_i32x8(sdiv0, s_div_table, storevq);
            v_lut_i32x8(sdiv1, s_div_table, storevq + 8);
            v_lut_i32x8(sdiv2, s_div_table, storevq + 16);
            v_lut_i32x8(sdiv3, s_div_table, storevq + 24);
        }

        __m256i hdiv0, hdiv1, hdiv2, hdiv3;
        __m256i diffd0, diffd1;
        v_expand_u8x32_u16x16(diff, diffd0, diffd1);
        __m256i diffq0, diffq1, diffq2, diffq3;
        v_expand_u16x16_i32x8(diffd0, diffq0, diffq1);
        v_expand_u16x16_i32x8(diffd1, diffq2, diffq3);
        {
            alignas(32) int32_t storediffq[32];
            _mm256_store_si256(reinterpret_cast<__m256i *>(storediffq), diffq0);
            _mm256_store_si256(reinterpret_cast<__m256i *>(storediffq + 8), diffq1);
            _mm256_store_si256(reinterpret_cast<__m256i *>(storediffq + 16), diffq2);
            _mm256_store_si256(reinterpret_cast<__m256i *>(storediffq + 24), diffq3);

            v_lut_i32x8(hdiv0, h_div_table, storediffq);
            v_lut_i32x8(hdiv1, h_div_table, storediffq + 8);
            v_lut_i32x8(hdiv2, h_div_table, storediffq + 16);
            v_lut_i32x8(hdiv3, h_div_table, storediffq + 24);
        }

        __m256i sq0, sq1, sq2, sq3;
        __m256i vdescale = _mm256_set1_epi32(1 << (hsv_shift - 1));
        v_shr_i32x8<hsv_shift>(sq0, _mm256_add_epi32(_mm256_mullo_epi32(diffq0, sdiv0), vdescale));
        v_shr_i32x8<hsv_shift>(sq1, _mm256_add_epi32(_mm256_mullo_epi32(diffq1, sdiv1), vdescale));
        v_shr_i32x8<hsv_shift>(sq2, _mm256_add_epi32(_mm256_mullo_epi32(diffq2, sdiv2), vdescale));
        v_shr_i32x8<hsv_shift>(sq3, _mm256_add_epi32(_mm256_mullo_epi32(diffq3, sdiv3), vdescale));
        __m256i sd0, sd1;
        v_pack_i16x16_i32x8(sd0, sq0, sq1);
        v_pack_i16x16_i32x8(sd1, sq2, sq3);
        v_pack_u8x32_i16x16(s, sd0, sd1);

        __m256i bdu0, bdu1, gdu0, gdu1, rdu0, rdu1;
        v_expand_u8x32_u16x16(b, bdu0, bdu1);
        v_expand_u8x32_u16x16(g, gdu0, gdu1);
        v_expand_u8x32_u16x16(r, rdu0, rdu1);
        __m256i bd0, bd1, gd0, gd1, rd0, rd1;
        bd0 = bdu0;
        bd1 = bdu1;
        gd0 = gdu0;
        gd1 = gdu1;
        rd0 = rdu0;
        rd1 = rdu1;

        __m256i vrd0, vrd1, vgd0, vgd1;
        v_expand_u8x32_u16x16(vr, vrd0, vrd1);
        v_expand_u8x32_u16x16(vg, vgd0, vgd1);
        __m256i diffsd0, diffsd1;
        diffsd0 = diffd0;
        diffsd1 = diffd1;

        __m256i hd0, hd1;
        __m256i gb = _mm256_sub_epi16(gd0, bd0);
        __m256i br = _mm256_add_epi16(_mm256_sub_epi16(bd0, rd0), _mm256_slli_epi16(diffsd0, 1));
        __m256i rg = _mm256_add_epi16(_mm256_sub_epi16(rd0, gd0), _mm256_slli_epi16(diffsd0, 2));
        hd0 = _mm256_add_epi16(
            _mm256_and_si256(vrd0, gb),
            _mm256_and_si256(_mm256_xor_si256(vrd0, _mm256_set1_epi32(-1)),
                             _mm256_add_epi16(_mm256_and_si256(vgd0, br),
                                              _mm256_and_si256(_mm256_xor_si256(vgd0, _mm256_set1_epi32(-1)), rg))));
        gb = _mm256_sub_epi16(gd1, bd1);
        br = _mm256_add_epi16(_mm256_sub_epi16(bd1, rd1), _mm256_slli_epi16(diffsd1, 1));
        rg = _mm256_add_epi16(_mm256_sub_epi16(rd1, gd1), _mm256_slli_epi16(diffsd1, 2));
        hd1 = _mm256_add_epi16(
            _mm256_and_si256(vrd1, gb),
            _mm256_and_si256(_mm256_xor_si256(vrd1, _mm256_set1_epi32(-1)),
                             _mm256_add_epi16(_mm256_and_si256(vgd1, br),
                                              _mm256_and_si256(_mm256_xor_si256(vgd1, _mm256_set1_epi32(-1)), rg))));

        __m256i hq0, hq1, hq2, hq3;
        v_expand_i16x16_i32x8(hd0, hq0, hq1);
        v_expand_i16x16_i32x8(hd1, hq2, hq3);
        v_shr_i32x8<hsv_shift>(hq0, _mm256_add_epi32(_mm256_mullo_epi32(hq0, hdiv0), vdescale));
        v_shr_i32x8<hsv_shift>(hq1, _mm256_add_epi32(_mm256_mullo_epi32(hq1, hdiv1), vdescale));
        v_shr_i32x8<hsv_shift>(hq2, _mm256_add_epi32(_mm256_mullo_epi32(hq2, hdiv2), vdescale));
        v_shr_i32x8<hsv_shift>(hq3, _mm256_add_epi32(_mm256_mullo_epi32(hq3, hdiv3), vdescale));

        v_pack_i16x16_i32x8(hd0, hq0, hq1);
        v_pack_i16x16_i32x8(hd1, hq2, hq3);
        __m256i vhr = _mm256_set1_epi16(h_range_max);
        __m256i vzd = _mm256_setzero_si256();
        hd0 = _mm256_add_epi16(hd0, _mm256_blendv_epi8(vhr, vzd, _mm256_cmpgt_epi16(vzd, hd0)));
        hd1 = _mm256_add_epi16(hd1, _mm256_blendv_epi8(vhr, vzd, _mm256_cmpgt_epi16(vzd, hd1)));
        v_pack_u8x32_i16x16(h, hd0, hd1);

        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&out_h[index]), h);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&out_s[index]), s);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&out_v[index]), v);
    }

  private:
    static inline constexpr void v_expand_u8x32_u16x16(const __m256i &src, __m256i &low, __m256i &high) noexcept
    {
        low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(src));
        high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(src, 1));
    }

    static inline constexpr void v_expand_u16x16_i32x8(const __m256i &src, __m256i &low, __m256i &high) noexcept
    {
        low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src));
        high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src, 1));
    }

    static inline constexpr void v_expand_i16x16_i32x8(const __m256i &src, __m256i &low, __m256i &high) noexcept
    {
        low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(src));
        high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(src, 1));
    }

    static inline constexpr void v_lut_i32x8(__m256i &store, const std::array<int, 256> &tab, const int *idx) noexcept
    {
        store = _mm256_i32gather_epi32(tab.data(), _mm256_load_si256(reinterpret_cast<const __m256i *>(idx)), 4);
    }

    template <size_t N> static inline constexpr void v_shr_i32x8(__m256i &store, const __m256i &val) noexcept
    {
        store = _mm256_srai_epi32(val, N);
    }

    static inline constexpr void v_pack_i16x16_i32x8(__m256i &store, const __m256i &low, const __m256i &high) noexcept
    {
        store = _mm256_permute4x64_epi64(_mm256_packs_epi32(low, high), _MM_SHUFFLE(3, 1, 2, 0));
    }

    static inline constexpr void v_pack_u8x32_i16x16(__m256i &store, const __m256i &low, const __m256i &high) noexcept
    {
        store = _mm256_permute4x64_epi64(_mm256_packus_epi16(low, high), _MM_SHUFFLE(3, 1, 2, 0));
    }

  private:
    static constexpr int h_range_max = HRangeMax;
    static_assert(h_range_max == 180 || h_range_max == 256);
    static constexpr int h_scale = h_range_max == 180 ? 15 : 21;
    static constexpr int hsv_shift = 12;

    static constexpr auto h_div_table{[]() noexcept {
        std::array<int, 256> table;
        table[0] = 0;
        for (size_t i = 1; i < table.size(); ++i)
        {
            table[i] = utils::saturate_cast<int>((h_range_max << hsv_shift) / (6. * i));
        }
        return table;
    }()};
    static constexpr auto s_div_table{[]() noexcept {
        std::array<int, 256> table;
        table[0] = 0;
        for (size_t i = 1; i < table.size(); ++i)
        {
            table[i] = utils::saturate_cast<int>((255 << hsv_shift) / (1. * i));
        }
        return table;
    }()};
};

#endif

} // namespace kernels

namespace backends
{

class Sequential final : public Backend<Sequential>
{
  public:
    static constexpr std::string_view name = "Sequential";

    template <typename... KernelTypes, typename... Args>
    static inline constexpr size_t operator()(size_t start, const size_t size, Args &&...args) noexcept
    {
        const size_t end = start + size;
        (dispatch<KernelTypes>(start, end, std::forward<Args>(args)...), ...);
        return end - start;
    }

  private:
    template <typename KernelType, typename... Args>
    static inline constexpr void dispatch(size_t &start, const size_t end, Args &&...args) noexcept
    {
        constexpr size_t data_parallel = KernelType::data_parallel;
        while (start + data_parallel <= end)
        {
            KernelType::operator()(start, std::forward<Args>(args)...);
            start += data_parallel;
        }
    }
};

class STLParallel final : public Backend<STLParallel>
{
  public:
    static constexpr std::string_view name = "STLParallel";

    template <typename... KernelTypes, typename... Args>
    static inline constexpr size_t operator()(const size_t start, const size_t size, Args &&...args)
    {
        constexpr size_t seq_data_parallel = traits::cal_kernels_product<KernelTypes...>();
        const size_t remain = size % seq_data_parallel;
        const size_t view_end = start + size - remain;
        auto par_data_chunk_iter = std::ranges::iota_view(start, view_end);
        size_t unprocessed = std::transform_reduce(std::execution::par_unseq, par_data_chunk_iter.begin(),
                                                   par_data_chunk_iter.end(), size_t{0}, ::std::plus<size_t>{},
                                                   [seq_data_parallel, &args...](const auto i) noexcept {
                                                       return Sequential::template operator()<KernelTypes...>(
                                                           i, seq_data_parallel, std::forward<Args>(args)...);
                                                   });
        if (remain > 0)
        {
            unprocessed +=
                Sequential::template operator()<KernelTypes...>(view_end, remain, std::forward<Args>(args)...);
        }
        return unprocessed;
    }
};

class STLThreads final : public Backend<STLThreads>
{
  public:
    static constexpr std::string_view name = "STLThreads";
    static size_t concurrency_limit;

    template <typename... KernelTypes, typename... Args>
    static inline constexpr size_t operator()(const size_t start, const size_t size, Args &&...args)
    {
        size_t concurrency = concurrency_limit;
        if (concurrency == 0)
        {
            if (concurrency = std::thread::hardware_concurrency(); concurrency == 0) [[unlikely]]
            {
                return size;
            }
        }
        const size_t seq_data_parallel = traits::cal_kernels_product<KernelTypes...>();
        const size_t parallel_units = size / seq_data_parallel;
        const size_t workloads = std::ceil(static_cast<float>(parallel_units) / static_cast<float>(concurrency));
        const size_t chunk_size = parallel_units * workloads;
        std::atomic<size_t> unprocessed = 0;
        std::forward_list<std::thread> threads;
        size_t remain = size;
        while (remain > chunk_size)
        {
            remain -= chunk_size;
            threads.emplace_front([start, remain, chunk_size, &unprocessed, &args...]() noexcept {
                unprocessed.fetch_add(Sequential::template operator()<KernelTypes...>(start + remain, chunk_size,
                                                                                      std::forward<Args>(args)...));
            });
        }
        if (remain > 0)
        {
            unprocessed.fetch_add(
                Sequential::template operator()<KernelTypes...>(start, remain, std::forward<Args>(args)...));
        }
        for (auto &thread : threads)
        {
            thread.join();
        }
        return unprocessed.load();
    }
};

size_t STLThreads::concurrency_limit = std::thread::hardware_concurrency();

} // namespace backends

namespace tasks
{

namespace details
{

template <typename BackendType, typename... KernelTypes, size_t... Indices>
static inline constexpr bool cvtColor(std::span<const uint8_t> in, std::span<uint8_t> out,
                                      std::index_sequence<Indices...>) noexcept
{
    constexpr size_t channels = sizeof...(Indices);

    const auto in_size = in.size();
    const auto out_size = out.size();
    if (in_size != out_size) [[unlikely]]
    {
        return false;
    }
    if (in_size % channels != 0) [[unlikely]]
    {
        return false;
    }

    // Channels,H,W
    const size_t layer_size = in_size / channels;

    return BackendType::template operator()<KernelTypes...>(0, layer_size,
                                                            in.subspan(Indices * layer_size, layer_size)...,
                                                            out.subspan(Indices * layer_size, layer_size)...) == 0;
}

} // namespace details

template <size_t Channels, typename BackendType, typename... KernelTypes>
static inline constexpr bool cvtColor(std::span<const uint8_t> in, std::span<uint8_t> out) noexcept
{
    static_assert(
        traits::is_kernels_convergent<KernelTypes...>(),
        "All kernels must have a data_parallel value that converges to 1 for the last kernel in the sequence");

    return details::cvtColor<BackendType, KernelTypes...>(std::move(in), std::move(out),
                                                          std::make_index_sequence<Channels>{});
}

} // namespace tasks

} // namespace cv
