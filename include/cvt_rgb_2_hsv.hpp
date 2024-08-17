#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <execution>
#include <forward_list>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace cv
{

namespace traits
{

template <typename T, typename P> static constexpr bool compare_kernel()
{
    return T::data_parallel > P::data_parallel;
}

template <std::size_t... I, typename... Types>
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

template <typename ArtihmeticType = float> struct RGB2HSVPIL final : public Kernel<RGB2HSVPIL<ArtihmeticType>>
{
  public:
    constexpr static std::string_view name = "PIL";
    constexpr static size_t data_parallel = 1;

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

} // namespace kernels

namespace backends
{

class Squential final : public Backend<Squential>
{
  public:
    constexpr static std::string_view name = "Sequential";

    template <typename... KernelTypes, typename... Args>
    static inline constexpr bool operator()(size_t size, Args &&...args) noexcept
    {
        ((size = dispatch<KernelTypes>(size, std::forward<Args>(args)...)), ...);
        return size == 0;
    }

  protected:
    template <typename KernelType, typename... Args>
    static inline constexpr size_t dispatch(size_t remain, Args &&...args) noexcept
    {
        const size_t data_parallel = KernelType::data_parallel;
        while (remain >= data_parallel)
        {
            remain -= data_parallel;
            KernelType::operator()(remain, std::forward<Args>(args)...);
        }
        return remain;
    }
};

} // namespace backends

namespace tasks
{

namespace detail
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

    return BackendType::template operator()<KernelTypes...>(layer_size, in.subspan(Indices * layer_size, layer_size)...,
                                                            out.subspan(Indices * layer_size, layer_size)...);
}

} // namespace detail

template <size_t Channels, typename BackendType, typename... KernelTypes>
static inline constexpr bool cvtColor(std::span<const uint8_t> in, std::span<uint8_t> out) noexcept
{
    static_assert(
        traits::is_kernels_convergent<KernelTypes...>(),
        "All kernels must have a data_parallel value that converges to 1 for the last kernel in the sequence");

    return detail::cvtColor<BackendType, KernelTypes...>(in, out, std::make_index_sequence<Channels>{});
}

} // namespace tasks

template <size_t DataParallel = 1, typename ArtihmeticType = float> struct KernelPIL final
{
    constexpr static std::string_view name = "PIL";
    constexpr static size_t data_parallel = DataParallel;
    using fallback_kernel_type = KernelPIL<1, ArtihmeticType>;

    using artihmetic_type = ArtihmeticType;
};

template <typename KT,
          typename std::enable_if_t<std::is_same_v<KT, KernelPIL<1, float>> || std::is_same_v<KT, KernelPIL<1, double>>,
                                    bool> = true>
constexpr static inline void cvtRGB2HSVKernel(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t &out_h,
                                              uint8_t &out_s, uint8_t &out_v) noexcept
{
    using T = typename KT::artihmetic_type;

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

    out_h = uh;
    out_s = us;
    out_v = uv;
}

template <size_t DataParallel = 1, int HRangeMax = 180> struct KernelCV final
{
    constexpr static std::string_view name = "CV";
    constexpr static size_t data_parallel = DataParallel;
    using fallback_kernel_type = KernelCV<1, HRangeMax>;

    static constexpr int h_range_max = HRangeMax;
    static constexpr int h_scale = h_range_max == 180 ? 15 : 21;

    static constexpr auto div_table{[]() noexcept {
        std::array<int, 256> table;
        table[0] = 0;
        for (size_t i = 1; i < table.size(); i++)
        {
            table[i] = (1 << 14) / i;
        }
        return table;
    }()};
};

template <typename KT, typename std::enable_if_t<
                           std::is_same_v<KT, KernelCV<1, 180>> || std::is_same_v<KT, KernelCV<1, 256>>, bool> = true>
constexpr static inline void cvtRGB2HSVKernel(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t &out_h,
                                              uint8_t &out_s, uint8_t &out_v) noexcept
{
    constexpr int hsv_shift = 12;

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

    s = diff * KT::div_table[v] >> hsv_shift;
    h = (vr & (g - b)) + (~vr & ((vg & (b - r + (2 * diff))) + ((~vg) & (r - g + (4 * diff)))));
    h = (h * KT::div_table[diff] * KT::h_scale + (1 << (hsv_shift + 6))) >> (7 + hsv_shift);
    h += h < 0 ? KT::h_range_max : 0;

    out_h = static_cast<uint8_t>(h);
    out_s = static_cast<uint8_t>(s);
    out_v = static_cast<uint8_t>(v);
}

template <size_t DataParallel = 8> struct KernelNeon final
{
    constexpr static std::string_view name = "Neon";
    constexpr static size_t data_parallel = DataParallel;
    using fallback_kernel_type = KernelPIL<1, float>;
};

#include <arm_neon.h>

template <typename KT, typename std::enable_if_t<std::is_same_v<KT, KernelNeon<8>>, bool> = true>
constexpr static inline void cvtRGB2HSVKernel(const uint8_t &r, const uint8_t &g, const uint8_t &b, uint8_t &out_h,
                                              uint8_t &out_s, uint8_t &out_v) noexcept
{
    uint8x16_t r_vec = vld1q_u8(&r);
    uint8x16_t g_vec = vld1q_u8(&g);
    uint8x16_t b_vec = vld1q_u8(&b);

    uint8x16_t maxc = vmaxq_u8(r_vec, vmaxq_u8(g_vec, b_vec));
    uint8x16_t minc = vminq_u8(r_vec, vminq_u8(g_vec, b_vec));

    uint8x16_t uv = maxc;

    uint8x16_t eq = vceqq_u8(minc, maxc);
    uint8x16_t uh = vdupq_n_u8(0);
    uint8x16_t us = vdupq_n_u8(0);

    // TODO: Implement the rest of the kernel
}

struct BackendSequential final
{
    constexpr static std::string_view name = "Sequential";
};

struct BackendStdExecution final
{
    constexpr static std::string_view name = "StdExecution";
};

struct BackendThread final
{
    constexpr static std::string_view name = "Thread";
};

template <typename BackendType, typename KernelType, typename FallbackKernelType = KernelType::fallback_kernel_type>
constexpr static inline bool cvtRGB2HSV(std::span<const uint8_t> in, std::span<uint8_t> out) noexcept
{
    const auto in_size = in.size();
    const auto out_size = out.size();
    if (in_size != out_size) [[unlikely]]
    {
        return false;
    }
    if (in_size % 3 != 0) [[unlikely]]
    {
        return false;
    }

    // 3,H,W
    const size_t layer_size = in_size / 3;

    const auto r_span = in.subspan(0, layer_size);
    const auto g_span = in.subspan(layer_size, layer_size);
    const auto b_span = in.subspan(2 * layer_size, layer_size);

    auto h_span = out.subspan(0, layer_size);
    auto s_span = out.subspan(layer_size, layer_size);
    auto v_span = out.subspan(2 * layer_size, layer_size);

    static_assert(KernelType::data_parallel >= 1);
    static_assert(FallbackKernelType::data_parallel == 1);

    if constexpr (std::is_same_v<BackendType, BackendSequential>)
    {
        constexpr size_t data_parallel = KernelType::data_parallel;
        size_t remain = layer_size;
        while (remain >= data_parallel)
        {
            remain -= data_parallel;
            cvtRGB2HSVKernel<KernelType>(r_span[remain], g_span[remain], b_span[remain], h_span[remain], s_span[remain],
                                         v_span[remain]);
        }
        for (size_t i = 0; i < remain; ++i)
        {
            cvtRGB2HSVKernel<FallbackKernelType>(r_span[i], g_span[i], b_span[i], h_span[i], s_span[i], v_span[i]);
        }
    }
    else if constexpr (std::is_same_v<BackendType, BackendStdExecution>)
    {
        constexpr size_t data_parallel = KernelType::data_parallel;
        auto data_iter = std::ranges::iota_view(0ul, layer_size) | std::views::stride(data_parallel) |
                         std::views::reverse | std::views::drop(1);
        std::for_each(std::execution::par_unseq, data_iter.begin(), data_iter.end(),
                      [&r_span, &g_span, &b_span, &h_span, &s_span, &v_span](const auto i) noexcept {
                          cvtRGB2HSVKernel<KernelType>(r_span[i], g_span[i], b_span[i], h_span[i], s_span[i],
                                                       v_span[i]);
                      });
        const auto remain = layer_size % data_parallel;
        for (size_t i = layer_size - remain; i < layer_size; ++i)
        {
            cvtRGB2HSVKernel<FallbackKernelType>(r_span[i], g_span[i], b_span[i], h_span[i], s_span[i], v_span[i]);
        }
    }
    else if constexpr (std::is_same_v<BackendType, BackendThread>)
    {
        const size_t concurrency = std::thread::hardware_concurrency();
        if (concurrency == 0) [[unlikely]]
        {
            return false;
        }
        constexpr size_t data_parallel = KernelType::data_parallel;
        const size_t dispatcher_size = concurrency * data_parallel;
        const size_t chunk_size = layer_size / dispatcher_size;
        const size_t remain = layer_size % dispatcher_size;

        std::forward_list<std::thread> threads;
        auto concurrency_iter = std::ranges::iota_view(0ul, std::min(concurrency, chunk_size));
        for (const auto i : concurrency_iter)
        {
            const size_t start = i * chunk_size;
            const size_t end = start + chunk_size;
            threads.emplace_front(
                [start, end, data_parallel, &r_span, &g_span, &b_span, &h_span, &s_span, &v_span] noexcept {
                    for (size_t i = start; i < end; i += data_parallel)
                    {
                        cvtRGB2HSVKernel<KernelType>(r_span[i], g_span[i], b_span[i], h_span[i], s_span[i], v_span[i]);
                    }
                });
        }

        const size_t remain_data_start = layer_size - remain;
        const size_t remain_group_size = remain / data_parallel;
        const size_t remain_remain = remain % data_parallel;

        for (size_t i = 0, start = remain_data_start; i < remain_group_size; ++i, start += data_parallel)
        {
            cvtRGB2HSVKernel<KernelType>(r_span[start], g_span[start], b_span[start], h_span[start], s_span[start],
                                         v_span[start]);
        }

        for (size_t i = 0, start = layer_size - remain_remain; i < remain_remain; ++i, ++start)
        {
            cvtRGB2HSVKernel<FallbackKernelType>(r_span[start], g_span[start], b_span[start], h_span[start],
                                                 s_span[start], v_span[start]);
        }

        for (auto &thread : threads)
        {
            thread.join();
        }
    }
    else
    {
        return false;
    }

    return true;
}

} // namespace cv
