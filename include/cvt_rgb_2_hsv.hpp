#pragma once

#include <algorithm>
#include <atomic>
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

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace cv
{

namespace traits
{

template <typename T, typename P> static constexpr bool compare_kernel()
{
    return T::data_parallel >= P::data_parallel;
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

        s = diff * div_table[v] >> hsv_shift;
        h = (vr & (g - b)) + (~vr & ((vg & (b - r + (2 * diff))) + ((~vg) & (r - g + (4 * diff)))));
        h = (h * div_table[diff] * h_scale + (1 << (hsv_shift + 6))) >> (7 + hsv_shift);
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

    static constexpr auto div_table{[]() noexcept {
        std::array<int, 256> table;
        table[0] = 0;
        for (size_t i = 1; i < table.size(); ++i)
        {
            table[i] = (1 << 14) / i;
        }
        return table;
    }()};
};

#ifdef __ARM_NEON

// class RGB2HSVNeon final : public Kernel<RGB2HSVNeon>
// {
//   public:
//     static constexpr std::string_view name = "RGB2HSVNeon";
//     static constexpr size_t data_parallel = 16;

//     static inline constexpr void operator()(size_t index, const std::span<const uint8_t> &in_r,
//                                             const std::span<const uint8_t> &in_g, const std::span<const uint8_t>
//                                             &in_b, const std::span<uint8_t> &out_h, const std::span<uint8_t> &out_s,
//                                             const std::span<uint8_t> &out_v) noexcept
//     {
//         uint8x16_t r_vec = vld1q_u8(&in_r[index]);
//         uint8x16_t g_vec = vld1q_u8(&in_g[index]);
//         uint8x16_t b_vec = vld1q_u8(&in_b[index]);

//         uint8x16_t maxc = vmaxq_u8(r_vec, vmaxq_u8(g_vec, b_vec));
//         uint8x16_t minc = vminq_u8(r_vec, vminq_u8(g_vec, b_vec));

//         uint8x16_t eq = vceqq_u8(minc, maxc);
//         uint8x16_t uh = vdupq_n_u8(0);
//         uint8x16_t us = vdupq_n_u8(0);

//         // TODO: Implement the rest of the kernel
//     }
// };

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
        while (start < end)
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
    static inline constexpr size_t operator()(const size_t start, const size_t size, Args &&...args) noexcept
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
    static inline constexpr size_t operator()(const size_t start, const size_t size, Args &&...args) noexcept
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
        const size_t dispatcher_size = concurrency * seq_data_parallel;
        std::atomic<size_t> unprocessed = 0;
        std::forward_list<std::thread> threads;
        if (const size_t chunk_size = size / dispatcher_size; chunk_size > 0)
        {
            for (size_t i = 0; i < concurrency; ++i)
            {
                const size_t chunk_start = i * chunk_size;
                threads.emplace_front([start, chunk_start, chunk_size, &unprocessed, &args...]() noexcept {
                    unprocessed.fetch_add(Sequential::template operator()<KernelTypes...>(
                        start + chunk_start, chunk_size, std::forward<Args>(args)...));
                });
            }
        }
        if (const size_t remain = size % dispatcher_size; remain > 0)
        {
            const size_t remain_data_start = size - remain;
            unprocessed.fetch_add(Sequential::template operator()<KernelTypes...>(start + remain_data_start, remain,
                                                                                  std::forward<Args>(args)...));
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

    return BackendType::template operator()<KernelTypes...>(0, layer_size,
                                                            in.subspan(Indices * layer_size, layer_size)...,
                                                            out.subspan(Indices * layer_size, layer_size)...) == 0;
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

} // namespace cv
