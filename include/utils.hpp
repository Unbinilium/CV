#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>

#include "cvt_color.hpp"

namespace cv
{

template <typename Backend, typename... KernelTypes>
constexpr void benchmarkCvtColor(std::span<const uint8_t> in, std::span<uint8_t> out)
{
    auto start = std::chrono::high_resolution_clock::now();
    bool success = tasks::cvtColor<3, Backend, KernelTypes...>(in, out);
    auto end = std::chrono::high_resolution_clock::now();
    if (!success)
    {
        throw std::runtime_error("Benchmark failed");
    }
    std::cout << "Backend: " << Backend::name << " - Kernels: " << (std::string{KernelTypes::name} + " " + ...)
              << "- Duration: " << std::chrono::duration<double, std::milli>(end - start).count() << "ms" << std::endl;
}

void compareData(std::span<const uint8_t> a, std::span<const uint8_t> b, float tolerance = 5.f)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error("Data size mismatch");
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        const auto diff = std::abs(a[i] - b[i]);
        if (diff > tolerance)
        {
            std::cout << "Mismatch at index: " << i << " - A: " << static_cast<int>(a[i])
                      << " - B: " << static_cast<int>(b[i]) << " - Diff: " << static_cast<int>(diff) << std::endl;
        }
    }
}

} // namespace cv
