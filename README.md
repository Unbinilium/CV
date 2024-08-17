### Simple header-only C++23 library for Computer Vision

```cpp
#include <chrono>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "cvt_rgb_2_hsv.hpp"

using namespace cv;

template <typename Backend, typename... KernelTypes>
constexpr void benchmarkCvtColor(std::span<const uint8_t> in, std::span<uint8_t> out)
{
    auto start = std::chrono::high_resolution_clock::now();
    bool success = tasks::cvtColor<3, Backend, KernelTypes...>(in, out);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    if (!success)
        throw std::runtime_error("Benchmark failed");
    std::cout << "Backend: " << Backend::name << " - Kernels: " << (std::string{KernelTypes::name} + " " + ...)
              << "- Duration: " << duration << "ms" << std::endl;
}

void compareData(std::span<const uint8_t> a, std::span<const uint8_t> b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Data size mismatch");
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i])
            std::cout << "Mismatch at index: " << i << " - A: " << static_cast<int>(a[i])
                      << " - B: " << static_cast<int>(b[i])
                      << " - Diff: " << std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])) << std::endl;
}

int main()
{
    // generate some an 3 * N * N image
    size_t N = 4096, size = 3 * N * N;
    std::vector<uint8_t> image(size);
    std::ranges::generate(image, [n = 0]() mutable { return ++n % 256; });

    std::vector<uint8_t> hsv_1(size);
    std::vector<uint8_t> hsv_2(size);

    benchmarkCvtColor<backends::Sequential, kernels::RGB2HSVPIL<>>(image, hsv_1);
    benchmarkCvtColor<backends::Sequential, kernels::RGB2HSVCV<>>(image, hsv_2);
    compareData(hsv_1, hsv_2);

    benchmarkCvtColor<backends::STLParallel, kernels::RGB2HSVPIL<>>(image, hsv_1);
    compareData(hsv_1, hsv_2);
    benchmarkCvtColor<backends::STLParallel, kernels::RGB2HSVCV<>>(image, hsv_2);
    compareData(hsv_1, hsv_2);

    benchmarkCvtColor<backends::STLThreads, kernels::RGB2HSVPIL<>>(image, hsv_1);
    compareData(hsv_1, hsv_2);
    benchmarkCvtColor<backends::STLThreads, kernels::RGB2HSVCV<>>(image, hsv_2);
    compareData(hsv_1, hsv_2);

    // TODO:
    // benchmarkCvtColor<..., kernels::RGB2HSVNeon<16>, kernels::RGB2HSVNeon<8>, ...>(image, hsv_1);
    // benchmarkCvtColor<..., kernels::RGB2HSVAVX512<>, kernels::RGB2HSVSSE42<>, ...>(image, hsv_1);
}
```
