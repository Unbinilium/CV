### Simple header-only C++23 library for Computer Vision

```cpp

#include <chrono>
#include <iostream>
#include <span>
#include <vector>

#include "cvt_rgb_2_hsv.hpp"

using namespace cv;

template <typename Backend, typename KernelType>
constexpr void benchmarkRGB2HSV(std::span<const uint8_t> in, std::span<uint8_t> out)
{
    auto start = std::chrono::high_resolution_clock::now();
    bool success = cvtRGB2HSV<Backend, KernelType>(in, out);
    auto end = std::chrono::high_resolution_clock::now();
    if (!success)
    {
        throw std::runtime_error("Benchmark failed");
    }
    std::printf("Benchmark - Backend[%s] - Kernel[%s]: %fms\n", Backend::name.data(), KernelType::name.data(),
                std::chrono::duration<double, std::milli>(end - start).count());
}

void compareData(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error("Size mismatch");
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i] != b[i])
        {
            auto diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
            std::cout << "Mismatch at index: " << i << " - A: " << static_cast<int>(a[i])
                      << " - B: " << static_cast<int>(b[i]) << " - Diff: " << diff << std::endl;
        }
    }
}

int main()
{
    // generate some an 3 * N * N image
    constexpr size_t N = 2048;
    constexpr size_t size = 3 * N * N;
    std::vector<uint8_t> image(size);
    std::ranges::generate(image, [n = 0]() mutable { return n++ % 256; });

    std::vector<uint8_t> hsv_1(size);
    std::vector<uint8_t> hsv_2(size);

    benchmarkRGB2HSV<BackendSequential, KernelPIL<1, float>>(image, hsv_1);
    benchmarkRGB2HSV<BackendSequential, KernelCV<1, 180>>(image, hsv_2);
    compareData(hsv_1, hsv_2);

    benchmarkRGB2HSV<BackendStdExecution, KernelPIL<1, float>>(image, hsv_1);
    compareData(hsv_1, hsv_2);
    benchmarkRGB2HSV<BackendStdExecution, KernelCV<1, 180>>(image, hsv_2);
    compareData(hsv_1, hsv_2);

    benchmarkRGB2HSV<BackendThread, KernelPIL<1, float>>(image, hsv_1);
    compareData(hsv_1, hsv_2);
    benchmarkRGB2HSV<BackendThread, KernelCV<1, 180>>(image, hsv_2);
    compareData(hsv_1, hsv_2);

    // TODO:
    // benchmarkRGB2HSV<BackendThread, KernelNeon<16>>(image, hsv_2);
    // benchmarkRGB2HSV<BackendThread, KernelAVX512<>>(image, hsv_2);
    // benchmarkRGB2HSV<BackendThread, KernelSSE4<>>(image, hsv_2);

    return 0;
}

```
