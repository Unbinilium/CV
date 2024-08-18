### A simple C++ header-only library for Computer Vision

This library is designed for high-performance parallel computing, including thread based parallelism and instruction-level parallelism, and is abstracted by the concept of homogeneous computing. Currently, it is just a personal interest project with very limited supported features. 

```cpp
#include <chrono>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "cvtColor.hpp"

using namespace cv;

template <typename Backend, typename... KernelTypes>
constexpr void benchmarkCvtColor(std::span<const uint8_t> in, std::span<uint8_t> out) {
    auto start = std::chrono::high_resolution_clock::now();
    bool success = tasks::cvtColor<3, Backend, KernelTypes...>(in, out);
    auto end = std::chrono::high_resolution_clock::now();
    if (!success) throw std::runtime_error("Benchmark failed");
    std::cout << "Backend: " << Backend::name << " - Kernels: " << (std::string{KernelTypes::name} + " " + ...) << "- Duration: " << std::chrono::duration<double, std::milli>(end - start).count() << "ms" << std::endl;
}

void compareData(std::span<const uint8_t> a, std::span<const uint8_t> b) {
    if (a.size() != b.size()) throw std::runtime_error("Data size mismatch");
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) std::cout << "Mismatch at index: " << i << " - A: " << static_cast<int>(a[i]) << " - B: " << static_cast<int>(b[i]) << " - Diff: " << std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])) << std::endl;
}

int main() {
    const size_t N = 4096, size = 3 * N * N;
    std::vector<uint8_t> image(size); std::ranges::generate(image, [n = 0]() mutable { return ++n % 256; });
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

    benchmarkCvtColor<backends::Sequential, kernels::RGB2HSVCVNeonAAarch64<>, kernels::RGB2HSVCV<>>(image, hsv_1);
    compareData(hsv_1, hsv_2);
    benchmarkCvtColor<backends::STLThreads, kernels::RGB2HSVCVNeonAAarch64<>, kernels::RGB2HSVCV<>> (image, hsv_2);
    compareData(hsv_1, hsv_2);
}
```

Todo:

- Implement AVX512 kernels
- Implement SSE4.x kernels
- Support more image processing tasks
