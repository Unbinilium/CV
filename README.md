### A simple C++ header-only library for Computer Vision

This library is designed for high-performance parallel computing, including thread based parallelism and instruction-level parallelism, and is abstracted by the concept of homogeneous computing. Currently, it is just a personal interest project with very limited supported features.

```cpp
#include <vector>

#include "utils.hpp"

using namespace cv;

int main()
{
    const size_t N = 1024, size = 3 * N * N;
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

    benchmarkCvtColor<backends::Sequential, kernels::RGB2HSVCVNeonAAarch64<>, kernels::RGB2HSVCV<>>(image, hsv_1);
    compareData(hsv_1, hsv_2);
    benchmarkCvtColor<backends::STLThreads, kernels::RGB2HSVCVNeonAAarch64<>, kernels::RGB2HSVCV<>>(image, hsv_2);
    compareData(hsv_1, hsv_2);
}
```

Todo:

- Implement SSE4.x kernels
- Implement ThreadPool backends
- Support more image processing tasks
