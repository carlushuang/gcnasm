#include "warp_histogram.hpp"
#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// keep sync the name with the extension installed
// otherwise can't find this module
#ifndef MODULE_NAME
#define MODULE_NAME warp_histogram_cpp
#endif

// an adaptor to dispatch to host API
// write less sw logic here, only torch->C++ dispatch
at::Tensor warp_histogram_torch(at::Tensor x, int buckets)
{
    int num_element = x.size(0);
    auto options = x.options();
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};

    auto o = torch::empty({buckets}, options.dtype(torch::kInt32));

    warp_histogram(x.data_ptr(), o.data_ptr(), buckets, num_element);

    return o;
}

PYBIND11_MODULE(MODULE_NAME, m) {
    m.doc() = "warp_histogram python module";
    m.def("warp_histogram", &warp_histogram_torch, "warp_histogram function");
}
