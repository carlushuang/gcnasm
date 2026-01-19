#include "warp_bitonic_sort.hpp"
#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// keep sync the name with the extension installed
// otherwise can't find this module
#ifndef MODULE_NAME
#define MODULE_NAME opus_fmm_cpp
#endif

// an adaptor to dispatch to host API
// write less sw logic here, only torch->C++ dispatch
at::Tensor opus_fmm_torch(at::Tensor x, bool is_descending)
{
    int num_element = x.size(0);
    /// auto options = x.options();
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};

    auto y = at::empty_like(x);

    opus_fmm(x.data_ptr(), y.data_ptr(), num_element, is_descending ? 1 : 0);

    return y;
}

PYBIND11_MODULE(MODULE_NAME, m) {
    m.doc() = "opus_fmm python module";
    m.def("opus_fmm", &opus_fmm_torch, "opus flatmm function");
}
