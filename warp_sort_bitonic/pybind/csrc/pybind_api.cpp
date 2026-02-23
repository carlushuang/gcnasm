#include <pybind11/pybind11.h>
#include <cstdint>
#include "warp_bitonic_sort.hpp"

namespace py = pybind11;

// Pure pybind11 binding — no torch headers, no ATen, no CUDAExtension.
// Accepts raw device pointers as int64 from Python side.
// The Python wrapper (warp_bitonic_sort/__init__.py) handles tensor
// allocation and data_ptr() extraction.

void warp_bitonic_sort_pybind(std::int64_t input_ptr,
                              std::int64_t output_ptr,
                              int num_element,
                              bool is_descending) {
    warp_bitonic_sort_kernel(
        reinterpret_cast<void*>(input_ptr),
        reinterpret_cast<void*>(output_ptr),
        num_element,
        is_descending ? 1 : 0);
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "warp_bitonic_sort — pure pybind11 binding";
    m.def("warp_bitonic_sort", &warp_bitonic_sort_pybind,
          "Warp-level bitonic merge sort on GPU",
          py::arg("input_ptr"), py::arg("output_ptr"),
          py::arg("num_element"), py::arg("is_descending"));
}
