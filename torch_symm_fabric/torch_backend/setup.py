"""Build the FABRIC SymmetricMemory backend as a torch extension.

    python3 setup.py build_ext --inplace

Built against torch's internal c10d headers, which carry no stability guarantee -- expect
to revisit this on a torch upgrade. That is the main cost of the register_allocator route
compared with the ctypes module one directory up.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))

setup(
    name="fabric_backend",
    ext_modules=[
        CUDAExtension(
            name="fabric_backend",
            sources=[os.path.join(HERE, "fabric_backend.hip")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
