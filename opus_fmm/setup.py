from glob import glob
import os
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension

import copy
import torch

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.1.0"

__ARCH__="native"

# this example dependes on AITER headers, hence add it to include dir
AITER_INCLUDE_DIR="/raid0/carhuang/repo/aiter/csrc/include"


ext_modules = [
    # the simplest way is to reuse CUDAExtension (inside a rocm torch installed env)
    # it can help add all the include/link path to our module
    CUDAExtension(
        name = "opus_fmm",     # this is cpp->python module name
        sources = sorted(glob("csrc/*.cpp") + glob("csrc/*.hip")),
        include_dirs = [f'{AITER_INCLUDE_DIR}'],
        extra_compile_args = {
            # NOTE: offload-arch is important to prevent torch extension
            # to compiler on multi gfx arch
            'nvcc': [f'--offload-arch={__ARCH__}', '-O3', '-Wall']
        },
    ),
]

setup(
    name="warp_bitonic_sort",
    version=__version__,
    packages=find_packages('.'),
    zip_safe=False,
    python_requires=">=3.7",
    # cmdclass={"build_ext": custom_hip_build_ext},
    cmdclass={"build_ext": BuildExtension},
    ext_modules=ext_modules,
)
