from glob import glob
import os
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension
import pathlib

import copy
import torch

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.1.0"

__ARCH__="native"

__PWD__ = pathlib.Path(__file__).parent.resolve()

# this example dependes on OPUS headers, hence add it to include dir
__OPUS_INCLUDE__="/raid0/carhuang/repo/aiter/csrc/include"


ext_modules = [
    # the simplest way is to reuse CUDAExtension (inside a rocm torch installed env)
    # it can help add all the include/link path to our module
    CUDAExtension(
        name = "warp_histogram_cpp",     # this is cpp->python module name
        sources = sorted(glob(f"{__PWD__}/csrc/*.cpp") + glob(f"{__PWD__}/csrc/*.hip")),
        include_dirs = [f'{__OPUS_INCLUDE__}'],
        extra_compile_args = {
            # NOTE: offload-arch is important to prevent torch extension
            # to compiler on multi gfx arch
            # 'nvcc': [f'--offload-arch={__ARCH__}', '-O3', '-Wall']
            'nvcc': [f'--offload-arch={__ARCH__}', '-O3', '-Wall', '-v', '--save-temps', '-Wno-gnu-line-marker']
        },
        # is_standalone = True,
    ),
]

setup(
    name="warp_histogram",
    version=__version__,
    packages=find_packages('.'),
    zip_safe=False,
    python_requires=">=3.7",
    cmdclass={"build_ext": BuildExtension},
    ext_modules=ext_modules,
)
