"""
JIT compilation module for warp_bitonic_sort (pure pybind11).

Generates Ninja build files, compiles on first use, caches the .so,
and loads the extension via importlib.
"""

import functools
import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────────
# env: paths
# ──────────────────────────────────────────────────────────────────────

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # pybind/
CSRC_DIR = _PACKAGE_ROOT / "csrc"

CACHE_DIR = Path(
    os.getenv(
        "WARP_SORT_CACHE_DIR",
        Path.home() / ".cache" / "warp_bitonic_sort_pybind",
    )
)
JIT_DIR = CACHE_DIR / "jit"

ROCM_PATH = Path(os.getenv("ROCM_PATH", "/opt/rocm"))
OPUS_DIR = Path(os.getenv("OPUS_DIR", "/raid0/carhuang/repo/aiter/csrc/include"))
GPU_ARCH = os.getenv("GPU_ARCH", "native")

# pybind11 / Python include paths
_PYBIND11_INC = None
_PYTHON_INC = sysconfig.get_path("include")
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _get_pybind11_include() -> str:
    global _PYBIND11_INC
    if _PYBIND11_INC is None:
        import pybind11
        _PYBIND11_INC = pybind11.get_include()
    return _PYBIND11_INC


# ──────────────────────────────────────────────────────────────────────
# ninja build generation
# ──────────────────────────────────────────────────────────────────────

def _join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def _generate_ninja_build(
    name: str,
    sources: List[Path],
    extra_hipcc_flags: Optional[List[str]] = None,
    extra_cxx_flags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
) -> str:
    """Generate a build.ninja for compiling HIP + C++ sources into a pybind11 module."""

    hipcc = str(ROCM_PATH / "bin" / "hipcc")
    cxx = os.environ.get("CXX", "g++")

    pb_inc = _get_pybind11_include()

    common_includes = [
        f"-I{CSRC_DIR.resolve()}",
        f"-isystem {pb_inc}",
        f"-isystem {_PYTHON_INC}",
    ]

    hip_flags = [
        f"--offload-arch={GPU_ARCH}",
        "-O3",
        "-fPIC",
        f"-I{ROCM_PATH / 'include'}",
        f"-I{OPUS_DIR}",
    ] + common_includes
    if extra_hipcc_flags:
        hip_flags += extra_hipcc_flags

    cxx_flags = [
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-fvisibility=hidden",
    ] + common_includes
    if extra_cxx_flags:
        cxx_flags += extra_cxx_flags

    ldflags = [
        "-shared",
        f"-L{ROCM_PATH / 'lib'}",
        "-lamdhip64",
    ]
    if extra_ldflags:
        ldflags += extra_ldflags

    output_dir = JIT_DIR / name

    lines = [
        "ninja_required_version = 1.3",
        f"hipcc = {hipcc}",
        f"cxx = {cxx}",
        "",
        "hip_flags = " + _join_multiline(hip_flags),
        "cxx_flags = " + _join_multiline(cxx_flags),
        "ldflags = " + _join_multiline(ldflags),
        "",
        "rule hip_compile",
        "  command = $hipcc $hip_flags -c $in -o $out",
        "",
        "rule cxx_compile",
        "  command = $cxx $cxx_flags -c $in -o $out",
        "",
        "rule link",
        "  command = $cxx $in $ldflags -o $out",
        "",
    ]

    objects = []
    for source in sources:
        is_hip = source.suffix == ".hip"
        rule = "hip_compile" if is_hip else "cxx_compile"
        obj_suffix = ".hip.o" if is_hip else ".o"
        obj_name = source.with_suffix(obj_suffix).name
        obj = str((output_dir / obj_name).resolve())
        objects.append(obj)
        lines.append(f"build {obj}: {rule} {source.resolve()}")

    lines.append("")
    # Output must use Python extension suffix so importlib can load it
    so_name = f"_C{_EXT_SUFFIX}"
    output_so = str((output_dir / so_name).resolve())
    lines.append(f"build {output_so}: link " + " ".join(objects))
    lines.append(f"default {output_so}")
    lines.append("")

    return "\n".join(lines)


def _run_ninja(workdir: Path, ninja_file: Path, verbose: bool = False) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja", "-v",
        "-C", str(workdir.resolve()),
        "-f", str(ninja_file.resolve()),
    ]

    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs and max_jobs.isdigit():
        command += ["-j", max_jobs]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e


# ──────────────────────────────────────────────────────────────────────
# JitSpec: build and load
# ──────────────────────────────────────────────────────────────────────

class JitSpec:
    def __init__(self, name: str, sources: List[Path]):
        self.name = name
        self.sources = sources

    @property
    def build_dir(self) -> Path:
        return JIT_DIR / self.name

    @property
    def ninja_path(self) -> Path:
        return self.build_dir / "build.ninja"

    @property
    def library_path(self) -> Path:
        return self.build_dir / f"_C{_EXT_SUFFIX}"

    @property
    def is_compiled(self) -> bool:
        return self.library_path.exists()

    def _write_ninja(self) -> None:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        content = _generate_ninja_build(self.name, self.sources)
        if self.ninja_path.exists() and self.ninja_path.read_text() == content:
            return
        self.ninja_path.write_text(content)

    def build(self, verbose: bool = False) -> None:
        self._write_ninja()
        _run_ninja(self.build_dir, self.ninja_path, verbose)

    def load(self):
        """Load the compiled .so as a Python module via importlib."""
        spec = importlib.util.spec_from_file_location("_C", str(self.library_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def build_and_load(self):
        verbose = os.environ.get("WARP_SORT_JIT_VERBOSE", "0") == "1"
        if not self.is_compiled:
            self.build(verbose)
        return self.load()


# ──────────────────────────────────────────────────────────────────────
# Module specification
# ──────────────────────────────────────────────────────────────────────

@functools.cache
def gen_sort_module() -> JitSpec:
    return JitSpec(
        name="warp_bitonic_sort",
        sources=[
            CSRC_DIR / "warp_bitonic_sort.hip",
            CSRC_DIR / "pybind_api.cpp",
        ],
    )
