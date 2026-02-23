#!/usr/bin/env python3
"""
Compile-time benchmark: torch/ vs pybind/ vs tvmffi/
Measures end-to-end and per-step timings, averaged over multiple runs.
"""

import os
import sys
import time
import shutil
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TORCH_DIR = os.path.join(SCRIPT_DIR, "torch")
PYBIND_DIR = os.path.join(SCRIPT_DIR, "pybind")
TVMFFI_DIR = os.path.join(SCRIPT_DIR, "tvmffi")
CK_DIR = os.environ.get("CK_DIR", "/opt/rocm-7.1.1")
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")

NUM_RUNS = 3


def time_cmd(cmd, cwd=None):
    t0 = time.monotonic()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    t1 = time.monotonic()
    return t1 - t0, r.returncode, r.stderr


# ── torch/ (setup.py build_ext) ──────────────────────────────────────

def clean_torch():
    for d in ["build", "dist"]:
        p = os.path.join(TORCH_DIR, d)
        if os.path.exists(p):
            shutil.rmtree(p)
    for d in os.listdir(TORCH_DIR):
        if d.endswith(".egg-info"):
            shutil.rmtree(os.path.join(TORCH_DIR, d))
    for f in os.listdir(TORCH_DIR):
        if f.endswith(".so"):
            os.remove(os.path.join(TORCH_DIR, f))
    # remove hipified files
    csrc = os.path.join(TORCH_DIR, "csrc")
    for f in os.listdir(csrc):
        if "_hip." in f:
            os.remove(os.path.join(csrc, f))


def bench_torch_e2e():
    clean_torch()
    setup_py = os.path.join(TORCH_DIR, "setup.py")
    with open(setup_py) as f:
        code = f.read()
    code = code.replace("/raid0/carhuang/repo/composable_kernel", CK_DIR)
    patched = os.path.join(TORCH_DIR, "_setup_patched.py")
    with open(patched, "w") as f:
        f.write(code)
    elapsed, rc, stderr = time_cmd(
        f"{sys.executable} {patched} build_ext --inplace", cwd=TORCH_DIR
    )
    os.remove(patched)
    clean_torch()
    if rc != 0:
        print(f"    [WARN] torch build failed: {stderr[-200:]}")
    return elapsed


# ── pybind/ (ninja JIT) ──────────────────────────────────────────────

def clean_pybind():
    cache = os.path.expanduser("~/.cache/warp_bitonic_sort_pybind")
    if os.path.exists(cache):
        shutil.rmtree(cache)


def bench_pybind_e2e():
    clean_pybind()
    os.environ["CK_DIR"] = CK_DIR

    for mod_name in list(sys.modules):
        if "warp_bitonic_sort" in mod_name:
            del sys.modules[mod_name]

    sys.path.insert(0, PYBIND_DIR)
    from warp_bitonic_sort.jit import gen_sort_module
    gen_sort_module.cache_clear()

    t0 = time.monotonic()
    spec = gen_sort_module()
    spec.build(verbose=False)
    t1 = time.monotonic()
    clean_pybind()
    return t1 - t0


# ── tvmffi/ (ninja JIT) ──────────────────────────────────────────────

def clean_tvmffi():
    cache = os.path.expanduser("~/.cache/warp_bitonic_sort")
    if os.path.exists(cache):
        shutil.rmtree(cache)


def bench_tvmffi_e2e():
    clean_tvmffi()
    os.environ["CK_DIR"] = CK_DIR

    for mod_name in list(sys.modules):
        if "warp_bitonic_sort" in mod_name:
            del sys.modules[mod_name]

    sys.path.insert(0, TVMFFI_DIR)
    from warp_bitonic_sort.jit import gen_sort_module
    gen_sort_module.cache_clear()

    t0 = time.monotonic()
    spec = gen_sort_module()
    spec.build(verbose=False)
    t1 = time.monotonic()
    clean_tvmffi()
    return t1 - t0


# ── per-step: isolated compiler commands ──────────────────────────────

def bench_hipcc_kernel():
    """hipcc .hip — shared across all three."""
    import tempfile
    hip_src = os.path.join(TORCH_DIR, "csrc", "warp_bitonic_sort.hip")
    with tempfile.TemporaryDirectory() as td:
        elapsed, _, _ = time_cmd(
            f"{ROCM_PATH}/bin/hipcc --offload-arch=native -O3 -fPIC "
            f"-I{ROCM_PATH}/include -I{CK_DIR}/include "
            f"-I{TORCH_DIR}/csrc "
            f"-c {hip_src} -o {td}/k.o"
        )
    return elapsed


def bench_torch_binding():
    """hipcc torch_api.cpp with torch/pybind11/ATen headers."""
    import torch
    torch_inc = os.path.join(os.path.dirname(torch.__file__), "include")
    python_inc = subprocess.check_output(
        [sys.executable, "-c",
         "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True,
    ).strip()

    import tempfile
    cpp_src = os.path.join(TORCH_DIR, "csrc", "torch_api.cpp")
    with tempfile.TemporaryDirectory() as td:
        elapsed, rc, _ = time_cmd(
            f"{ROCM_PATH}/bin/hipcc --offload-arch=native -O2 -fPIC -std=c++17 "
            f"-I{CK_DIR}/include "
            f"-I{torch_inc} -I{torch_inc}/torch/csrc/api/include "
            f"-I{torch_inc}/THH -I{ROCM_PATH}/include -I{python_inc} "
            "-D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 "
            "-DCUDA_HAS_FP16=1 -D__HIP_NO_HALF_OPERATORS__=1 "
            "-D__HIP_NO_HALF_CONVERSIONS__=1 -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 "
            "-DTORCH_API_INCLUDE_EXTENSION_H "
            "-DTORCH_EXTENSION_NAME=warp_bitonic_sort_cpp "
            f"-c {cpp_src} -o {td}/t.o"
        )
    return elapsed


def bench_pybind_binding():
    """g++ pybind_api.cpp with pybind11 headers."""
    import pybind11
    pb_inc = pybind11.get_include()
    python_inc = subprocess.check_output(
        [sys.executable, "-c",
         "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True,
    ).strip()

    import tempfile
    cpp_src = os.path.join(PYBIND_DIR, "csrc", "pybind_api.cpp")
    with tempfile.TemporaryDirectory() as td:
        elapsed, _, _ = time_cmd(
            f"g++ -std=c++17 -O2 -fPIC -fvisibility=hidden "
            f"-I{PYBIND_DIR}/csrc -isystem {pb_inc} -isystem {python_inc} "
            f"-c {cpp_src} -o {td}/p.o"
        )
    return elapsed


def bench_tvmffi_binding():
    """g++ tvm_api.cc with tvm_ffi headers."""
    import tvm_ffi
    tvm_inc = tvm_ffi.libinfo.find_include_path()
    tvm_dlpack_inc = tvm_ffi.libinfo.find_dlpack_include_path()

    import tempfile
    cc_src = os.path.join(TVMFFI_DIR, "csrc", "tvm_api.cc")
    with tempfile.TemporaryDirectory() as td:
        elapsed, _, _ = time_cmd(
            f"g++ -std=c++17 -O2 -fPIC "
            f"-I{TVMFFI_DIR}/csrc -isystem {tvm_inc} -isystem {tvm_dlpack_inc} "
            f"-c {cc_src} -o {td}/v.o"
        )
    return elapsed


# ── main ──────────────────────────────────────────────────────────────

def avg(lst):
    return sum(lst) / len(lst)


def main():
    print("=" * 70)
    print("COMPILE-TIME BENCHMARK:  torch/  vs  pybind/  vs  tvmffi/")
    print(f"  {NUM_RUNS} runs each, clean build every time")
    print("=" * 70)

    # warm up hipcc
    print("\nWarming up hipcc ...")
    bench_hipcc_kernel()

    results = {k: [] for k in [
        "torch_e2e", "pybind_e2e", "tvmffi_e2e",
        "hipcc", "torch_bind", "pybind_bind", "tvmffi_bind",
    ]}

    for i in range(NUM_RUNS):
        print(f"\n--- Run {i+1}/{NUM_RUNS} ---")

        t = bench_hipcc_kernel()
        results["hipcc"].append(t)
        print(f"  hipcc .hip kernel:         {t:.2f}s")

        t = bench_torch_binding()
        results["torch_bind"].append(t)
        print(f"  hipcc torch_api.cpp:       {t:.2f}s")

        t = bench_pybind_binding()
        results["pybind_bind"].append(t)
        print(f"  g++   pybind_api.cpp:      {t:.2f}s")

        t = bench_tvmffi_binding()
        results["tvmffi_bind"].append(t)
        print(f"  g++   tvm_api.cc:          {t:.2f}s")

        t = bench_torch_e2e()
        results["torch_e2e"].append(t)
        print(f"  torch  end-to-end:         {t:.2f}s")

        t = bench_pybind_e2e()
        results["pybind_e2e"].append(t)
        print(f"  pybind end-to-end:         {t:.2f}s")

        t = bench_tvmffi_e2e()
        results["tvmffi_e2e"].append(t)
        print(f"  tvmffi end-to-end:         {t:.2f}s")

    # averages
    h   = avg(results["hipcc"])
    tb  = avg(results["torch_bind"])
    pb  = avg(results["pybind_bind"])
    vb  = avg(results["tvmffi_bind"])
    te  = avg(results["torch_e2e"])
    pe  = avg(results["pybind_e2e"])
    ve  = avg(results["tvmffi_e2e"])

    print()
    print("=" * 70)
    print(f"RESULTS  (averaged over {NUM_RUNS} runs)")
    print("=" * 70)
    print()
    hdr = f"{'Component':<40} {'torch/':>8} {'pybind/':>8} {'tvmffi/':>8}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'hipcc .hip (GPU kernel)':<40} {h:>7.2f}s {h:>7.2f}s {h:>7.2f}s")
    print(f"{'C++ binding compile':<40} {tb:>7.2f}s {pb:>7.2f}s {vb:>7.2f}s")
    print(f"{'  compiler used':<40} {'hipcc':>8} {'g++':>8} {'g++':>8}")
    print(f"{'  headers':<40} {'torch+pb':>8} {'pb only':>8} {'tvm_ffi':>8}")
    print("-" * len(hdr))
    print(f"{'END-TO-END TOTAL':<40} {te:>7.2f}s {pe:>7.2f}s {ve:>7.2f}s")
    print()

    torch_overhead = te - h - tb
    pybind_overhead = pe - h - pb
    tvmffi_overhead = ve - h - vb

    print(f"{'Python/framework overhead (estimated)':<40} {torch_overhead:>7.1f}s {pybind_overhead:>7.1f}s {tvmffi_overhead:>7.1f}s")
    print()
    print(f"  Speedup vs torch (end-to-end):")
    print(f"    pybind/:  {te/pe:.1f}x faster")
    print(f"    tvmffi/:  {te/ve:.1f}x faster")
    print(f"    pybind/ vs tvmffi/:  {pe/ve:.2f}x")


if __name__ == "__main__":
    main()
