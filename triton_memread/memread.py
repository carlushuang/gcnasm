import torch
import torch.profiler as tpf

import triton
import triton.language as tl
import copy
import os
import pandas as pd

DEVICE = torch.device('cuda:0') # triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

def get_trace_perf(prof, num_iters):
    assert num_iters > 1
    num_iters -= 1
    df = []
    cols = [
        "name",
        "self_cpu_time_total",
        "self_device_time_total",
        "device_type",
        "device_index",
    ]
    for el in prof.events():
        df.append([getattr(el, x, None) for x in cols])
    df = pd.DataFrame(df, columns=cols)
    df["cnt"] = 1
    rets = []
    for name, d in df.groupby("name", sort=False):
        r = d.iloc[1:][["cnt", "self_cpu_time_total", "self_device_time_total"]].sum()
        if not r.empty:
            device_type = str(d["device_type"].iat[0]).split(".")[-1]
            r["name"] = name
            r["device_type"] = device_type
            r["device_index"] = str(d["device_index"].iat[0])
            if device_type == "CUDA":
                r["device_time_sum"] = r["self_device_time_total"]
                r["host_time_sum"] = 0
            else:
                r["host_time_sum"] = r["self_device_time_total"]
                r["device_time_sum"] = 0

        rets.append(r)
    df = pd.DataFrame(rets)

    cols = [
        "name",
        "cnt",
        "host_time_sum",
        "device_time_sum",
        "device_type",
        "device_index",
    ]
    cols = [el for el in cols if el in df.columns]
    df = df[(df.host_time_sum > 0) | (df.device_time_sum > 0)]

    timerList = [
        "host_time_sum",
        "device_time_sum",
    ]
    df = df[cols].sort_values(timerList, ignore_index=True)
    avg_name = "[avg us/iter]"
    for el in timerList:
        df.at[avg_name, el] = df[el].sum() / num_iters
    if int(os.environ.get("AITER_LOG_MORE", 0)):
        pd.set_option("display.expand_frame_repr", False)
        pd.set_option("display.max_colwidth", 90)
        pd.set_option("display.float_format", "{:,.1f}".format)
        #logger.info(f"{df}")
    return df.at[avg_name, "device_time_sum"]

def device_memory_profiling(func, *args, **kwargs):
    gpu_id = torch.cuda.current_device()
    inputSize = (
        sum(
            [
                el.nbytes
                for el in args
                if isinstance(el, torch.Tensor) and el.device.index == gpu_id
            ]
        )
        + 1
    )
    torch.cuda.reset_peak_memory_stats(gpu_id)
    cuda_memory_before = (
        torch.cuda.mem_get_info(gpu_id)[1] - torch.cuda.mem_get_info(gpu_id)[0]
    )
    torch_memory_before = torch.cuda.memory_reserved(gpu_id)
    torch_peak_before = torch.cuda.memory_stats(gpu_id).get(
        "allocated_bytes.all.peak", 0
    )
    non_torch_memory_before = cuda_memory_before - torch_memory_before

    data = func(*args, **kwargs)

    torch.cuda.reset_peak_memory_stats(gpu_id)
    cuda_memory_after = (
        torch.cuda.mem_get_info(gpu_id)[1] - torch.cuda.mem_get_info(gpu_id)[0]
    )
    torch_memory_after = torch.cuda.memory_reserved(gpu_id)
    torch_peak_after = torch.cuda.memory_stats(gpu_id).get(
        "allocated_bytes.all.peak", 0
    )
    non_torch_memory_after = cuda_memory_after - torch_memory_after

    torch_peak_increase = torch_peak_after - torch_peak_before
    non_torch_increase = non_torch_memory_after - non_torch_memory_before
    iter_used_memory = torch_peak_increase + non_torch_increase + inputSize

    return iter_used_memory, inputSize, torch_peak_increase, non_torch_increase


def run_iters(num_iters, func, *args, **kwargs):
    data = None
    for _ in range(num_iters):
        data = func(*args, **kwargs)
    return data


def run_iters_rotate(num_iters, func, rotate_args):
    data = None
    num_rotate_args = len(rotate_args)
    for _ in range(num_iters):
        args, kwargs = rotate_args[_ % num_rotate_args]
        data = func(*args, **kwargs)
    return data

def perftest(
    num_iters=101, num_warmup=2, testGraph=False, num_rotate_args=0, needTrace=False
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            num = num_rotate_args
            if num < 1:
                gpu_id = torch.cuda.current_device()

                iter_used_memory, inputSize, _, _ = device_memory_profiling(
                    func, *args, **kwargs
                )

                properties = torch.cuda.get_device_properties(gpu_id)
                free_memory = torch.cuda.mem_get_info(gpu_id)[0]
                cache_size = min(
                    getattr(properties, "L2_cache_size", 4096 * 1024) * 64 * 128,
                    (free_memory - iter_used_memory + inputSize) * 0.9,
                )
                cache_size = max(cache_size, 0)
                num = int((cache_size + inputSize - 1) // inputSize)
                # print(f"{iter_used_memory=}, {inputSize=}, {cache_size=}, {free_memory=}, {num=}")
            num = min(num, num_iters)

            rotate_args = [
                (copy.deepcopy(args), copy.deepcopy(kwargs)) for _ in range(num - 1)
            ] + [(args, kwargs)]

            run_iters(num_warmup, func, *args, **kwargs)
            if int(os.environ.get("AITER_LOG_MORE", 0)):
                latencies = []
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                for _ in range(num_iters):
                    start_event.record()
                    data = func(*args, **kwargs)
                    end_event.record()
                    end_event.synchronize()
                    latencies.append(start_event.elapsed_time(end_event))
                avg = np.mean(latencies) * 1000
                #logger.info(f"avg: {avg} us/iter from cuda.Event")

            if testGraph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    data = run_iters_rotate(num_iters, func, rotate_args)
                with tpf.profile(
                    activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                    profile_memory=True,
                    with_stack=True,
                    with_modules=True,
                    on_trace_ready=(
                        tpf.tensorboard_trace_handler("./aiter_logs/")
                        if needTrace
                        else None
                    ),
                ) as prof:
                    run_iters(1, graph.replay)
                avg = get_trace_perf(prof, num_iters)
                print(f"avg: {avg} us/iter with hipgraph")
            with tpf.profile(
                activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                profile_memory=True,
                with_stack=True,
                with_modules=True,
                #  record_shapes=True,
                on_trace_ready=(
                    tpf.tensorboard_trace_handler("./aiter_logs/")
                    if needTrace
                    else None
                ),
            ) as prof:
                data = run_iters_rotate(num_iters, func, rotate_args)
            avg = get_trace_perf(prof, num_iters)
            return data, avg
        return wrapper
    return decorator

def run_perftest(
    func,
    *args,
    num_iters=101,
    num_warmup=2,
    testGraph=False,
    num_rotate_args=0,
    needTrace=False,
    **kwargs,
):
    @perftest(
        num_iters=num_iters,
        num_warmup=num_warmup,
        testGraph=testGraph,
        num_rotate_args=num_rotate_args,
        needTrace=needTrace,
    )
    def worker(*args, **kwargs):
        return func(*args, **kwargs)

    return worker(*args, **kwargs)


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
# output_triton = add(x, y)
output_triton, us = run_perftest(add, x, y,  testGraph=True, needTrace=True)
print(output_torch)
print(output_triton)
print(f'time:{us}us')
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
