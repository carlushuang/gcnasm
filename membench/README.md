# Test Type
* memcpy_persistent
  * global memory -> register buffer -> global memory
* memread_stream
  * global memory read
* memcpy_stream
  * global memory -> global memory
* memcpy_stream_async
  * global memory -> shared memory (LDS) -> global memory
* memcpy_stream_swizzled
  * global memory -> global memory (swizzled like the mfma 32x32x8(16) case)


# Compile
<pre>
Usage: ./build.sh [options]
   -b Block_Size (defualt is 1024)
   -g Grid_Size (default is same as the CU number)
   -u UNROLL (default is 4)
   -i INNER (default is 1)
   -n nontemporal load/store (Enable by 1, default is 0)
   -k memory test type: (By default, all will be executed.)
                 memcpy_persistent
                 memread_stream
                 memcpy_stream
                 memcpy_stream_async
                 memcpy_stream_swizzled
   -a offload architecture (default is gfx942)
   -v Verification (Enable by 1, default is 0.)
   -s Array Size MB (default is 1750)
   -h Show this help message
</pre>


Example:
If you want to test memory read and need to set the following parameters on gfx942:
Block Size = 1024, Grid Size = 80, UNROLL = 4, Array Size = 200 MB, set nontemporal load.
<pre>
<code>
./build.sh -b 1024 -g 80 -u 4 -s 1750 -n 1 -k memread_stream -a gfx942
</code>
</pre>

Example:
If you want to test global memory copy and need to set the following parameters on gfx942:
Block Size = 768, Grid Size = 2880, UNROLL = 28, Array Size = 1890 MB, set nontemporal load & store.
<pre>
<code>
./build.sh -b 768 -g 2880 -u 4 -s 1890 -n 1 -k memcpy_stream -a gfx942
</code>
</pre>

# Run
<pre>
<code>
./membench
</code>
</pre>

# Result
<pre>
1.72GB memcpy_persistent -->> 19.343(GB/s)
1.71GB memread_stream -->> 4172.590(GB/s)
1.85GB memcpy_stream -->> 3323.929(GB/s)
1.71GB memcpy_stream_async -->> 2670.484(GB/s)
1.71GB memcpy_stream_swizzled -->> 1595.172(GB/s)
</pre>

# Config different test cases
## Modify config.json
<pre>
<code>
{
    "memcpy_persistent": {
        "ENABLE_MEMCPY_PERSISTENT": "1",
        "MEMCPY_PERSISTENT_BS_VAL": "1024",
        "MEMCPY_PERSISTENT_GS_VAL": "160",
        "MEMCPY_PERSISTENT_UNROLL_VAL": "16",
        "MEMCPY_PERSISTENT_PIXELS_MB_VAL": "440",
        "MEMCPY_PERSISTENT_VERIFY": "1"
    },
    "memread_stream": {
        "ENABLE_MEMREAD_STREAM": "1",
        "READ_BS_VAL": "1024",
        "READ_GS_VAL": "80",
        "READ_UNROLL_VAL": "4",
        "READ_PIXELS_MB_VAL": "437.5",
        "READ_STREAM_NONTEMP": "1",
        "READ_VERIFY": "0"
    },
    "memcpy_stream": {
        "ENABLE_MEMCPY_STREAM": "1",
        "MEMCPY_BS_VAL": "768",
        "MEMCPY_GS_VAL": "2880",
        "MEMCPY_UNROLL_VAL": "28",
        "MEMCPY_PIXELS_MB_VAL": "472.5",
        "MEMCPY_STREAM_NONTEMP": "1",
        "MEMCPY_STREAM_VERIFY": "1"
    },
    "memcpy_stream_async": {
        "ENABLE_MEMCPY_STREAM_ASYNC": "1",
        "MEMCPY_ASYNC_BS_VAL": "256",
        "MEMCPY_ASYNC_GS_VAL": "1200",
        "MEMCPY_ASYNC_UNROLL_VAL": "22",
        "MEMCPY_ASYNC_PIXELS_MB_VAL": "438.28125",
        "MEMCPY_STREAM_ASYNC_VERIFY": "1"
    },
    "memcpy_stream_swizzled": {
        "ENABLE_MEMCPY_STREAM_SWIZZLED": "1",
        "MEMCPY_SWIZZLED_BS_VAL": "256",
        "MEMCPY_SWIZZLED_GS_VAL": "160",
        "MEMCPY_SWIZZLED_CHUNKS_VAL": "4",
        "MEMCPY_SWIZZLED_INNER_VAL": "1",
        "MEMCPY_SWIZZLED_PIXELS_MB_VAL": "437.5",
        "MEMCPY_STREAM_SWIZZLED_NONTEMP": "0",
        "MEMCPY_STREAM_SWIZZLED_VERIFY": "1"
    }
}
</code>
</pre>

The above file provides fine adjustments for each test item. Its parameters will be expanded in the form of codegen in the test items. For example, if you want to test the memcpy_stream item, you can set: 
<pre>
"ENABLE_MEMCPY_STREAM": "1", 
</pre>
To set the block size used by the kernel function: 
<pre>
"MEMCPY_BS_VAL": "768", 
</pre>  
To set the grid size used by the kernel function: 
<pre>
"MEMCPY_GS_VAL": "2880", 
</pre>
To set the UNROLL value used in the kernel function: 
<pre>
"MEMCPY_UNROLL_VAL": "28", 
</pre>
To set the size value of the src/des buffer used in the kernel function: 
<pre>
"MEMCPY_PIXELS_MB_VAL": "472.5", 
</pre>  
To set whether to enable NONTEMP read/write: 
<pre>
"MEMCPY_STREAM_NONTEMP": "1", 
</pre>
To set whether to verify test results: 
<pre>
"MEMCPY_STREAM_VERIFY": "1"
</pre>
Alternatively, for different test cases, you can add different test fields yourself.

## Compile source & run
<pre>
<code>
ARCH=gfx942 bash build.sh
</code>
</pre>
<pre>
<code>
./membench
</code>
</pre>
The default above is set to compile under the gfx942 architecture. You can change the **ARCH=** to switch to other architectures.


# How to find the best bandwidth settings
Run find_config_bw.sh
<pre>
<code>
bash find_config_bw.sh
</code>
</pre>
The following information will be shown
<pre>
Available section names are:
memcpy_persistent
memread_stream
memcpy_stream
memcpy_stream_async
memcpy_stream_swizzled
</pre>
Select a section from the available options as the test case you want to optimize, and use it as the section_name argument. Additionally, provide the array_size_in_MB and ARCH_value options for further configuration.
<pre>
bash find_config_bw.sh &lt;section_name&gt;  &lt;array_size_in_MB&gt;  &lt;ARCH_value&gt;
</pre>

Usage example:
<pre>
<code>
bash find_config_bw.sh memcpy_persistent 1760 gfx942
</code>
</pre>
After all the combinations have been executed, the top 10 best combination data sets will be printed.
<pre>
     BS   GS  UNROLL   PIXELS_MB  INNER  GFlops
50  128  880       1  440.000000      1  43.200
45  128  800       1  440.234375      1  42.505
40  128  720       1  440.156250      1  42.206
55  128  960       1  440.156250      1  40.963
51  128  880       2  440.000000      1  39.782
28  128  480       6  440.156250      1  39.542
23  128  400       6  440.625000      1  39.132
35  128  640       1  440.000000      1  38.472
56  128  960       2  440.625000      1  37.894
37  128  640       4  440.000000      1  37.879
</pre>

To further refine the search space for tuning, adjust the parameters for BS (block size), GS (grid size), UNROLL (unroll number), and INNER (inner number)  within the **gen_all.py** script.


