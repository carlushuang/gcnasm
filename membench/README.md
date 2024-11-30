# Prqrequests
* pip install pandas
* pip install tqdm
* sudo apt-get install jq


# Test Type
* memcpy_persistent
  * global memory -> register buffer -> global memory
* memread_stream
  * global memory read
* memwrite_stream
  * global memory write
* memcpy_stream
  * global memory -> global memory
* memcpy_stream_async
  * global memory A -> shared memory (LDS) -> global memory B
* memcpy_stream_async_inplace
  * global memory A -> shared memory (LDS) -> global memory A
* memcpy_stream_swizzled
  * global memory -> global memory (swizzled like the mfma 32x32x8(16) case)

# Compile
<pre>
Usage: ./build.sh [options]
   -b Block_Size (defualt is 1024)
   -g Grid_Size (default is same as the CU number)
   -u UNROLL (default is 4)
   -i INNER (default is 1)
   -n Nontemporal load/store (Enable by 1, default is 0)
   -k Memory test type: (By default, all will be executed.)
                 memcpy_persistent
                 memread_stream
                 memwrite_stream
                 memcpy_stream
                 memcpy_stream_async
                 memcpy_stream_async_inplace
                 memcpy_stream_swizzled
   -a Offload architecture (default is gfx942)
   -v Verification (Enable by 1, default is 0.)
   -s Array Size MB (default is 1750)
   -r Row per thread (default is 1)
   -p Padding (default is 0)
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

Example:
If you want to test the copy between global memory and shared memory and need to set the following parameters on gfx942:
Block Size = 768, Grid Size = 1440, UNROLL = 32, Array Size = 1890 MB, Row per thtead = 2, Padding byte = 4.
<pre>
<code>
./build.sh -b 768 -g 1440 -u 32 -s 1890 -k memcpy_stream_async -r 2 -p 4 -a gfx942
</code>
</pre>

# Run
<pre>
<code>
./membench
</code>
</pre>

# Result
&lt;access array size&gt; &lt;test type&gt; -->> &lt;bandwidth&gt;
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
        "ENABLE_MEMCPY_PERSISTENT": "0",
        "MEMCPY_PERSISTENT_BS_VAL": "512",
        "MEMCPY_PERSISTENT_GS_VAL": "1600",
        "MEMCPY_PERSISTENT_UNROLL_VAL": "8",
        "MEMCPY_PERSISTENT_PIXELS_MB_VAL": "450.0",
        "MEMCPY_PERSISTENT_VERIFY": "0"
    },
    "memread_stream": {
        "ENABLE_MEMREAD_STREAM": "0",
        "READ_BS_VAL": "1024",
        "READ_GS_VAL": "80",
        "READ_UNROLL_VAL": "4",
        "READ_PIXELS_MB_VAL": "437.5",
        "READ_STREAM_NONTEMP": "1",
        "READ_VERIFY": "1"
    },
    "memwrite_stream": {
        "ENABLE_MEMWRITE_STREAM": "0",
        "WRITE_BS_VAL": "1024",
        "WRITE_GS_VAL": "80",
        "WRITE_UNROLL_VAL": "4",
        "WRITE_PIXELS_MB_VAL": "437.5",
        "WRITE_STREAM_NONTEMP": "1",
        "WRITE_VERIFY": "0"
    },
    "memcpy_stream": {
        "ENABLE_MEMCPY_STREAM": "1",
        "MEMCPY_BS_VAL": "256",
        "MEMCPY_GS_VAL": "80",
        "MEMCPY_UNROLL_VAL": "1",
        "MEMCPY_ROW_PER_THREAD_VAL": "1",
        "MEMCPY_PADDING_VAL": "28",
        "MEMCPY_PIXELS_MB_VAL": "437.5",
        "MEMCPY_STREAM_NONTEMP": "1",
        "MEMCPY_STREAM_VERIFY": "0"
    },
    "memcpy_stream_async": {
        "ENABLE_MEMCPY_STREAM_ASYNC": "0",
        "MEMCPY_ASYNC_BS_VAL": "256",
        "MEMCPY_ASYNC_GS_VAL": "1200",
        "MEMCPY_ASYNC_UNROLL_VAL": "22",
        "MEMCPY_ASYNC_ROW_PER_THREAD_VAL": "1",
        "MEMCPY_ASYNC_PADDING_VAL": "0",
        "MEMCPY_ASYNC_PIXELS_MB_VAL": "438.28125",
        "MEMCPY_STREAM_ASYNC_VERIFY": "1"
    },
    "memcpy_stream_async_inplace": {
        "ENABLE_MEMCPY_STREAM_ASYNC_INPLACE": "0",
        "MEMCPY_ASYNC_INPLACE_BS_VAL": "256",
        "MEMCPY_ASYNC_INPLACE_GS_VAL": "1200",
        "MEMCPY_ASYNC_INPLACE_UNROLL_VAL": "22",
        "MEMCPY_ASYNC_INPLACE_ROW_PER_THREAD_VAL": "1",
        "MEMCPY_ASYNC_INPLACE_PADDING_VAL": "0",
        "MEMCPY_ASYNC_INPLACE_PIXELS_MB_VAL": "438.28125",
        "MEMCPY_STREAM_ASYNC_INPLACE_VERIFY": "1"
    },
    "memcpy_stream_swizzled": {
        "ENABLE_MEMCPY_STREAM_SWIZZLED": "0",
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
The following compile step will configure settings based on the parameters in the config.json file
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
Please provide a section name as the first argument.
Available section names are:
memcpy_persistent
memread_stream
memwrite_stream
memcpy_stream
memcpy_stream_async
memcpy_stream_async_inplace
memcpy_stream_swizzled
</pre>
Select a section from the available options as the test case you want to optimize, and use it as the section_name argument. Additionally, provide the array_size_in_MB and ARCH_value options for further configuration.
<pre>
bash find_config_bw.sh &lt;section_name&gt; &lt;array_size_in_MB&gt; &lt;ARCH_value&gt; &lt;GPU_tuning_num&gt;
</pre>

Usage example:
<pre>
<code>
bash find_config_bw.sh memcpy_persistent 1760 gfx942 8
</code>
</pre>

After the tuning is completed, the best parameter results will be listed as follows:
<pre>
<code>
       BS    GS  UNROLL  PIXELS_MB  INNER  ROW_PER_THREAD  PADDING      GB/s
973   768  1440      32  540.00000      1               1        0  4160.281
977   768  1600      30  562.50000      1               1        0  4118.950
969   768  1360      26  517.96875      1               1        0  4118.415
707   768  1120      32  525.00000      1               1        0  4038.225
742  1024   880      30  515.62500      1               1        0  4022.319
768  1024  1600      30  562.50000      1               1        0  4014.824
464   768  1600      22  515.62500      1               1        0  4011.738
507  1024  1600      32  600.00000      1               1        0  4008.631
500  1024  1360      20  531.25000      1               1        0  4006.123
123  1024  1520      22  522.50000      1               1        0  4000.289
</code>
</pre>

To further refine the search space for tuning, adjust the parameters for BS (block size), GS (grid size), UNROLL (unroll number), and INNER (inner number)  within the **gen_all.py** script.


