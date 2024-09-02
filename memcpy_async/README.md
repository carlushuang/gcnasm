# How to build a test case
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
bash build.sh
./memcpy_async
</pre>
The default above is set to compile under the gfx942 architecture. You can change the **--offload-arch** option in the **HIP.make** file to switch to other architectures.

## How to set a *_PIXELS_MB_VAL value
In the **config.json** file, the *_PIXELS_MB_VAL field represents the PIXELS MB size required for the src/des buffer. The corresponding total size is *_PIXELS_MB_VAL * 4 * 1024 * 1024 bytes. 
Since we allow users to specify the grid size and block size, be aware of whether the calculated value of *_PxIXELS_MB_VAL can be allocated to the processing range of all GPU kernel threads.
You can also try modifying the BS (block size), GS (grid size), and UNROLL parameter combinations in **gen_all.py** and execute: 
<pre>
python gen_all.py
</pre>
This will generate the **all_input.csv** file. Inside it, each combination has a corresponding PIXELS_MB value, representing that a size greater than 1GB is suitable for the specific block size, grid size, and unroll settings.

# How to find the best bandwidth settings
Run find_config_bw.sh
<pre>
bash find_config_bw.sh
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
Choose one of the sections as the test case you want to tune, and use it as the execution parameter, like this:
<pre>
bash find_config_bw.sh memcpy_stream 
</pre>
To adjust the tuning search space, you can modify the BS (block size), GS (grid size), and UNROLL (unroll number) parameters in **gen_all.py**. 
After **find_config_bw.sh** finishes running, it will generate an **all_output.csv** file containing the results of all combinations, and it will print the top 10 best combinations for the user's reference.




