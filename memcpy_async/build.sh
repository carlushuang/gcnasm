#!/bin/bash


make -f ./HIP.make clean

config_file="config.json"

read_val() {
    jq -r ".$1.$2" $config_file
}

while IFS=$'\n' read -r key_value
do
    key=$(echo $key_value | cut -d '=' -f 1)
    value=$(echo $key_value | cut -d '=' -f 2)
    export $key=$value
done < <(jq -r 'to_entries[] | (.key as $sname | .value | to_entries[] | "\(.key)=\(.value)")' $config_file)

make -f ./HIP.make \
MEMCPY_PERSISTENT_BS_VAL=$(read_val "memcpy_persistent" "MEMCPY_PERSISTENT_BS_VAL") \
MEMCPY_PERSISTENT_GS_VAL=$(read_val "memcpy_persistent" "MEMCPY_PERSISTENT_GS_VAL") \
MEMCPY_PERSISTENT_VERIFY=$(read_val "memcpy_persistent" "MEMCPY_PERSISTENT_VERIFY") \
MEMCPY_PERSISTENT_UNROLL_VAL=$(read_val "memcpy_persistent" "MEMCPY_PERSISTENT_UNROLL_VAL") \
MEMCPY_PERSISTENT_PIXELS_MB_VAL=$(read_val "memcpy_persistent" "MEMCPY_PERSISTENT_PIXELS_MB_VAL") \
ENABLE_MEMCPY_PERSISTENT=$(read_val "memcpy_persistent" "ENABLE_MEMCPY_PERSISTENT") \

READ_BS_VAL=$(read_val "memread_stream" "READ_BS_VAL") \
READ_GS_VAL=$(read_val "memread_stream" "READ_GS_VAL") \
READ_VERIFY=$(read_val "memread_stream" "READ_VERIFY") \
READ_UNROLL_VAL=$(read_val "memread_stream" "READ_UNROLL_VAL") \
READ_PIXELS_MB_VAL=$(read_val "memread_stream" "READ_PIXELS_MB_VAL") \
READ_STREAM_NONTEMP=$(read_val "memread_stream" "READ_STREAM_NONTEMP") \
ENABLE_MEMREAD_STREAM=$(read_val "memread_stream" "ENABLE_MEMREAD_STREAM") \

MEMCPY_BS_VAL=$(read_val "memcpy_stream" "MEMCPY_BS_VAL") \
MEMCPY_GS_VAL=$(read_val "memcpy_stream" "MEMCPY_GS_VAL") \
MEMCPY_UNROLL_VAL=$(read_val "memcpy_stream" "MEMCPY_UNROLL_VAL") \
MEMCPY_PIXELS_MB_VAL=$(read_val "memcpy_stream" "MEMCPY_PIXELS_MB_VAL") \
MEMCPY_STREAM_NONTEMP=$(read_val "memcpy_stream" "MEMCPY_STREAM_NONTEMP") \
MEMCPY_STREAM_VERIFY=$(read_val "memcpy_stream" "MEMCPY_STREAM_VERIFY") \
ENABLE_MEMCPY_STREAM=$(read_val "memcpy_stream" "ENABLE_MEMCPY_STREAM") \

MEMCPY_ASYNC_BS_VAL=$(read_val "memcpy_stream_async" "MEMCPY_ASYNC_BS_VAL") \
MEMCPY_ASYNC_GS_VAL=$(read_val "memcpy_stream_async" "MEMCPY_ASYNC_GS_VAL") \
MEMCPY_ASYNC_UNROLL_VAL=$(read_val "memcpy_stream_async" "MEMCPY_ASYNC_UNROLL_VAL") \
MEMCPY_ASYNC_PIXELS_MB_VAL=$(read_val "memcpy_stream_async" "MEMCPY_ASYNC_PIXELS_MB_VAL") \
MEMCPY_STREAM_ASYNC_VERIFY=$(read_val "memcpy_stream_async" "MEMCPY_STREAM_ASYNC_VERIFY") \
ENABLE_MEMCPY_STREAM_ASYNC=$(read_val "memcpy_stream_async" "ENABLE_MEMCPY_STREAM_ASYNC") \

MEMCPY_SWIZZLED_BS_VAL=$(read_val "memcpy_stream_swizzled" "MEMCPY_SWIZZLED_BS_VAL") \
MEMCPY_SWIZZLED_GS_VAL=$(read_val "memcpy_stream_swizzled" "MEMCPY_SWIZZLED_GS_VAL") \
MEMCPY_SWIZZLED_CHUNKS_VAL=$(read_val "memcpy_stream_swizzled" "MEMCPY_SWIZZLED_CHUNKS_VAL") \
MEMCPY_SWIZZLED_INNER_VAL=$(read_val "memcpy_stream_swizzled" "MEMCPY_SWIZZLED_INNER_VAL") \
MEMCPY_SWIZZLED_PIXELS_MB_VAL=$(read_val "memcpy_stream_swizzled" "MEMCPY_SWIZZLED_PIXELS_MB_VAL") \
MEMCPY_STREAM_SWIZZLED_VERIFY=$(read_val "memcpy_stream_swizzled" "MEMCPY_STREAM_SWIZZLED_VERIFY") \
ENABLE_MEMCPY_STREAM_SWIZZLED=$(read_val "memcpy_stream_swizzled" "ENABLE_MEMCPY_STREAM_SWIZZLED")