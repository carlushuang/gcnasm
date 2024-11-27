#!/bin/bash

# config_file="config.json"
config_file="${CONFIG_FILE:-config.json}"


read_val() {
    jq -r ".$1.$2" $config_file
}

enable_config_json=0
any_option_set=0
block_size=1024
grid_size=0
row_per_thread=1
padding=0
mem_op="all"
verification=0
unroll=4
inner=1
array_size=1750
nontemp=0
arch="gfx942"

while getopts ":b:g:u:n:i:k:s:r:p:a:v:h" opt; do
  case $opt in
    b) block_size="$OPTARG" 
       any_option_set=1
       ;;
    g) grid_size="$OPTARG" 
       any_option_set=1
       ;;
    u) unroll="$OPTARG" 
       any_option_set=1
       ;;
    n) nontemp="$OPTARG" 
       any_option_set=1
       ;;
    i) inner="$OPTARG" 
       any_option_set=1
       ;;
    k) mem_op="$OPTARG" 
       any_option_set=1
       ;;
    s) array_size="$OPTARG" 
       any_option_set=1
       ;;
    r) row_per_thread="$OPTARG"
       any_option_set=1
       ;;
    p) padding="$OPTARG"
       any_option_set=1
       ;;
    a) arch="$OPTARG" 
       any_option_set=1
       ;;
    v) verification="$OPTARG" 
       any_option_set=1
       ;;
    h) 
       printf "Usage: ./build.sh [options]\n"
       printf "   -b Block_Size (defualt is 1024)\n"
       printf "   -g Grid_Size (default is same as the CU number)\n"
       printf "   -u UNROLL (default is 4)\n"
       printf "   -i INNER (default is 1)\n"
       printf "   -n Nontemporal load/store (Enable by 1, default is 0)\n"
       printf "   -k Memory test type: (By default, all will be executed.)\n\t\t memcpy_persistent\n\t\t memread_stream\n\t\t memwrite_stream\n\t\t memcpy_stream\n\t\t memcpy_stream_async\n\t\t memcpy_stream_async_inplace\n\t\t memcpy_stream_swizzled\n"
       printf "   -a Offload architecture (default is gfx942)\n"
       printf "   -v Verification (Enable by 1, default is 0.)\n"
       printf "   -s Array Size MB (default is 1750)\n"
       printf "   -r Row per thread (default is 1)\n"
       printf "   -p Padding (default is 0)\n"
       printf "   -h Show this help message\n"
       exit 0
       ;;
    \?) printf "Invalid option: -$OPTARG\n" >&2
        exit 1
        ;;
    :) printf "Option -$OPTARG requires an argument.\n" >&2
       exit 1
       ;;
  esac
done

#make -f ./HIP.make clean
rm -f membench

# Check if any option was set, and if not, set enable_config_json=1
if [ $any_option_set -eq 0 ]; then
  enable_config_json=1
fi

if [ $enable_config_json -eq 1 ]; then
  echo "****** Set options by config.json ****** "

  while IFS=$'\n' read -r key_value
  do
    key=$(echo $key_value | cut -d '=' -f 1)
    value=$(echo $key_value | cut -d '=' -f 2)
    export $key=$value
  done < <(jq -r 'to_entries[] | (.key as $sname | .value | to_entries[] | "\(.key)=\(.value)")' $config_file)

  $@ make -f ./HIP.make \
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
  WRITE_BS_VAL=$(read_val "memwrite_stream" "WRITE_BS_VAL") \
  WRITE_GS_VAL=$(read_val "memwrite_stream" "WRITE_GS_VAL") \
  WRITE_VERIFY=$(read_val "memwrite_stream" "WRITE_VERIFY") \
  WRITE_UNROLL_VAL=$(read_val "memwrite_stream" "WRITE_UNROLL_VAL") \
  WRITE_PIXELS_MB_VAL=$(read_val "memwrite_stream" "WRITE_PIXELS_MB_VAL") \
  WRITE_STREAM_NONTEMP=$(read_val "memwrite_stream" "WRITE_STREAM_NONTEMP") \
  ENABLE_MEMWRITE_STREAM=$(read_val "memwrite_stream" "ENABLE_MEMWRITE_STREAM") \
  MEMCPY_BS_VAL=$(read_val "memcpy_stream" "MEMCPY_BS_VAL") \
  MEMCPY_GS_VAL=$(read_val "memcpy_stream" "MEMCPY_GS_VAL") \
  MEMCPY_UNROLL_VAL=$(read_val "memcpy_stream" "MEMCPY_UNROLL_VAL") \
  MEMCPY_ROW_PER_THREAD_VAL=$(read_val "memcpy_stream" "MEMCPY_ROW_PER_THREAD_VAL") \
  MEMCPY_PADDING_VAL=$(read_val "memcpy_stream" "MEMCPY_PADDING_VAL") \
  MEMCPY_PIXELS_MB_VAL=$(read_val "memcpy_stream" "MEMCPY_PIXELS_MB_VAL") \
  MEMCPY_STREAM_NONTEMP=$(read_val "memcpy_stream" "MEMCPY_STREAM_NONTEMP") \
  MEMCPY_STREAM_VERIFY=$(read_val "memcpy_stream" "MEMCPY_STREAM_VERIFY") \
  ENABLE_MEMCPY_STREAM=$(read_val "memcpy_stream" "ENABLE_MEMCPY_STREAM") \
  MEMCPY_ASYNC_BS_VAL=$(read_val "memcpy_stream_async" "MEMCPY_ASYNC_BS_VAL") \
  MEMCPY_ASYNC_GS_VAL=$(read_val "memcpy_stream_async" "MEMCPY_ASYNC_GS_VAL") \
  MEMCPY_ASYNC_UNROLL_VAL=$(read_val "memcpy_stream_async" "MEMCPY_ASYNC_UNROLL_VAL") \
  MEMCPY_ASYNC_ROW_PER_THREAD_VAL=$(read_val "memcpy_stream_async" "MEMCPY_ASYNC_ROW_PER_THREAD_VAL") \
  MEMCPY_ASYNC_PADDING_VAL=$(read_val "memcpy_stream_async" "MEMCPY_ASYNC_PADDING_VAL") \
  MEMCPY_ASYNC_PIXELS_MB_VAL=$(read_val "memcpy_stream_async" "MEMCPY_ASYNC_PIXELS_MB_VAL") \
  MEMCPY_STREAM_ASYNC_VERIFY=$(read_val "memcpy_stream_async" "MEMCPY_STREAM_ASYNC_VERIFY") \
  ENABLE_MEMCPY_STREAM_ASYNC=$(read_val "memcpy_stream_async" "ENABLE_MEMCPY_STREAM_ASYNC") \
  MEMCPY_ASYNC_INPLACE_BS_VAL=$(read_val "memcpy_stream_async_inplace" "MEMCPY_ASYNC_INPLACE_BS_VAL") \
  MEMCPY_ASYNC_INPLACE_GS_VAL=$(read_val "memcpy_stream_async_inplace" "MEMCPY_ASYNC_INPLACE_GS_VAL") \
  MEMCPY_ASYNC_INPLACE_UNROLL_VAL=$(read_val "memcpy_stream_async_inplace" "MEMCPY_ASYNC_INPLACE_UNROLL_VAL") \
  MEMCPY_ASYNC_INPLACE_ROW_PER_THREAD_VAL=$(read_val "memcpy_stream_async_inplace" "MEMCPY_ASYNC_INPLACE_ROW_PER_THREAD_VAL") \
  MEMCPY_ASYNC_INPLACE_PADDING_VAL=$(read_val "memcpy_stream_async_inplace" "MEMCPY_ASYNC_INPLACE_PADDING_VAL") \
  MEMCPY_ASYNC_INPLACE_PIXELS_MB_VAL=$(read_val "memcpy_stream_async_inplace" "MEMCPY_ASYNC_INPLACE_PIXELS_MB_VAL") \
  MEMCPY_STREAM_ASYNC_INPLACE_VERIFY=$(read_val "memcpy_stream_async_inplace" "MEMCPY_STREAM_ASYNC_INPLACE_VERIFY") \
  ENABLE_MEMCPY_STREAM_ASYNC_INPLACE=$(read_val "memcpy_stream_async_inplace" "ENABLE_MEMCPY_STREAM_ASYNC_INPLACE") \
  MEMCPY_SWIZZLED_BS_VAL=$(read_val "memcpy_stream_swizzled" "MEMCPY_SWIZZLED_BS_VAL") \
  MEMCPY_SWIZZLED_GS_VAL=$(read_val "memcpy_stream_swizzled" "MEMCPY_SWIZZLED_GS_VAL") \
  MEMCPY_SWIZZLED_CHUNKS_VAL=$(read_val "memcpy_stream_swizzled" "MEMCPY_SWIZZLED_CHUNKS_VAL") \
  MEMCPY_SWIZZLED_INNER_VAL=$(read_val "memcpy_stream_swizzled" "MEMCPY_SWIZZLED_INNER_VAL") \
  MEMCPY_SWIZZLED_PIXELS_MB_VAL=$(read_val "memcpy_stream_swizzled" "MEMCPY_SWIZZLED_PIXELS_MB_VAL") \
  MEMCPY_STREAM_SWIZZLED_VERIFY=$(read_val "memcpy_stream_swizzled" "MEMCPY_STREAM_SWIZZLED_VERIFY") \
  ENABLE_MEMCPY_STREAM_SWIZZLED=$(read_val "memcpy_stream_swizzled" "ENABLE_MEMCPY_STREAM_SWIZZLED") 

else
  if [ $grid_size -eq 0 ]; then
    hipcc -o query_cu query_cu.cpp
    grid_size=$(./query_cu)
    rm query_cu
    echo $grid_size
  fi


  base_size=$(awk -v arr_sz="$array_size" 'BEGIN {print (arr_sz / 4)}')

  i=1
  PIXELS_MB=$(awk -v block="$block_size" -v grid="$grid_size" -v unroll="$unroll" -v inner="$inner" -v row_per_thread="$row_per_thread" -v i="$i" 'BEGIN {print (((block * grid * 16 / 4 / 1024 / 1024) * unroll * inner * row_per_thread) * i) }')
  while awk -v px_mb="$PIXELS_MB" -v arr_sz_base="$base_size" 'BEGIN {exit !(px_mb < arr_sz_base)}'
  do
    PIXELS_MB=$(awk -v block="$block_size" -v grid="$grid_size" -v unroll="$unroll" -v inner="$inner" -v row_per_thread="$row_per_thread" -v i="$i" 'BEGIN {print (((block * grid * 16 / 4 / 1024 / 1024) * unroll * inner * row_per_thread) * i) }')
    i=$((i + 1))
  done


  command_string=""
  if [[ "$mem_op" == "all" ]] || [[ "$mem_op" == "memcpy_persistent" ]]; then
    command_string=" $command_string MEMCPY_PERSISTENT_BS_VAL=$block_size" 
    command_string=" $command_string MEMCPY_PERSISTENT_GS_VAL=$grid_size"
    command_string=" $command_string MEMCPY_PERSISTENT_VERIFY=$verification"
    command_string=" $command_string MEMCPY_PERSISTENT_UNROLL_VAL=$unroll"
    command_string=" $command_string MEMCPY_PERSISTENT_PIXELS_MB_VAL=$PIXELS_MB"
    command_string=" $command_string ENABLE_MEMCPY_PERSISTENT=1"
  fi

  if [[ "$mem_op" == "all" ]] || [[ "$mem_op" == "memread_stream" ]]; then
    command_string=" $command_string READ_BS_VAL=$block_size"
    command_string=" $command_string READ_GS_VAL=$grid_size"
    command_string=" $command_string READ_VERIFY=$verification"
    command_string=" $command_string READ_UNROLL_VAL=$unroll"
    command_string=" $command_string READ_PIXELS_MB_VAL=$PIXELS_MB"
    command_string=" $command_string READ_STREAM_NONTEMP=$nontemp"
    command_string=" $command_string ENABLE_MEMREAD_STREAM=1"
  fi

  if [[ "$mem_op" == "all" ]] || [[ "$mem_op" == "memwrite_stream" ]]; then
    command_string=" $command_string WRITE_BS_VAL=$block_size"
    command_string=" $command_string WRITE_GS_VAL=$grid_size"
    command_string=" $command_string WRITE_VERIFY=$verification"
    command_string=" $command_string WRITE_UNROLL_VAL=$unroll"
    command_string=" $command_string WRITE_PIXELS_MB_VAL=$PIXELS_MB"
    command_string=" $command_string WRITE_STREAM_NONTEMP=$nontemp"
    command_string=" $command_string ENABLE_MEMWRITE_STREAM=1"
  fi

  if [[ "$mem_op" == "all" ]] || [[ "$mem_op" == "memcpy_stream" ]]; then
    command_string=" $command_string MEMCPY_BS_VAL=$block_size"
    command_string=" $command_string MEMCPY_GS_VAL=$grid_size"
    command_string=" $command_string MEMCPY_UNROLL_VAL=$unroll"
    command_string=" $command_string MEMCPY_PIXELS_MB_VAL=$PIXELS_MB"
    command_string=" $command_string MEMCPY_ROW_PER_THREAD_VAL=$row_per_thread"
    command_string=" $command_string MEMCPY_PADDING_VAL=$padding"
    command_string=" $command_string MEMCPY_STREAM_NONTEMP=$nontemp"
    command_string=" $command_string MEMCPY_STREAM_VERIFY=$verification"
    command_string=" $command_string ENABLE_MEMCPY_STREAM=1"
  fi

  if [[ "$mem_op" == "all" ]] || [[ "$mem_op" == "memcpy_stream_async" ]]; then
    command_string=" $command_string MEMCPY_ASYNC_BS_VAL=$block_size"
    command_string=" $command_string MEMCPY_ASYNC_GS_VAL=$grid_size"
    command_string=" $command_string MEMCPY_ASYNC_UNROLL_VAL=$unroll"
    command_string=" $command_string MEMCPY_ASYNC_PIXELS_MB_VAL=$PIXELS_MB"
    command_string=" $command_string MEMCPY_ASYNC_ROW_PER_THREAD_VAL=$row_per_thread"
    command_string=" $command_string MEMCPY_ASYNC_PADDING_VAL=$padding"
    command_string=" $command_string MEMCPY_STREAM_ASYNC_VERIFY=$verification"
    command_string=" $command_string ENABLE_MEMCPY_STREAM_ASYNC=1"
  fi

  if [[ "$mem_op" == "all" ]] || [[ "$mem_op" == "memcpy_stream_async_inplace" ]]; then
    command_string=" $command_string MEMCPY_ASYNC_INPLACE_BS_VAL=$block_size"
    command_string=" $command_string MEMCPY_ASYNC_INPLACE_GS_VAL=$grid_size"
    command_string=" $command_string MEMCPY_ASYNC_INPLACE_UNROLL_VAL=$unroll"
    command_string=" $command_string MEMCPY_ASYNC_INPLACE_PIXELS_MB_VAL=$PIXELS_MB"
    command_string=" $command_string MEMCPY_ASYNC_INPLACE_ROW_PER_THREAD_VAL=$row_per_thread"
    command_string=" $command_string MEMCPY_ASYNC_INPLACE_PADDING_VAL=$padding"
    command_string=" $command_string MEMCPY_STREAM_ASYNC_INPLACE_VERIFY=$verification"
    command_string=" $command_string ENABLE_MEMCPY_STREAM_ASYNC_INPLACE=1"
  fi

  if [[ "$mem_op" == "all" ]] || [[ "$mem_op" == "memcpy_stream_swizzled" ]]; then
    command_string=" $command_string MEMCPY_SWIZZLED_BS_VAL=$block_size"
    command_string=" $command_string MEMCPY_SWIZZLED_GS_VAL=$grid_size"
    command_string=" $command_string MEMCPY_SWIZZLED_CHUNKS_VAL=$unroll"
    command_string=" $command_string MEMCPY_SWIZZLED_INNER_VAL=$inner"
    command_string=" $command_string MEMCPY_SWIZZLED_PIXELS_MB_VAL=$PIXELS_MB"
    command_string=" $command_string MEMCPY_STREAM_SWIZZLED_VERIFY=$verification"
    command_string=" $command_string ENABLE_MEMCPY_STREAM_SWIZZLED=1"
  fi

  command_string="make -f ./HIP.make $command_string ARCH=$arch"

  echo $command_string

  eval "$command_string"

fi



