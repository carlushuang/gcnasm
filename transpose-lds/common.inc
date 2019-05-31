.macro .g_load32_col v_dst32, v_os, s_src
    .cnt=0
    .rept 32
        global_load_dword v[\v_dst32+.cnt], v[\v_os:\v_os+1], s[\s_src:\s_src+1], offset:32*4*.cnt
        .cnt = .cnt+1
    .endr
.endm

.macro .g_store32_col s_dst, v_os, v_src32
    .cnt=0
    .rept 32
        global_store_dword v[\v_os:\v_os+1], v[\v_src32+.cnt], s[\s_dst:\s_dst+1], offset:32*4*.cnt
        .cnt = .cnt+1
    .endr
.endm

.macro ._ds_store_n dst, src, od, os, n
    .cnt = 0
    .rept \n
        ds_write_b32 v[\dst], v[\src+\os*.cnt], offset:4*\od*.cnt
        .cnt = .cnt+1
    .endr
.endm
.macro ._ds_load_n dst, src, od, os, n
    .cnt = 0
    .rept \n
        ds_read_b32 v[\dst+\od*.cnt], v[\src], offset:4*\os*.cnt
        .cnt = .cnt+1
    .endr
.endm

.macro .ds_store16 dst, src, od, os
    ._ds_store_n \dst, \src, \od, \os, 16
.endm

.macro .ds_load16 dst, src, od, os
    ._ds_load_n \dst, \src, \od, \os, 16
.endm

.macro .ds_store32 dst, src, od, os
    ._ds_store_n \dst, \src, \od, \os, 32
.endm

.macro .ds_load32 dst, src, od, os
    ._ds_load_n \dst, \src, \od, \os, 32
.endm
