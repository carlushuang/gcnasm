## warp sort
this example implement warp level sort using dpp instructions

## build/run
```
sh rebuild.sh # build
./build/warp_sort.exe
```
will have output like:
```
[origin-2]2.129 8.371
[sorted-2]8.371 2.129
--------------------- ordered
[origin-4]0.829 6.186 5.900 2.100
[sorted-4]6.186 5.900 2.100 0.829
--------------------- ordered
[origin-8]4.871 1.886 7.014 1.457 2.614 7.329 1.843 1.486
[sorted-8]7.329 7.014 4.871 2.614 1.886 1.843 1.486 1.457
--------------------- ordered
```

this example rely on [ck](https://github.com/ROCm/composable_kernel/), please modify `CK_DIR` inside `rebuild.sh` before build
