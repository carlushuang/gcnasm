## warp sort
this example implement warp level sort using dpp instructions

## build/run
```
sh rebuild.sh # build
./build/warp_sort.exe
```
will have output like:
```
[origin-2]1.786 4.257
[sorted-2]4.257 1.786
-------------------------------------- ordered
[origin-4]7.529 3.500 8.400 6.900
[sorted-4]8.400 7.529 6.900 3.500
-------------------------------------- ordered
[origin-8]0.014 1.257 1.914 6.300 0.457 2.671 1.757 4.114
[sorted-8]6.300 4.114 2.671 1.914 1.757 1.257 0.457 0.014
-------------------------------------- ordered
[origin-16]3.071 4.829 3.329 3.786 4.371 1.086 1.329 1.914 4.257 4.143 6.543 1.514 4.986 4.186 1.957 7.600
[sorted-16]7.600 6.543 4.986 4.829 4.371 4.257 4.186 4.143 3.786 3.329 3.071 1.957 1.914 1.514 1.329 1.086
-------------------------------------- ordered
```

this example rely on [ck](https://github.com/ROCm/composable_kernel/), please modify `CK_DIR` inside `rebuild.sh` before build
