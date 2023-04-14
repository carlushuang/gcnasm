# hip jit
kernel is emit at runtime, compile at runtime, and use `dlopen` to dynamically load the compiled module(so)

# compile the host code
```
#under current folder do:
sh rebuild.sh
```
will compile the code under `tmp/memcpy_driver.exe`

# run the host to emit/build/launch kernel
run `./tmp/memcpy_driver.exe`, this will do:
1. emit the kernel to `tmp/memcpy_kernel.hip.cc`
2. compile it into `tmp/memcpy_kernel.so`
3. dlopen this so and launch the kernel inside.
