### test tanh
```
# build on rocm
sh rebuild.sh rocm

# build on cuda, make sure at least compute_75,sm_75 to use fast tanh
sh rebuild.sh cuda
```

will result in executable in `build/` folder
