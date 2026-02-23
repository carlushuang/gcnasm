# build and install
```
#build
python3 setup.py develop

# or
python3 setup.py install
```
This will build & install the project

```
#uninstall
pip3 uninstall warp_bitonic_sort
```

```
#clean
python3 setup.py clean --all
```
after above instruction, `build/` will be deleted 

# CUDAExtension on HIP practice
- kernel is `csrc/warp_bitonic_sort.hpp` and its header, they are torch independent. This could be `.hip` or `.cu` surfix, cause anyway the cpp_extension will hipify it. better not use `.cpp` cause [cpp_extension here](https://github.com/pytorch/pytorch/blob/1927783aa3ad676db6f4c34fc77ef3825a4e2ed5/torch/utils/cpp_extension.py#L750) will check the surfix and apply different compile flag.
- torch adaptor(api) is `src/torch_api.cpp`. We need to use `at::Tensor` to accept tensor from python side to c++ side within this cpp file. better use `.cpp` here.
- `hipify` always happen for every `sources` file. it will append `_hip.<original_surfix>` to the hipified file. better add `.gitignore` (they call `os.getcwd()` [here](https://github.com/pytorch/pytorch/blob/322091d8d8542a0cbff524306029bef4d7338747/torch/utils/cpp_extension.py#L1375) to decide where to put the hipified file)
- better not name your source file with `<some_name>_hip.<some_surfix>`
- prefer to use `CUDAExtension` from `torch.utils.cpp_extension ` to construct the cpp extension. It's too complicated to use `Pybind11Extension` from `pybind11` or `Extension` from`setuptool` directly to rewrite a lot logics.
- better add `--offload-arch` inside `extra_compile_args` to specify current build gfx arch, otherwise it will build for every gfx arch (in case e.g. wave_size is arch dependent and you hardcoded them, otherwise need to make sure every arch can compile)
