# build and install
```
#build
python3 setup.py develop

# or
python3 setup.py install
```
This will build the project and under current `build/` directory as temp folder, copy a `.so` file under current directory.

```
#uninstall
pip3 uninstall warp_bitonic_sort
```

```
#clean
python3 setup.py clean --all
```
after above instruction, `build/` will be deleted 
