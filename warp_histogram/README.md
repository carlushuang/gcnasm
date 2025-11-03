# build and install
```
# need change setup.py for aiter/opus location
sh rebuild.sh
```

# test
```
#NOTE: change the torch path based on need
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/venv/lib/python3.12/site-packages/torch/lib python3 test/test.py
```
