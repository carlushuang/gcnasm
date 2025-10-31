pip3 uninstall -y warp_histogram
python3 setup.py clean --all

rm -rf tmp
mkdir tmp && cd tmp
python3 ../setup.py install
