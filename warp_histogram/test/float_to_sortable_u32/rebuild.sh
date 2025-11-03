CLANG=/opt/rocm/llvm/bin/clang++

rm -rf o.exe
$CLANG -Wall -std=c++17 -O3 main.cpp -o o.exe
