# memcpy 2d benchmark

This is a benchmark for memcpy-2d-1Dword example on gfx1030. You can specify matrix parameters (rows, cols, padding) by modifying params.config. 
Here rows and padding can pass in any value, while cols currently only support multiples of 2048 (2K).

## build and run
Go to the benchmark root and build by
'''
$ ./run.sh
'''
Then you can run by
'''
$ ./out.exe params.config
'''

## conclusion
I have tested in different paramter combination, which shows:
| Rows | Cols | Padding | GBPS |
| :--: | :--: | :-----: | :--: |
| 128  | 147456 | 1024    | 254.202 |
| 256  | 147456 | 1024    | 309.121 |
| 512  | 147456 | 1024    | 368.616 |
| 512  | 73728  | 1024    | 316.32  |
| 512  | 184320 | 1024    | 349.346 |
| 512  | 147456 | 0       | 365.193 |
| 512  | 147456 | 1       | 226.867 |
| 512  | 147456 | 8192    | 367.966 |
| 512  | 147456 | 32768   | 363.132 |