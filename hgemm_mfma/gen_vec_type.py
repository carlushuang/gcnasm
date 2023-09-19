import sys
import os
import itertools
import numpy as np
import enum

OUTPUT_FILE = 'vector_type_predef.hpp'
STATIC_BUFFER_TYPE = 'static_buffer'
DEVICE_HOST_MACRO = 'DEVICE_HOST'

def gen_p2_array(n):
    i = 1
    rtn = []
    while i <= n:
        rtn.append(i)
        i = i * 2
    return rtn

class vector_type(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, fp):
        n = self.n
        fp.write(f'template <typename T> struct vector_type<T, {n}> {{\n')
        for d in gen_p2_array(n):
            if d == 1:
                fp.write(f'    using d1_t = T;\n')
            else:
                fp.write(f'    typedef T d{d}_t __attribute__((ext_vector_type({d})));\n')
        fp.write(f'    using type = d{n}_t;\n')
        fp.write(f'    union {{\n')
        fp.write(f'        type d{n}_;\n')
        for d in gen_p2_array(n):
            fp.write(f'        {STATIC_BUFFER_TYPE}<d{d}_t, {n // d}> d{d}x{n // d}_;\n')
        fp.write(f'    }} data_;\n')
        fp.write(f'    {DEVICE_HOST_MACRO} constexpr vector_type() : data_{{type{{0}}}} {{}}\n')
        fp.write(f'    {DEVICE_HOST_MACRO} constexpr vector_type(type v) : data_{{v}} {{}}\n')
        fp.write(f'    template<typename VEC> {DEVICE_HOST_MACRO} constexpr const auto& to_varray() const {{ return data_.d{n}_; }}\n')
        fp.write(f'    template<typename VEC> {DEVICE_HOST_MACRO} constexpr auto&       to_varray()       {{ return data_.d{n}_; }}\n')

        for d in gen_p2_array(n):
            fp.write(f'    template<> {DEVICE_HOST_MACRO} constexpr const auto& to_varray<d{d}_t>() const {{ return data_.d{d}x{n//d}_;}}\n')
            fp.write(f'    template<> {DEVICE_HOST_MACRO} constexpr auto&       to_varray<d{d}_t>()       {{ return data_.d{d}x{n//d}_;}}\n')
        fp.write(f'}};\n')

def gen(file_name):
    fp = None
    try:
        fp = open(file_name, "w")
    except IOError as e:
        print("can't open file:{}({})".format(file_name, e))
        sys.exit()
    fp.write(f'template <typename T, index_t N>\n')
    fp.write(f'struct vector_type;\n')
    fp.write(f'\n')
    fp.write(f'// clang-format off\n')
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        vector_type(n)(fp)
        fp.write(f'\n')
    fp.write(f'// clang-format on\n')

if __name__ == '__main__':
    output_file = OUTPUT_FILE
    if len(sys.argv) >= 2:
        output_file = sys.argv[1]
    gen(output_file)