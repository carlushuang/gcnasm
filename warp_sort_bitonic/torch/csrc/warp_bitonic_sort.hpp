#pragma once

// pure C like host API
void warp_bitonic_sort_kernel(void* i_ptr, void* o_ptr, int num_element, int is_descending);
