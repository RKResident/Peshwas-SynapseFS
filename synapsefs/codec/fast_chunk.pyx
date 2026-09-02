# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport uint8_t, uint16_t, uint64_t
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_SIMPLE
cimport numpy as cnp
import numpy as np
cnp.import_array()

# Fast SIMD / C unshuffle for 16-bit elements (width == 2)
cdef void _unshuffle_w2_c(const uint8_t* src, uint16_t* dst, size_t num_elements) noexcept nogil:
    cdef size_t i
    cdef const uint8_t* p0 = src
    cdef const uint8_t* p1 = src + num_elements
    cdef uint8_t* d = <uint8_t*>dst

    # Interleave low and high bytes directly into destination
    for i in range(num_elements):
        d[2 * i]     = p0[i]
        d[2 * i + 1] = p1[i]

# In-place modular addition: base + residual (width == 2)
cdef void _add_residual_w2_c(const uint16_t* base, const uint16_t* res, uint16_t* dst, size_t n) noexcept nogil:
    cdef size_t i
    for i in range(n):
        dst[i] = base[i] + res[i]

def fast_unshuffle_w2(const uint8_t[::1] stream):
    """Zero-overhead unshuffle for 16-bit streams returning uint16 array."""
    cdef size_t total_bytes = stream.shape[0]
    cdef size_t num_elements = total_bytes // 2

    # Preallocate output array directly
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] out = np.empty(num_elements, dtype=np.uint16)
    cdef uint16_t* out_ptr = <uint16_t*>cnp.PyArray_DATA(out)
    cdef const uint8_t* src_ptr = &stream[0]

    with nogil:
        _unshuffle_w2_c(src_ptr, out_ptr, num_elements)

    return out

def fast_decode_delta_shuffle_w2(const uint8_t[::1] stream, cnp.ndarray base_arr):
    """Combined unshuffle + modular add with GIL released."""
    cdef size_t total_bytes = stream.shape[0]
    cdef size_t num_elements = total_bytes // 2

    cdef cnp.ndarray[cnp.uint16_t, ndim=1] out = np.empty(num_elements, dtype=np.uint16)
    cdef uint16_t* out_ptr = <uint16_t*>cnp.PyArray_DATA(out)
    cdef const uint8_t* src_ptr = &stream[0]

    # Ensure base is contiguous uint16
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] b = np.ascontiguousarray(base_arr).view(np.uint16).reshape(-1)
    cdef const uint16_t* base_ptr = <const uint16_t*>cnp.PyArray_DATA(b)

    with nogil:
        # 1. Unshuffle into output buffer
        _unshuffle_w2_c(src_ptr, out_ptr, num_elements)
        # 2. In-place add base: out = base + out
        _add_residual_w2_c(base_ptr, out_ptr, out_ptr, num_elements)

    return out