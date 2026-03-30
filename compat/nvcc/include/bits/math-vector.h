/* CUDA/NVCC ARM64 compatibility shim.
   NVCC front-end in some CUDA versions cannot parse the aarch64
   vector typedefs used by glibc's bits/math-vector.h. For CUDA
   compilations we only need the default libm SIMD declaration stubs. */

#ifndef _MATH_H
# error "Never include <bits/math-vector.h> directly; include <math.h> instead."
#endif

#include <bits/libm-simd-decl-stubs.h>
