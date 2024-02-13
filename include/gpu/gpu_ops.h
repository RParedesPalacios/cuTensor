#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>


// MAX THREADS PER BLOCK
#define MAX_TPB 1024
#define setDims(size) int setdim_r,setdim_c;setdim_r=(size/MAX_TPB);if (setdim_r==0) {setdim_r=1;setdim_c=size;}else {if (size%MAX_TPB) setdim_r++;setdim_c=MAX_TPB;}dim3 dimGrid(setdim_r);dim3 dimBlock(setdim_c);

// sum of two tensor in gpu
// C=A+B or
// C+=A+B
void gpu_sum(float *ptrA, float *ptrB, float *ptrC, long int Asize, long int Bsize,int device, bool inc);

// matrix multiplication of two tensor in gpu
// C=A*B
void gpu_mult2D(float *ptrA, float *ptrB, float *ptrC, int m, int n, int k, int device);