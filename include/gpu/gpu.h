#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>


// MAX THREADS PER BLOCK
#define MAX_TPB 1024
#define setDims(size) int setdim_r,setdim_c;setdim_r=(size/MAX_TPB);if (setdim_r==0) {setdim_r=1;setdim_c=size;}else {if (size%MAX_TPB) setdim_r++;setdim_c=MAX_TPB;}dim3 dimGrid(setdim_r);dim3 dimBlock(setdim_c);

extern cublasHandle_t hcublas[64];

void check_cuda(cudaError_t err,const char *msg);
void check_cublas(cublasStatus_t status, const char *f);

void hw_info();
void gpu_init();

void gpu_set_device(int device);
int gpu_devices();

float* gpu_create_tensor(int dev,long int size);
void gpu_delete_tensor(int dev,float* p);
void gpu_copy_from(int device, long int size, float *src, float *dst);
void gpu_copy_to(int device, long int size, float *src, float *dst);

void gpu_fill(int device, long int size, float *ptr, float v);
void gpu_fill_void(int device, long int size, float *ptr);

void gpu_print_(int device, long int size, float *ptr);

void gpu_permute_(int device, long int size, int dim, int *strides, int *nstrides,int *perm, float *ptr);

