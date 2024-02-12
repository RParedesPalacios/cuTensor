#include <cstdio>
#include <string>
#include <stdexcept>
#include <iostream>

#include "gpu.h"
#include "gpu_ops.h"

__global__ void gpu_sum_(float* a, float *b, float *c, long int size, bool inc){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        if (inc) 
            c[thread_id_x]+=a[thread_id_x]+b[thread_id_x];
        else 
            c[thread_id_x]=a[thread_id_x]+b[thread_id_x];            
    }
}


void gpu_sum(float *ptrA, float *ptrB, float *ptrC, long int size, int device, bool inc)
{
    cudaSetDevice(device);

    setDims(size);

    gpu_sum_<<<dimGrid,dimBlock>>>(ptrA,ptrB,ptrC,size,inc);
    check_cuda(cudaDeviceSynchronize(),"gpu_sum_");
}