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

void gpu_sum(float *ptrA, float *ptrB, float *ptrC, long int Asize, long int Bsize, int device, bool inc)
{
    cudaSetDevice(device);
    setDims(Bsize);

    int m=Asize/Bsize;
    for(int i=0;i<m;i++){
        gpu_sum_<<<dimGrid,dimBlock>>>(ptrA+i*Bsize,ptrB,ptrC+i*Bsize,Bsize,inc);
        check_cuda(cudaDeviceSynchronize(),"gpu_sum_");
    }
}

__global__ void gpu_sumf_(float* a, float *c, long int size, float s, int device){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        c[thread_id_x]=a[thread_id_x]+s;
               
    }
}
void gpu_sumf(float *ptrA, float *ptrC, long int size, float s, int device)
{
    cudaSetDevice(device);
    setDims(size);
    gpu_sumf_<<<dimGrid,dimBlock>>>(ptrA,ptrC,size,s,device);
    check_cuda(cudaDeviceSynchronize(),"gpu_sumf_");
}

// gpu mult2D C=A*B ussin cuBLAS taking into account that the matrices are stored in row-major order
void gpu_mult2D(float *ptrA, float *ptrB, float *ptrC, int m, int n, int k, int device) // m=A0,n=A1,k=B1
{
    cudaSetDevice(device);
    
    float alpha = 1.0;
    float beta = 0.0;

    check_cublas(cublasSgemm(hcublas[device], CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, ptrB, k, ptrA, n, &beta, ptrC, k), "cublasSgemm");
}

__global__ void gpu_elementwise_product_(float *a, float *b, float *c, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        c[thread_id_x]=a[thread_id_x]*b[thread_id_x];
    }
}   

void gpu_elementwise_product(float *ptrA, float *ptrB, float *ptrC, long int size, int device)
{
    cudaSetDevice(device);
    setDims(size);
    gpu_elementwise_product_<<<dimGrid,dimBlock>>>(ptrA,ptrB,ptrC,size);
    check_cuda(cudaDeviceSynchronize(),"elementwise_product");
}

// gpu scalar multiplication
__global__ void gpu_mult_(float *a, float *c, long int size, float s){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        c[thread_id_x]=a[thread_id_x]*s;
    }
}

void gpu_mult(float *ptrA, float *ptrC,long int size,float s,int device)
{
    cudaSetDevice(device);
    setDims(size);
    gpu_mult_<<<dimGrid,dimBlock>>>(ptrA,ptrC,size,s);
    check_cuda(cudaDeviceSynchronize(),"gpu_mult_");
}

__global__ void gpu_inv_(float *a, float *c, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        c[thread_id_x]=1.0/a[thread_id_x];
    }
}

void gpu_inv(float *ptrA, float *ptrC, long int size, int device)
{
    cudaSetDevice(device);
    setDims(size);
    
    gpu_inv_<<<dimGrid,dimBlock>>>(ptrA,ptrC,size);    
    check_cuda(cudaDeviceSynchronize(),"gpu_inv_");

}


__global__ void gpu_pow_(float *a, float *c, long int size, float s){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        c[thread_id_x]=pow(a[thread_id_x],s);
    }
}

void gpu_pow(float *ptrA, float *ptrC, long int size, float s, int device)
{
    cudaSetDevice(device);
    setDims(size);
    gpu_pow_<<<dimGrid,dimBlock>>>(ptrA,ptrC,size,s);
    check_cuda(cudaDeviceSynchronize(),"gpu_pow_");
}
