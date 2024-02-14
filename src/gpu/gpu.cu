#include <cstdio>
#include <string>
#include <stdexcept>
#include <iostream>

#include "../../include/gpu/gpu.h"

cublasHandle_t hcublas[64];
curandGenerator_t random_generator[64];
cublasStatus_t bstatus;
curandStatus_t rstatus;


void check_cuda(cudaError_t err,const char *msg)
{
    if(err!=cudaSuccess)
    {
        std::string error_type = cudaGetErrorString(err);
        std::string text = "[CUDA ERROR]: " + error_type + " ("+ std::to_string(err) + ") raised in " + std::string(msg) + " | (check_cuda)";
        throw std::runtime_error(text);
    }

}
void check_cublas(cublasStatus_t status, const char *f)
{
    if ( status!=  CUBLAS_STATUS_SUCCESS)
    {
        std::string text = "error in cublas execution in " + std::string(f) + " | (check_cublas)";
        throw std::runtime_error(text);
    }
}
void hw_info()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i=0;i<nDevices;i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);
        fprintf(stderr,"GPU device %d, %s\n",i,prop.name);
    }
}

void gpu_init()
{

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for (int i=0;i<nDevices;i++)
    {
        cudaSetDevice(i);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);

        //fprintf(stderr,"GPU device %d, %s, ready\n",i,prop.name);

        check_cublas(cublasCreate(&(hcublas[i])),"cublasCreate");
        //fprintf(stderr,"CuBLAS running on GPU device %s\n",prop.name);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

void gpu_set_device(int device)
{
    cudaSetDevice(device);
}

int gpu_devices()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    return nDevices;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

float* gpu_create_tensor(int dev,long int size)
{
    float* devicePointer;
    if (cudaSetDevice(dev)!=cudaSuccess)
    {
        std::string text = "error setting device "+std::to_string(dev)+" in gpu_create_tensor | (gpu_create_tensor)";
        throw std::runtime_error(text);
    }
    check_cuda(cudaMalloc((void**)&devicePointer,size*sizeof(float)),"create_tensor");
    return devicePointer;
}

void gpu_delete_tensor(int dev, float* p)
{
    cudaSetDevice(dev);
    check_cuda(cudaFree(p),"delete_tensor");
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
void gpu_copy_from(int device, long int size, float *ptr, float *cpu_ptr)
{
    cudaSetDevice(device);
    check_cuda(cudaMemcpy(cpu_ptr,ptr,size*sizeof(float),cudaMemcpyDeviceToHost),"gpu_copy");
}

void gpu_copy_to(int device, long int size, float *cpu_ptr, float *ptr)
{
    cudaSetDevice(device);
    //copy to device
    check_cuda(cudaMemcpy(ptr,cpu_ptr,size*sizeof(float),cudaMemcpyHostToDevice),"gpu_copy");
}

/////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void fill_(float* a, float v, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        a[thread_id_x]=v;
    }
}
void gpu_fill(int device, long int size, float *ptr, float v) {
    cudaSetDevice(device);

    setDims(size);

    fill_<<<dimGrid,dimBlock>>>(ptr,v,size);
    check_cuda(cudaDeviceSynchronize(),"set");
}

/////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void fill_void_(float* a, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        a[thread_id_x]=thread_id_x;
    }
}
void gpu_fill_void(int device, long int size, float *ptr) {
    cudaSetDevice(device);

    setDims(size);

    fill_void_<<<dimGrid,dimBlock>>>(ptr,size);
    check_cuda(cudaDeviceSynchronize(),"set");
}


__global__ void print_(float* a, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        printf("%f ",a[thread_id_x]);
    }
}
void gpu_print_(int device, long int size, float *ptr) {
    cudaSetDevice(device);

    setDims(size);

    //for(int i=0;i<size;i++) 
      //printf("%f ",ptr[i]);

    //print_<<<dimGrid,dimBlock>>>(ptr,size);
    //check_cuda(cudaDeviceSynchronize(),"print");
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fill_strides_(float* a, float *b, int dim, int *strides,int *nstrides,int *perm,long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        int offset = 0;
        int pos=thread_id_x;
        for (int i=0;i<dim-1;i++){
            offset += (pos/strides[i])*nstrides[perm[i]];
            pos=pos%strides[i];
        }
        offset+=pos*nstrides[perm[dim-1]];
    
        b[offset]=a[thread_id_x];
    }
}
void gpu_permute_(int device, long int size, int dim, int *strides, int *nstrides,int *perm, float *ptr)
{   
    float *ptr2;

    cudaSetDevice(device); 
    setDims(size);
    
    // create memory for ptr2
    check_cuda(cudaMalloc((void**)&ptr2,size*sizeof(float)),"gpu_contiguous");
    // copy the strides and perm to device memory
    int *strides_d; 
    int *nstrides_d;
    int *perm_d;
    check_cuda(cudaMalloc((void**)&strides_d,dim*sizeof(int)),"gpu_contiguous");
    check_cuda(cudaMalloc((void**)&nstrides_d,dim*sizeof(int)),"gpu_contiguous");
    check_cuda(cudaMalloc((void**)&perm_d,dim*sizeof(int)),"gpu_contiguous");

    check_cuda(cudaMemcpy(strides_d,strides,dim*sizeof(int),cudaMemcpyHostToDevice),"gpu_contiguous");
    check_cuda(cudaMemcpy(nstrides_d,nstrides,dim*sizeof(int),cudaMemcpyHostToDevice),"gpu_contiguous");
    check_cuda(cudaMemcpy(perm_d,perm,dim*sizeof(int),cudaMemcpyHostToDevice),"gpu_contiguous");


    fill_strides_<<<dimGrid,dimBlock>>>(ptr,ptr2,dim,strides_d,nstrides_d,perm_d,size);
    check_cuda(cudaDeviceSynchronize(),"gpu_contiguous");

    // copy ptr2 to ptr and delete ptr2
    check_cuda(cudaMemcpy(ptr,ptr2,size*sizeof(float),cudaMemcpyDeviceToDevice),"gpu_contiguous");
    check_cuda(cudaFree(ptr2),"gpu_contiguous");
    check_cuda(cudaFree(strides_d),"gpu_contiguous");    
    check_cuda(cudaFree(nstrides_d),"gpu_contiguous");
    check_cuda(cudaFree(perm_d),"gpu_contiguous");
}
