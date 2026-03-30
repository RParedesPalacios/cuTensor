#include <cstdio>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cstddef>

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

static constexpr size_t LT_WORKSPACE_BYTES = 1 << 25; // 32 MiB per device (lazy)
static void* lt_workspace[64] = {nullptr};

static void* get_lt_workspace(int device, size_t &workspace_size)
{
    workspace_size = 0;
    if (device < 0 || device >= 64) return nullptr;
    if (lt_workspace[device] == nullptr) {
        cudaError_t err = cudaMalloc(&(lt_workspace[device]), LT_WORKSPACE_BYTES);
        if (err != cudaSuccess) {
            lt_workspace[device] = nullptr;
            return nullptr;
        }
    }
    workspace_size = LT_WORKSPACE_BYTES;
    return lt_workspace[device];
}

static bool gpu_mult2D_lt(float *ptrA, float *ptrB, float *ptrC, int m, int n, int k, int device)
{
    // Row-major trick:
    // C_row(m,k)=A_row(m,n)*B_row(n,k) <=> C_col(k,m)=B_col(k,n)*A_col(n,m)
    int m_col = k;
    int n_col = m;
    int k_col = n;
    int lda = k;
    int ldb = n;
    int ldc = k;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasLtMatmulDesc_t operation_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic_result = {};
    int returned_results = 0;
    cublasOperation_t op = CUBLAS_OP_N;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    size_t workspace_size = 0;
    void *workspace = nullptr;

    status = cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulDescSetAttribute(
        operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)
    );
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulDescSetAttribute(
        operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)
    );
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_32F, m_col, k_col, lda); // ptrB
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_32F, k_col, n_col, ldb); // ptrA
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, m_col, n_col, ldc); // ptrC
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    workspace = get_lt_workspace(device, workspace_size);
    status = cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)
    );
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulAlgoGetHeuristic(
        hcublaslt[device],
        operation_desc,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        preference,
        1,
        &heuristic_result,
        &returned_results
    );
    if (status != CUBLAS_STATUS_SUCCESS || returned_results == 0) goto cleanup;

    status = cublasLtMatmul(
        hcublaslt[device],
        operation_desc,
        &alpha,
        ptrB,
        a_desc,
        ptrA,
        b_desc,
        &beta,
        ptrC,
        c_desc,
        ptrC,
        c_desc,
        &heuristic_result.algo,
        workspace,
        workspace_size,
        0
    );

cleanup:
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
    if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (operation_desc) cublasLtMatmulDescDestroy(operation_desc);
    return status == CUBLAS_STATUS_SUCCESS;
}

// gpu mult2D C=A*B using cuBLAS taking into account that the matrices are stored in row-major order
void gpu_mult2D(float *ptrA, float *ptrB, float *ptrC, int m, int n, int k, int device) // m=A0,n=A1,k=B1
{
    if (device < 0 || device >= 64) {
        throw std::runtime_error("error setting device in gpu_mult2D");
    }
    check_cuda(cudaSetDevice(device), "gpu_mult2D_set_device");

    // Preferred path for newer CUDA stacks/GPUs.
    // If unsupported on this device/runtime, fallback is classic SGEMM.
    if (!gpu_mult2D_lt(ptrA, ptrB, ptrC, m, n, k, device)) {
        float alpha = 1.0f;
        float beta = 0.0f;
        check_cublas(
            cublasSgemm(
                hcublas[device], CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, &alpha, ptrB, k, ptrA, n, &beta, ptrC, k
            ),
            "cublasSgemm"
        );
    }
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
