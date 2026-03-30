#include <cstdio>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <unordered_map>
#include <mutex>

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

static constexpr size_t LT_MAX_WORKSPACE_BYTES = 1 << 26; // 64 MiB cap per device
static void* lt_workspace[64] = {nullptr};
static size_t lt_workspace_bytes[64] = {0};
static bool cublas_math_mode_configured[64] = {false};
static std::mutex cublas_math_mode_mutex[64];

enum class MatmulMode {
    FP32,
    TF32
};

struct MatmulConfig {
    MatmulMode mode;
    bool debug;
};

static bool parse_env_flag(const char *value)
{
    if (value == nullptr) return false;
    std::string v(value);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

static MatmulMode parse_matmul_mode(const char *value)
{
    if (value == nullptr) return MatmulMode::FP32;
    std::string v(value);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
    if (v == "tf32") return MatmulMode::TF32;
    return MatmulMode::FP32;
}

static const char* matmul_mode_name(MatmulMode mode)
{
    return mode == MatmulMode::TF32 ? "tf32" : "fp32";
}

static MatmulConfig load_matmul_config()
{
    MatmulConfig cfg;
    cfg.mode = parse_matmul_mode(std::getenv("CUTENSOR_MATMUL_MODE"));
    cfg.debug = parse_env_flag(std::getenv("CUTENSOR_MATMUL_DEBUG"));
    if (cfg.debug) {
        fprintf(stderr, "[cuTensor][matmul] config mode=%s debug=1\n", matmul_mode_name(cfg.mode));
    }
    return cfg;
}

static const MatmulConfig& get_matmul_config()
{
    static MatmulConfig cfg = load_matmul_config();
    return cfg;
}

static cublasComputeType_t lt_compute_type(MatmulMode mode)
{
#ifdef CUBLAS_COMPUTE_32F_FAST_TF32
    if (mode == MatmulMode::TF32) return CUBLAS_COMPUTE_32F_FAST_TF32;
#endif
    return CUBLAS_COMPUTE_32F;
}

static cublasMath_t cublas_math_mode(MatmulMode mode)
{
#ifdef CUBLAS_TF32_TENSOR_OP_MATH
    if (mode == MatmulMode::TF32) return CUBLAS_TF32_TENSOR_OP_MATH;
#endif
    return CUBLAS_DEFAULT_MATH;
}

static void ensure_cublas_math_mode(int device, MatmulMode mode)
{
    if (device < 0 || device >= 64) return;
    std::lock_guard<std::mutex> lock(cublas_math_mode_mutex[device]);
    if (cublas_math_mode_configured[device]) return;
    check_cublas(cublasSetMathMode(hcublas[device], cublas_math_mode(mode)), "cublasSetMathMode");
    cublas_math_mode_configured[device] = true;
}

struct LtKey {
    int m_col;
    int n_col;
    int k_col;
    int lda;
    int ldb;
    int ldc;
    int mode;

    bool operator==(const LtKey &other) const {
        return m_col == other.m_col &&
               n_col == other.n_col &&
               k_col == other.k_col &&
               lda == other.lda &&
               ldb == other.ldb &&
               ldc == other.ldc &&
               mode == other.mode;
    }
};

struct LtKeyHash {
    size_t operator()(const LtKey &key) const {
        size_t h = static_cast<size_t>(key.m_col);
        h = h * 1315423911u + static_cast<size_t>(key.n_col);
        h = h * 1315423911u + static_cast<size_t>(key.k_col);
        h = h * 1315423911u + static_cast<size_t>(key.lda);
        h = h * 1315423911u + static_cast<size_t>(key.ldb);
        h = h * 1315423911u + static_cast<size_t>(key.ldc);
        h = h * 1315423911u + static_cast<size_t>(key.mode);
        return h;
    }
};

struct LtPlan {
    cublasLtMatmulDesc_t operation_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulAlgo_t algo = {};
    size_t workspace_bytes = 0;
};

struct LtCacheEntry {
    bool initialized = false;
    bool available = false;
    LtPlan plan;
};

static std::unordered_map<LtKey, LtCacheEntry, LtKeyHash> lt_plan_cache[64];
static std::mutex lt_cache_mutex[64];

static void destroy_lt_plan(LtPlan &plan)
{
    if (plan.c_desc) cublasLtMatrixLayoutDestroy(plan.c_desc);
    if (plan.b_desc) cublasLtMatrixLayoutDestroy(plan.b_desc);
    if (plan.a_desc) cublasLtMatrixLayoutDestroy(plan.a_desc);
    if (plan.operation_desc) cublasLtMatmulDescDestroy(plan.operation_desc);
    plan = LtPlan{};
}

static LtKey make_lt_key(int m, int n, int k, MatmulMode mode)
{
    LtKey key;
    // Row-major trick:
    // C_row(m,k)=A_row(m,n)*B_row(n,k) <=> C_col(k,m)=B_col(k,n)*A_col(n,m)
    key.m_col = k;
    key.n_col = m;
    key.k_col = n;
    key.lda = k;
    key.ldb = n;
    key.ldc = k;
    key.mode = static_cast<int>(mode);
    return key;
}

static bool build_lt_plan(int device, const LtKey &key, MatmulMode mode, LtPlan &plan)
{
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic_result = {};
    int returned_results = 0;
    cublasOperation_t op = CUBLAS_OP_N;
    size_t max_workspace = LT_MAX_WORKSPACE_BYTES;

    status = cublasLtMatmulDescCreate(&plan.operation_desc, lt_compute_type(mode), CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) goto fail;

    status = cublasLtMatmulDescSetAttribute(
        plan.operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)
    );
    if (status != CUBLAS_STATUS_SUCCESS) goto fail;

    status = cublasLtMatmulDescSetAttribute(
        plan.operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)
    );
    if (status != CUBLAS_STATUS_SUCCESS) goto fail;

    status = cublasLtMatrixLayoutCreate(&plan.a_desc, CUDA_R_32F, key.m_col, key.k_col, key.lda); // ptrB
    if (status != CUBLAS_STATUS_SUCCESS) goto fail;

    status = cublasLtMatrixLayoutCreate(&plan.b_desc, CUDA_R_32F, key.k_col, key.n_col, key.ldb); // ptrA
    if (status != CUBLAS_STATUS_SUCCESS) goto fail;

    status = cublasLtMatrixLayoutCreate(&plan.c_desc, CUDA_R_32F, key.m_col, key.n_col, key.ldc); // ptrC
    if (status != CUBLAS_STATUS_SUCCESS) goto fail;

    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) goto fail;

    status = cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace,
        sizeof(max_workspace)
    );
    if (status != CUBLAS_STATUS_SUCCESS) goto fail;

    status = cublasLtMatmulAlgoGetHeuristic(
        hcublaslt[device],
        plan.operation_desc,
        plan.a_desc,
        plan.b_desc,
        plan.c_desc,
        plan.c_desc,
        preference,
        1,
        &heuristic_result,
        &returned_results
    );
    if (status != CUBLAS_STATUS_SUCCESS || returned_results == 0) goto fail;

    plan.algo = heuristic_result.algo;
    plan.workspace_bytes = heuristic_result.workspaceSize;
    if (plan.workspace_bytes > LT_MAX_WORKSPACE_BYTES) {
        plan.workspace_bytes = LT_MAX_WORKSPACE_BYTES;
    }

    cublasLtMatmulPreferenceDestroy(preference);
    return true;

fail:
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    destroy_lt_plan(plan);
    return false;
}

static bool get_cached_lt_plan(int device, int m, int n, int k, MatmulMode mode, LtPlan &plan_out)
{
    if (device < 0 || device >= 64) return false;

    LtKey key = make_lt_key(m, n, k, mode);
    std::lock_guard<std::mutex> lock(lt_cache_mutex[device]);
    auto &entry = lt_plan_cache[device][key];

    if (!entry.initialized) {
        entry.initialized = true;
        entry.available = build_lt_plan(device, key, mode, entry.plan);
        if (get_matmul_config().debug) {
            fprintf(
                stderr,
                "[cuTensor][matmul] plan %s for m=%d n=%d k=%d mode=%s\n",
                entry.available ? "ready" : "unavailable",
                m, n, k, matmul_mode_name(mode)
            );
        }
    }
    if (!entry.available) return false;

    plan_out = entry.plan;
    return true;
}

static void* get_lt_workspace(int device, size_t requested_bytes, size_t &granted_bytes)
{
    granted_bytes = 0;
    if (device < 0 || device >= 64) return nullptr;
    if (requested_bytes == 0) return nullptr;

    size_t target_bytes = requested_bytes;
    if (target_bytes > LT_MAX_WORKSPACE_BYTES) target_bytes = LT_MAX_WORKSPACE_BYTES;

    if (lt_workspace_bytes[device] < target_bytes) {
        if (lt_workspace[device] != nullptr) {
            cudaFree(lt_workspace[device]);
            lt_workspace[device] = nullptr;
            lt_workspace_bytes[device] = 0;
        }
        cudaError_t err = cudaMalloc(&(lt_workspace[device]), target_bytes);
        if (err != cudaSuccess) {
            lt_workspace[device] = nullptr;
            lt_workspace_bytes[device] = 0;
            return nullptr;
        }
        lt_workspace_bytes[device] = target_bytes;
    }

    granted_bytes = lt_workspace_bytes[device];
    return lt_workspace[device];
}

static bool gpu_mult2D_lt(float *ptrA, float *ptrB, float *ptrC, int m, int n, int k, int device, MatmulMode mode)
{
    LtPlan plan;
    if (!get_cached_lt_plan(device, m, n, k, mode, plan)) return false;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t workspace_bytes = 0;
    void *workspace = get_lt_workspace(device, plan.workspace_bytes, workspace_bytes);
    if (plan.workspace_bytes > 0 && workspace == nullptr) return false;

    cublasStatus_t status = cublasLtMatmul(
        hcublaslt[device],
        plan.operation_desc,
        &alpha,
        ptrB,
        plan.a_desc,
        ptrA,
        plan.b_desc,
        &beta,
        ptrC,
        plan.c_desc,
        ptrC,
        plan.c_desc,
        &plan.algo,
        workspace,
        workspace_bytes,
        0
    );
    return status == CUBLAS_STATUS_SUCCESS;
}

// gpu mult2D C=A*B using cuBLAS taking into account that the matrices are stored in row-major order
void gpu_mult2D(float *ptrA, float *ptrB, float *ptrC, int m, int n, int k, int device) // m=A0,n=A1,k=B1
{
    if (device < 0 || device >= 64) {
        throw std::runtime_error("error setting device in gpu_mult2D");
    }
    check_cuda(cudaSetDevice(device), "gpu_mult2D_set_device");
    const MatmulConfig &cfg = get_matmul_config();
    ensure_cublas_math_mode(device, cfg.mode);

    // Preferred path for newer CUDA stacks/GPUs.
    // If unsupported on this device/runtime, fallback is classic SGEMM.
    if (!gpu_mult2D_lt(ptrA, ptrB, ptrC, m, n, k, device, cfg.mode)) {
        if (cfg.debug) {
            fprintf(
                stderr,
                "[cuTensor][matmul] fallback sgemm for m=%d n=%d k=%d mode=%s\n",
                m, n, k, matmul_mode_name(cfg.mode)
            );
        }
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
