#include <cuda.h>

#if CUDA_VERSION >= 10000 && INCLUDE_MMA_KERNELS

#include "singleStepLSTMKernel.h"
#include <cuda_fp16.hpp>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}
 
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

// Device functions
__forceinline__ __device__ float sigmoidf(float in)
{
    float expResult;
    float denom;
    float ans;
 
    asm volatile("{  ex2.approx.f32.ftz %0, %1;}\n" : "=f"(expResult) : "f"(-in * 1.4426950216293334961f));
    asm volatile("{  add.f32.ftz %0, %1, %2;}\n" : "=f"(denom) : "f"(1.f), "f"(expResult));
    asm volatile("{  rcp.approx.f32 %0, %1;}\n" : "=f"(ans) : "f"(denom));
 
    return ans;
}

template<int M_STEPS, int K_WARPS, int N_MAX, int K_MAX, typename T_ACC>
__launch_bounds__(32 * K_WARPS, 256 / (32 * K_WARPS))
__global__ void gemm_hmma_smallN_TN_a1b0_ker(const int m, const int n, const int k, const int lda, const int ldb, const int ldc, half *__restrict__ A, half *__restrict__ B, half *__restrict__ C)
{
// Requires superHMMA
#if (__CUDA_ARCH__ >= 750)    
    int baseM = blockIdx.x * 16 * M_STEPS;
   
    int localN = threadIdx.x / 4;
    
    uint4 b[K_MAX / (K_WARPS * 8)][N_MAX / 8];
    
#pragma unroll
    for (int step = 0; step < K_MAX / K_WARPS; step += 32)
    {
        int localK = step + (threadIdx.x % 4) * 8 + threadIdx.y * K_MAX / K_WARPS;
        
        #pragma unroll
            for (int innerN = 0; innerN < N_MAX; innerN += 8)
            {
                int actualN = localN + innerN;
                if (actualN >= n || localK >= k) break;
                b[step / 32][innerN / 8] = (actualN >= n || localK>= k) ? make_uint4(0, 0, 0, 0) : *(uint4*)(&B[(localN + innerN) * ldb + localK]);
            }
    }
    
    __shared__ half2 reduceSmem[M_STEPS][K_WARPS == 1 ? 1 : K_WARPS - 1][2 * (N_MAX / 8)][32];
   
    uint32_t c_u32[M_STEPS][2 * N_MAX / 8];
    half2 c_h2[M_STEPS][2 * N_MAX / 8];
        
#pragma unroll
    for (int mStep = 0; mStep < M_STEPS; mStep++)
    {
        #pragma unroll
            for (int innerN = 0; innerN < N_MAX; innerN += 8)
            {                
                c_u32[mStep][0 + 2 * (innerN / 8)] = 0;
                c_u32[mStep][1 + 2 * (innerN / 8)] = 0;
            }
    }
    
#pragma unroll
    for (int mStep = 0; mStep < M_STEPS; mStep++)
    {
        int localM1 = baseM + mStep * 16 + threadIdx.x / 4;
        int localM2 = baseM + mStep * 16 + threadIdx.x / 4 + 8;
        
        #pragma unroll
            for (int step = 0; step < K_MAX / K_WARPS; step += 32)
            {
                int localK = step + (threadIdx.x % 4) * 8 + threadIdx.y * K_MAX / K_WARPS;

                if (localK >= k) break;
                
                uint4 a[2];
                
                a[0] = step >= k / K_WARPS ? make_uint4(0, 0, 0, 0) : *(uint4*)(&A[localM1 * lda + localK]);
                a[1] = step >= k / K_WARPS ? make_uint4(0, 0, 0, 0) : *(uint4*)(&A[localM2 * lda + localK]);
                
                #pragma unroll
                    for (int innerN = 0; innerN < N_MAX; innerN += 8)
                    {
                        if (std::is_same<T_ACC, half>::value) {
                            asm volatile( \
                                "_mma.m16n8k8.row.col.f16.f16 \n" \
                                "    {%0, %1}, \n" \
                                "    {%2, %3}, \n" \
                                "    {%4}, \n" \
                                "    {%0, %1}; \n" \
                                        : "+r"(c_u32[mStep][0 + 2 * (innerN / 8)]), "+r"(c_u32[mStep][1 + 2 * (innerN / 8)])
                                        :  "r"(a[0].x),  "r"(a[1].x)
                                        ,  "r"(b[step / 32][innerN / 8].x)); 
                        }
                        else if (std::is_same<T_ACC, float>::value) {                    
                            asm volatile( \
                                "_mma.m16n8k8.row.col.f32.f32 \n" \
                                "    {%0, %1}, \n" \
                                "    {%2, %3}, \n" \
                                "    {%4}, \n" \
                                "    {%0, %1}; \n" \
                                        : "+r"(c_u32[mStep][0 + 2 * (innerN / 8)]), "+r"(c_u32[mStep][1 + 2 * (innerN / 8)])
                                        :  "r"(a[0].x),  "r"(a[1].x)
                                        ,  "r"(b[step / 32][innerN / 8].x)); 
                        }
                    }
                #pragma unroll
                    for (int innerN = 0; innerN < N_MAX; innerN += 8)
                    {
                        if (std::is_same<T_ACC, half>::value)
                        {
                            asm volatile( \
                                "_mma.m16n8k8.row.col.f16.f16 \n" \
                                "    {%0, %1}, \n" \
                                "    {%2, %3}, \n" \
                                "    {%4}, \n" \
                                "    {%0, %1}; \n" \
                                        : "+r"(c_u32[mStep][0 + 2 * (innerN / 8)]), "+r"(c_u32[mStep][1 + 2 * (innerN / 8)])
                                        :  "r"(a[0].y),  "r"(a[1].y)
                                        ,  "r"(b[step / 32][innerN / 8].y)); 
                        }
                        else if (std::is_same<T_ACC, float>::value)
                        {
                            asm volatile( \
                                "_mma.m16n8k8.row.col.f32.f32 \n" \
                                "    {%0, %1}, \n" \
                                "    {%2, %3}, \n" \
                                "    {%4}, \n" \
                                "    {%0, %1}; \n" \
                                        : "+r"(c_u32[mStep][0 + 2 * (innerN / 8)]), "+r"(c_u32[mStep][1 + 2 * (innerN / 8)])
                                        :  "r"(a[0].y),  "r"(a[1].y)
                                        ,  "r"(b[step / 32][innerN / 8].y)); 
                        }
                    }
                #pragma unroll
                    for (int innerN = 0; innerN < N_MAX; innerN += 8)
                    {
                        if (std::is_same<T_ACC, half>::value)
                        {
                            asm volatile( \
                                "_mma.m16n8k8.row.col.f16.f16 \n" \
                                "    {%0, %1}, \n" \
                                "    {%2, %3}, \n" \
                                "    {%4}, \n" \
                                "    {%0, %1}; \n" \
                                        : "+r"(c_u32[mStep][0 + 2 * (innerN / 8)]), "+r"(c_u32[mStep][1 + 2 * (innerN / 8)])
                                        :  "r"(a[0].z),  "r"(a[1].z)
                                        ,  "r"(b[step / 32][innerN / 8].z));
                        }                                
                        else if (std::is_same<T_ACC, float>::value)
                        {
                            asm volatile( \
                                "_mma.m16n8k8.row.col.f32.f32 \n" \
                                "    {%0, %1}, \n" \
                                "    {%2, %3}, \n" \
                                "    {%4}, \n" \
                                "    {%0, %1}; \n" \
                                        : "+r"(c_u32[mStep][0 + 2 * (innerN / 8)]), "+r"(c_u32[mStep][1 + 2 * (innerN / 8)])
                                        :  "r"(a[0].z),  "r"(a[1].z)
                                        ,  "r"(b[step / 32][innerN / 8].z));
                        }                                
                    }
                #pragma unroll
                    for (int innerN = 0; innerN < N_MAX; innerN += 8)
                    {
                        if (std::is_same<T_ACC, half>::value)
                        {
                            asm volatile( \
                                "_mma.m16n8k8.row.col.f16.f16 \n" \
                                "    {%0, %1}, \n" \
                                "    {%2, %3}, \n" \
                                "    {%4}, \n" \
                                "    {%0, %1}; \n" \
                                        : "+r"(c_u32[mStep][0 + 2 * (innerN / 8)]), "+r"(c_u32[mStep][1 + 2 * (innerN / 8)])
                                        :  "r"(a[0].w),  "r"(a[1].w)
                                        ,  "r"(b[step / 32][innerN / 8].w)); 
                        }
                        else if (std::is_same<T_ACC, float>::value)
                        {
                            asm volatile( \
                                "_mma.m16n8k8.row.col.f32.f32 \n" \
                                "    {%0, %1}, \n" \
                                "    {%2, %3}, \n" \
                                "    {%4}, \n" \
                                "    {%0, %1}; \n" \
                                        : "+r"(c_u32[mStep][0 + 2 * (innerN / 8)]), "+r"(c_u32[mStep][1 + 2 * (innerN / 8)])
                                        :  "r"(a[0].w),  "r"(a[1].w)
                                        ,  "r"(b[step / 32][innerN / 8].w)); 
                        }
                        
                    }
            }
        
        // This is a bit bad when K_WARPS grows to 4+.
        if (K_WARPS > 1)
        {
#pragma unroll
            for (int innerN = 0; innerN < N_MAX; innerN += 8)
            {
                if (threadIdx.y != 0)
                {               
                    reduceSmem[mStep][threadIdx.y - 1][0 + 2 * (innerN / 8)][threadIdx.x] = *(reinterpret_cast<half2*>(&(c_u32[mStep][0 + 2 * (innerN / 8)])));
                    reduceSmem[mStep][threadIdx.y - 1][1 + 2 * (innerN / 8)][threadIdx.x] = *(reinterpret_cast<half2*>(&(c_u32[mStep][1 + 2 * (innerN / 8)])));
                }
            }
            
            __syncthreads();
            if (threadIdx.y == 0)
            {
                #pragma unroll
                    for (int innerN = 0; innerN < N_MAX; innerN += 8)
                    {
                        c_h2[mStep][0 + 2 * (innerN / 8)] = *(reinterpret_cast<half2*>(&(c_u32[mStep][0 + 2 * (innerN / 8)])));
                        c_h2[mStep][1 + 2 * (innerN / 8)] = *(reinterpret_cast<half2*>(&(c_u32[mStep][1 + 2 * (innerN / 8)])));
                        
                        #pragma unroll
                            for (int y = 0; y < K_WARPS - 1; y++)
                            {
                                c_h2[mStep][0 + 2 * (innerN / 8)] += reduceSmem[mStep][y][0 + 2 * (innerN / 8)][threadIdx.x];
                                c_h2[mStep][1 + 2 * (innerN / 8)] += reduceSmem[mStep][y][1 + 2 * (innerN / 8)][threadIdx.x];                    
                            }
                    }
            }
        }
        else
        {
            #pragma unroll
                for (int innerN = 0; innerN < N_MAX; innerN += 8)
                {
                    c_h2[mStep][0 + 2 * (innerN / 8)] = *(reinterpret_cast<half2*>(&(c_u32[mStep][0 + 2 * (innerN / 8)])));
                    c_h2[mStep][1 + 2 * (innerN / 8)] = *(reinterpret_cast<half2*>(&(c_u32[mStep][1 + 2 * (innerN / 8)])));
                }           
        }
        
        if (K_WARPS == 1 || threadIdx.y == 0) {
#if (USE_INTERLEAVED_OUTPUT)
            // This gives us a weird output layout. 
            #pragma unroll
                for (int innerN = 0; innerN < N_MAX; innerN += 8)
                {   
                    int actualN = (threadIdx.x % 4) * 2 + innerN;

                    if (actualN < n)
                    {                   
                        *(reinterpret_cast<uint32_t*>(&C[localM1 * n + actualN])) = c_h2[mStep][0 + 2 * (innerN / 8)]; 
                        *(reinterpret_cast<uint32_t*>(&C[localM2 * n + actualN])) = c_h2[mStep][1 + 2 * (innerN / 8)]; 
                    }
                }
#else
            #pragma unroll
                for (int innerN = 0; innerN < N_MAX; innerN += 8)
                {       
                    int actualN = (threadIdx.x % 4) * 2 + innerN;
                    
                    if (actualN < n)
                    {
                        C[localM1 + (actualN) * ldc] = c_h2[mStep][0 + 2 * (innerN / 8)].x; 
                        C[localM2 + (actualN) * ldc] = c_h2[mStep][1 + 2 * (innerN / 8)].x;
                        if (actualN + 1 < n)
                        {                       
                            C[localM1 + (actualN + 1) * ldc] = c_h2[mStep][0 + 2 * (innerN / 8)].y; 
                            C[localM2 + (actualN + 1) * ldc] = c_h2[mStep][1 + 2 * (innerN / 8)].y;
                        }
                    }
                }
#endif
        }
    }
#endif     
}

template<int M_STEPS, int K_WARPS, int N_MAX, int K_MAX, typename T_ACC>
__launch_bounds__(32 * K_WARPS, 256 / (32 * K_WARPS))
__global__ void gemm_hmma_smallN_TN_a1b0_ker(const int m, const int n, const int k, const int lda, const int ldb, const int ldc, int8_t *__restrict__ A, int8_t *__restrict__ B, int32_t *__restrict__ C) {
// Requires superHMMA
#if (__CUDA_ARCH__ >= 750)    
    int baseM = blockIdx.x * 8 * M_STEPS;
   
    int localN = threadIdx.x / 4;
    
    uint4 b[K_MAX / (K_WARPS * 16)][N_MAX / 8];

#pragma unroll
    for (int step = 0; step < K_MAX / K_WARPS; step += 64) {
        int localK = step + (threadIdx.x % 4) * 16 + threadIdx.y * K_MAX / K_WARPS;
        
        #pragma unroll
            for (int innerN = 0; innerN < N_MAX; innerN += 8) {
                int actualN = localN + innerN;
                b[step / 64][innerN / 8] = (actualN >= n || step >= k / K_WARPS) ? make_uint4(0, 0, 0, 0) : *(uint4*)(&B[(localN + innerN) * ldb + localK]);
            }
    }
    
    __shared__ int32_t reduceSmem[M_STEPS][K_WARPS == 1 ? 1 : K_WARPS - 1][2 * (N_MAX / 8)][32];
   
    int32_t c[M_STEPS][2 * N_MAX / 8];
        
#pragma unroll
    for (int mStep = 0; mStep < M_STEPS; mStep++) {
        #pragma unroll
            for (int innerN = 0; innerN < N_MAX; innerN += 8) {                
                c[mStep][0 + 2 * (innerN / 8)] = 0;
                c[mStep][1 + 2 * (innerN / 8)] = 0;
            }
    }
    
#pragma unroll
    for (int mStep = 0; mStep < M_STEPS; mStep++) {
        int localM = baseM + mStep * 8 + threadIdx.x / 4;
        
        #pragma unroll
            for (int step = 0; step < K_MAX / K_WARPS; step += 64)
            {
                int localK = step + (threadIdx.x % 4) * 16 + threadIdx.y * K_MAX / K_WARPS;

                uint4 a;
                
                a = step >= k / K_WARPS ? make_uint4(0, 0, 0, 0) : *(uint4*)(&A[localM * lda + localK]);
                
                #pragma unroll
                    for (int innerN = 0; innerN < N_MAX; innerN += 8)
                    {
                        asm volatile( \
                            "_mma.m8n8k16.row.col.s8.s8 \n" \
                            "    {%0, %1}, \n" \
                            "    {%2}, \n" \
                            "    {%3}, \n" \
                            "    {%0, %1}; \n" \
                                    : "+r"(c[mStep][0 + 2 * (innerN / 8)]), "+r"(c[mStep][1 + 2 * (innerN / 8)])
                                    :  "r"(a.x)
                                    ,  "r"(b[step / 64][innerN / 8].x)); 
                    }
                #pragma unroll
                    for (int innerN = 0; innerN < N_MAX; innerN += 8)
                    {
                        asm volatile( \
                            "_mma.m8n8k16.row.col.s8.s8 \n" \
                            "    {%0, %1}, \n" \
                            "    {%2}, \n" \
                            "    {%3}, \n" \
                            "    {%0, %1}; \n" \
                                    : "+r"(c[mStep][0 + 2 * (innerN / 8)]), "+r"(c[mStep][1 + 2 * (innerN / 8)])
                                    :  "r"(a.y)
                                    ,  "r"(b[step / 64][innerN / 8].y)); 
                    }
                #pragma unroll
                    for (int innerN = 0; innerN < N_MAX; innerN += 8)
                    {
                        asm volatile( \
                            "_mma.m8n8k16.row.col.s8.s8 \n" \
                            "    {%0, %1}, \n" \
                            "    {%2}, \n" \
                            "    {%3}, \n" \
                            "    {%0, %1}; \n" \
                                    : "+r"(c[mStep][0 + 2 * (innerN / 8)]), "+r"(c[mStep][1 + 2 * (innerN / 8)])
                                    :  "r"(a.z)
                                    ,  "r"(b[step / 64][innerN / 8].z)); 
                    }
                #pragma unroll
                    for (int innerN = 0; innerN < N_MAX; innerN += 8)
                    {
                        asm volatile( \
                            "_mma.m8n8k16.row.col.s8.s8 \n" \
                            "    {%0, %1}, \n" \
                            "    {%2}, \n" \
                            "    {%3}, \n" \
                            "    {%0, %1}; \n" \
                                    : "+r"(c[mStep][0 + 2 * (innerN / 8)]), "+r"(c[mStep][1 + 2 * (innerN / 8)])
                                    :  "r"(a.w)
                                    ,  "r"(b[step / 64][innerN / 8].w)); 
                    }
            }
        
        // This is a bit bad when K_WARPS grows to 4+.
        if (K_WARPS > 1) {
#pragma unroll
            for (int innerN = 0; innerN < N_MAX; innerN += 8)
            {
                if (threadIdx.y != 0)
                {               
                    reduceSmem[mStep][threadIdx.y - 1][0 + 2 * (innerN / 8)][threadIdx.x] = c[mStep][0 + 2 * (innerN / 8)];
                    reduceSmem[mStep][threadIdx.y - 1][1 + 2 * (innerN / 8)][threadIdx.x] = c[mStep][1 + 2 * (innerN / 8)];
                }
            }
            
            __syncthreads();
            if (threadIdx.y == 0)
            {
#pragma unroll
                for (int innerN = 0; innerN < N_MAX; innerN += 8)
                {
                    #pragma unroll
                        for (int y = 0; y < K_WARPS - 1; y++)
                        {
                            c[mStep][0 + 2 * (innerN / 8)] += reduceSmem[mStep][y][0 + 2 * (innerN / 8)][threadIdx.x];
                            c[mStep][1 + 2 * (innerN / 8)] += reduceSmem[mStep][y][1 + 2 * (innerN / 8)][threadIdx.x];
                        }
                }
            }
        }
        
        if (K_WARPS == 1 || threadIdx.y == 0) {
#if (USE_INTERLEAVED_OUTPUT)
            // Errr
            assert(false);
#else
#pragma unroll
            for (int innerN = 0; innerN < N_MAX; innerN += 8)
            {       
                int actualN = (threadIdx.x % 4) * 2 + innerN;

                if (actualN < n)
                {
                    C[localM + (actualN)     * ldc] = c[mStep][0 + 2 * (innerN / 8)]; 
                    C[localM + (actualN + 1) * ldc] = c[mStep][1 + 2 * (innerN / 8)]; 
                }
            }
#endif
        }
    }
#endif     
}

template<int M_STEPS, int K_WARPS, int K_MAX, typename T_DATA_IN, typename T_DATA_OUT>
void gemm_hmma_smallN_TN_fixed_K(const int m, const int n, const int k, const int lda, const int ldb, const int ldc, T_DATA_IN *__restrict__ A, T_DATA_IN *__restrict__ B, T_DATA_OUT *__restrict__ C, cudaStream_t stream) {
    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = K_WARPS;
    
    int mPerMMA;
    
    if (std::is_same<T_DATA_IN, int8_t>::value)
    {
        mPerMMA = 8;
    }
    else
    {
        mPerMMA = 16;
    }
    
    gridDim.x = (m + mPerMMA * M_STEPS - 1) / (mPerMMA * M_STEPS);

    if (n <= 8)
    {
        gemm_hmma_smallN_TN_a1b0_ker<M_STEPS, K_WARPS, 8, K_MAX, T_DATA_OUT> <<< gridDim, blockDim , 0, stream >>> (m, n, k, lda, ldb, ldc, A, B, C);
    }
    else if (n <= 16)
    {
        gemm_hmma_smallN_TN_a1b0_ker<M_STEPS, K_WARPS, 16, K_MAX, T_DATA_OUT> <<< gridDim, blockDim , 0, stream >>> (m, n, k, lda, ldb, ldc, A, B, C);
    }
    else if (n <= 24)
    {
        gemm_hmma_smallN_TN_a1b0_ker<M_STEPS, K_WARPS, 24, K_MAX, T_DATA_OUT> <<< gridDim, blockDim , 0, stream >>> (m, n, k, lda, ldb, ldc, A, B, C);
    }
    else if (n <= 32)
    {
        gemm_hmma_smallN_TN_a1b0_ker<M_STEPS, K_WARPS, 32, K_MAX, T_DATA_OUT> <<< gridDim, blockDim , 0, stream >>> (m, n, k, lda, ldb, ldc, A, B, C);
    }
    else if (n <= 40)
    {
        gemm_hmma_smallN_TN_a1b0_ker<M_STEPS, K_WARPS, 40, K_MAX, T_DATA_OUT> <<< gridDim, blockDim , 0, stream >>> (m, n, k, lda, ldb, ldc, A, B, C);
    } 
    else
    {
        printf("n too larget for small N kernel %d\n", n);
    }
}


template<int M_STEPS, int K_WARPS, typename T_DATA_IN, typename T_DATA_OUT>
void gemm_hmma_smallN_TN(const int m, const int n, const int k, const int lda, const int ldb, const int ldc, T_DATA_IN *__restrict__ A, T_DATA_IN *__restrict__ B, T_DATA_OUT *__restrict__ C, cudaStream_t stream) {
    if (k <= 512)
    {
        gemm_hmma_smallN_TN_fixed_K<M_STEPS, K_WARPS, 512, T_DATA_IN, T_DATA_OUT>(m, n, k, lda, ldb, ldc, A, B, C, stream);
    }
    else  if (k <= 1024)
    {
        gemm_hmma_smallN_TN_fixed_K<M_STEPS, K_WARPS, 1024, T_DATA_IN, T_DATA_OUT>(m, n, k, lda, ldb, ldc, A, B, C, stream);
    }
    else if (k <= 2048)
    {
        gemm_hmma_smallN_TN_fixed_K<M_STEPS, K_WARPS, 2048, T_DATA_IN, T_DATA_OUT>(m, n, k, lda, ldb, ldc, A, B, C, stream);
    } 
    else {
        printf("k too larget for small N kernel %d\n", k);
    }
}
 
// Fused forward kernel
__global__ void elementWise_fp(int hiddenSize, 
                               int outputSize_i, 
                               int miniBatch,
                               int numSplitKStreams,
                               half *tmp_h, 
                               half *tmp_i, 
                               half *tmp_i_resid, 
                               half *bias,
                               half *h_out,
                               half *i_out,
                               half *concatData,
                               half *c_in,
                               half *c_out) {
    int numElements = miniBatch * hiddenSize;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= numElements) return;
    
    // TODO: fast divmod.
    int batch = index / hiddenSize;
    int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;    
    
    float g[4];
 
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        g[i] = tmp_h[i * hiddenSize + gateIndex];
 
        // TODO: Should probably template NUM_SPLIT_K_STREAMS rather than using the define
        #pragma unroll
            for (int j = 0; j < NUM_SPLIT_K_STREAMS; j++)
            {
                g[i] += (float)tmp_i[i * hiddenSize + gateIndex + j * numElements * 4]; 
            }
        
        g[i] += (float)bias[i * hiddenSize + index % hiddenSize];                
    }

    float in_gate      = sigmoidf(g[0]);
    float forget_gate  = sigmoidf(1 + g[1]);
    float in_gate2     = tanhf(g[2]);
    float out_gate     = sigmoidf(g[3]);
    
    float val = (forget_gate * (float)c_in[index]) + (in_gate * in_gate2);
    
    c_out[index] = val;
    
    val = out_gate * tanhf(val);
    h_out[index] = (half)val;
    
    if (tmp_i_resid)
    {
       val += (float)tmp_i_resid[index];
    }
 
#if (!CONCAT_IN_GEMM)    
    if (false && concatData)
    {       
        i_out[batch * outputSize_i + index % hiddenSize] = (half)val;
        // Memcpy. Not ideal but concatData probably won't have a perfect format.
        i_out[batch * outputSize_i + hiddenSize + index % hiddenSize] = (half)concatData[index];
 
    }   
    else 
#endif    
    {
        i_out[index] = (half)val;
    }
}
 
template<cudaDataType_t dataTypeIn, cudaDataType_t dataTypeOut, bool firstSmallGemm, bool secondSmallGemm>
void singleStepLSTMKernel(int hiddenSize, 
                                  int inputSize,
                                  int miniBatch, 
                                  int seqLength, 
                                  int numLayers,
                                  cublasHandle_t cublasHandle,
                                  half *x, 
                                  half **hx, 
                                  half **cx, 
                                  half **w, 
                                  half **bias,
                                  half *y, 
                                  half **hy, 
                                  half **cy,
                                  half *concatData,
                                  half *tmp_io,
                                  half *tmp_i,
                                  half *tmp_h,
#if (BATCHED_GEMM)
                                  half **aPtrs,
                                  half **bPtrs,
                                  half **cPtrs,
#endif
                                  cudaStream_t streami,
                                  cudaStream_t* splitKStreams,
                                  cudaEvent_t* splitKEvents,
                                  int numSplitKStreams,
                                  cudaStream_t streamh) {
    half alphaR = 1.f;
    half betaR  = 0.f;    
    
    half alphaL = 1.f;
    half betaL  = 0.f;
 
    int numElements = hiddenSize * miniBatch;
    
    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
 
    if (transb != CUBLAS_OP_N)
    {
        printf("Only transb == CUBLAS_OP_N supported\n");
        return;
    }
    
    if (seqLength > 1)
    {
        printf("Seq length > 1 not supported in this test code.\n");
        return;
    }
    
    // This is faster on V100, slower on T4???   
#if (BATCHED_GEMM)
    cublasErrCheck(cublasSetStream(cublasHandle, streamh));
    cublasErrCheck(cublasHgemmBatched(cublasHandle,
                    transa, transb,
                    4 * hiddenSize, miniBatch, inputSize,
                    &alphaR,
                    aPtrs, 
                    transa == CUBLAS_OP_N ? 4 * hiddenSize : inputSize,
                    bPtrs,
                    inputSize,
                    &betaR,
                    cPtrs, 
                    4 * hiddenSize,
                    numLayers)); 
#endif
 
    for (int layer = 0; layer < numLayers; layer++)
    {
        cudaEvent_t event;

        half *layer_i_in = layer == 0 ? x : tmp_io + numElements * layer;
        half *layer_i_out = layer == numLayers - 1 ? y : tmp_io + numElements * (layer + 1);

        for (int i = 0; i < numSplitKStreams; i++)
        {           
            cublasErrCheck(cublasSetStream(cublasHandle, splitKStreams[i]));
            cudaErrCheck(cudaEventCreate(&splitKEvents[i], cudaEventDisableTiming));
            
            half *inData;
            
#if (CONCAT_IN_GEMM)
            if (i < numSplitKStreams / 2)
            {
                inData = layer_i_in + 2 * i * hiddenSize / numSplitKStreams;
            }
            else
            {
                inData = concatData;
            }
#else
            inData = layer_i_in + i * inputSize / numSplitKStreams;
#endif
              
            {
#if (NEW_GEMM)
                if (firstSmallGemm && miniBatch < 32)
                {
                    gemm_hmma_smallN_TN<4, 4, half, half>
                        (4 * hiddenSize, miniBatch, inputSize / numSplitKStreams, 
                         inputSize, hiddenSize, 4 * hiddenSize,
                         w[layer] + i * inputSize / numSplitKStreams, inData, tmp_i + 4 * i * numElements,
                         splitKStreams[i]);
                }
                else
                {
#endif                                        
                    cublasErrCheck(cublasGemmEx(cublasHandle,
                                                transa, transb,
                                                4 * hiddenSize, miniBatch, inputSize / numSplitKStreams,
                                                &alphaL,
                                                w[layer] + i * inputSize / numSplitKStreams,
                                                dataTypeIn,
                                                transa == CUBLAS_OP_N ? 4 * hiddenSize : inputSize,
                                                inData,
                                                dataTypeIn,
                                                hiddenSize,
                                                &betaL,
                                                tmp_i + 4 * i * numElements,
                                                dataTypeOut,
                                                4 * hiddenSize,
                                                dataTypeOut,
                                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
#if (NEW_GEMM)            
            }
#endif
            cudaErrCheck(cudaEventRecord(splitKEvents[i], splitKStreams[i]));  
        }

        // Wait for layer GEMM streams
        for (int i = 0; i < numSplitKStreams; i++)
        {           
            cudaErrCheck(cudaStreamWaitEvent(streami, splitKEvents[i], 0));
            cudaErrCheck(cudaEventDestroy(splitKEvents[i]));  
        }
        
#if (!BATCHED_GEMM) 
        cublasErrCheck(cublasSetStream(cublasHandle, streamh));
        {
#if (NEW_GEMM)            
            // For now just grabbing the sizes we're interested in
            // Needs to pass in the correct SM version to remove the false.
            if (secondSmallGemm && miniBatch < 32)
            {
                gemm_hmma_smallN_TN<4, 4, half, half>
                    (4 * hiddenSize, miniBatch, hiddenSize, 
                     hiddenSize, hiddenSize, 4 * hiddenSize,
                     &w[layer][4 * hiddenSize * inputSize], hx[layer], tmp_h + 4 * layer * numElements,
                     streamh);
            }
            else
            {
#endif              
                cublasErrCheck(cublasGemmEx(cublasHandle,
                                            transa, transb,
                                            4 * hiddenSize, miniBatch, hiddenSize,
                                            &alphaR,
                                            &w[layer][4 * hiddenSize * inputSize], 
                                            dataTypeIn,
                                            transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                                            hx[layer],
                                            dataTypeIn,
                                            hiddenSize,
                                            &betaR,
                                            tmp_h + 4 * layer * numElements, 
                                            dataTypeOut,
                                            4 * hiddenSize,
                                            dataTypeOut,
                                            CUBLAS_GEMM_DEFAULT_TENSOR_OP)); 
#if (NEW_GEMM)            
            }
#endif    
        }
#endif
        cudaErrCheck(cudaEventCreate(&event, cudaEventDisableTiming));
        cudaErrCheck(cudaEventRecord(event, streamh));  
        
        dim3 blockDim;
        dim3 gridDim;
        
        blockDim.x = 256;
#if (USE_INTERLEAVED_OUTPUT)
        gridDim.x = (numElements / 8 + blockDim.x - 1) / blockDim.x;
#else
        gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
#endif

        // Wait for recurrent GEMM. Already waited for layer GEMM above.
        cudaErrCheck(cudaStreamWaitEvent(streami, event, 0));
        
        elementWise_fp <<< gridDim, blockDim , 0, streami >>> 
                 (hiddenSize, 
                  inputSize,
                  miniBatch,
                  numSplitKStreams,
                  tmp_h + 4 * layer * numElements, 
                  tmp_i,
                  layer > 0 ? layer_i_in : NULL,
                  bias[layer],
                  hy[layer],
                  layer_i_out,
                  (layer == numLayers - 1 ? NULL : concatData),
                  cx[layer],
                  cy[layer]);
        cudaErrCheck(cudaGetLastError());

        // Wait for elementwise ops before starting the next layer GEMM.
        cudaErrCheck(cudaEventRecord(event, streami));  
        for (int i = 0; i < numSplitKStreams; i++)
        {           
            cudaErrCheck(cudaStreamWaitEvent(splitKStreams[i], event, 0));
        }
        
        cudaErrCheck(cudaEventDestroy(event));  
    }
}

template void singleStepLSTMKernel<CUDA_R_16F, CUDA_R_16F, /* firstSmallGemm= */ false, /* secondSmallGemm= */ false>(int, int, int, int, int, cublasHandle_t, half*,  half**,  half**,
                                                                        half**,  half**, half*, half **, half **, half*, half*, half*, half*,
#if (BATCHED_GEMM)
                                                                        half**, half**, half**,
#endif
                                                                        cudaStream_t, cudaStream_t*, cudaEvent_t*, int, cudaStream_t);
template void singleStepLSTMKernel<CUDA_R_16F, CUDA_R_16F, /* firstSmallGemm= */ false, /* secondSmallGemm= */ true>(int, int, int, int, int, cublasHandle_t, half*,  half**,  half**,
                                                                        half**,  half**, half*, half **, half **, half*, half*, half*, half*,
#if (BATCHED_GEMM)
                                                                        half**, half**, half**,
#endif
                                                                        cudaStream_t, cudaStream_t*, cudaEvent_t*, int, cudaStream_t);
template void singleStepLSTMKernel<CUDA_R_16F, CUDA_R_16F, /* firstSmallGemm= */ true, /* secondSmallGemm= */ false>(int, int, int, int, int, cublasHandle_t, half*,  half**,  half**,
                                                                        half**,  half**, half*, half **, half **, half*, half*, half*, half*,
#if (BATCHED_GEMM)
                                                                        half**, half**, half**,
#endif
                                                                        cudaStream_t, cudaStream_t*, cudaEvent_t*, int, cudaStream_t);
template void singleStepLSTMKernel<CUDA_R_16F, CUDA_R_16F, /* firstSmallGemm= */ true, /* secondSmallGemm= */ true>(int, int, int, int, int, cublasHandle_t, half*,  half**,  half**,
                                                                        half**,  half**, half*, half **, half **, half*, half*, half*, half*,
#if (BATCHED_GEMM)
                                                                        half**, half**, half**,
#endif
                                                                        cudaStream_t, cudaStream_t*, cudaEvent_t*, int, cudaStream_t);
#endif /* CUDA_VERSION >= 10000 && INCLUDE_MMA_KERNELS */
