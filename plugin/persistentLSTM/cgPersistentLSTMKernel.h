const char* tCoreSource
    = "#include <cooperative_groups.h>                                                                                         \n\
using namespace cooperative_groups;                                                                                     \n\
#include <mma.h>                                                                                                        \n\
using namespace nvcuda;                                                                                                 \n\
__device__ __forceinline__ float sigmoidf(float in)                                                                     \n\
{                                                                                                                       \n\
    if (in > 0.f)                                                                                                       \n\
    {                                                                                                                   \n\
        return 1.f / (1.f + expf(-in));                                                                                 \n\
    }                                                                                                                   \n\
    else                                                                                                                \n\
    {                                                                                                                   \n\
        float z = expf(in);                                                                                             \n\
        return z / (1.f + z);                                                                                           \n\
    }                                                                                                                   \n\
}                                                                                                                       \n\
__device__ inline int roundUp(int x, int y)                                                                             \n\
{                                                                                                                       \n\
    return ((x - 1) / y + 1) * y;                                                                                       \n\
}                                                                                                                       \n\
template <int bidirectionFactor, int bidirectionK>                                                                      \n\
__device__ __forceinline__ void PLSTM_pointwise(int step, T_DATA* tmp_h, T_DATA* bias, T_DATA* y, T_DATA* tmp_i,        \n\
    T_DATA* smemc, T_DATA* final_h, int blockSplitKFactor, int hiddenSize, int minibatch, int hInOffset, int validBatch,\n\
    int* miniBatchArray)                                                                                                \n\
{                                                                                                                       \n\
    int tid = blockIdx.x * BLOCK_DIM + threadIdx.x;                                                                     \n\
    int idx;                                                                                                            \n\
    int c_idx;                                                                                                          \n\
    bool compileTimeCopyCheck = (bidirectionK == 1);                                                                    \n\
    for (idx = tid, c_idx = threadIdx.x; idx < validBatch * hiddenSize; idx += GRID_DIM * BLOCK_DIM, c_idx += BLOCK_DIM)\n\
    {                                                                                                                   \n\
        float g[4];                                                                                                     \n\
        int batch = idx / hiddenSize;                                                                                   \n\
        int gateIndex = (idx % hiddenSize) + 4 * batch * hiddenSize;                                                    \n\
        int biasOffset = bidirectionK * 2 * 4 * hiddenSize;                                                             \n\
        for (int i = 0; i < 4; i++)                                                                                     \n\
        {                                                                                                               \n\
            g[i] = (float) tmp_i[(hInOffset * 4 + 4 * batch * hiddenSize) * bidirectionFactor                           \n\
                       + bidirectionK * 4 * hiddenSize + i * hiddenSize + idx % hiddenSize]                             \n\
                + (float) bias[biasOffset + i * hiddenSize + idx % hiddenSize]                                          \n\
                + (float) bias[biasOffset + 4 * hiddenSize + i * hiddenSize + idx % hiddenSize];                        \n\
            for (int k = 0; k < blockSplitKFactor; k++)                                                                 \n\
            {                                                                                                           \n\
                g[i] += (float) tmp_h[k * hiddenSize * 4 * roundUp(minibatch, 8) + i * hiddenSize + gateIndex           \n\
                    + bidirectionK * hiddenSize * 4 * blockSplitKFactor * roundUp(minibatch, 8)];                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        float in_gate = sigmoidf(g[0]);                                                                                 \n\
        float forget_gate = sigmoidf(g[2]);                                                                             \n\
        float in_gate2 = tanhf(g[1]);                                                                                   \n\
        float out_gate = sigmoidf(g[3]);                                                                                \n\
        int cPerThread = (hiddenSize * minibatch + BLOCK_DIM * GRID_DIM - 1) / (BLOCK_DIM * GRID_DIM);                  \n\
        float res = in_gate * in_gate2;                                                                                 \n\
        res += forget_gate * (float) smemc[bidirectionK * cPerThread * BLOCK_DIM + c_idx];                              \n\
        smemc[bidirectionK * cPerThread * BLOCK_DIM + c_idx] = res;                                                     \n\
        res = out_gate * tanhf(res);                                                                                    \n\
        size_t yIdx = hInOffset * bidirectionFactor + batch * bidirectionFactor * hiddenSize + bidirectionK * hiddenSize\n\
            + idx % hiddenSize;                                                                                         \n\
        y[yIdx] = res;                                                                                                  \n\
        if (final_h != NULL)                                                                                            \n\
        {                                                                                                               \n\
            final_h[bidirectionK * miniBatchArray[0] * hiddenSize + idx] = res;                                         \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
}                                                                                                                       \n\
extern __shared__ char dsmem[];                                                                                         \n\
template <int kFragsPerWarp, int mFragsPerWarp, int kFragsPerWarpInRF, int kFragsPerWarpInSM, int WARPS_PER_BLOCK_K,    \n\
    int WARPS_PER_BLOCK_M, int bidirectionFactor>                                                                       \n\
__device__ __forceinline__ void PLSTM_load_rMat_tcores(const T_DATA* rMat,                                              \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> rMat_frags[bidirectionFactor]         \n\
                                                                                            [kFragsPerWarpInRF]         \n\
                                                                                            [mFragsPerWarp],            \n\
    const int blockSplitKFactor, const int hiddenSize, char* smem)                                                      \n\
{                                                                                                                       \n\
    half* tmpSmem = (half*) smem;                                                                                       \n\
    int kBlock = blockIdx.x % blockSplitKFactor;                                                                        \n\
    int laneIdx = (threadIdx.x % 32);                                                                                   \n\
    int localWarpIdx = threadIdx.x / 32;                                                                                \n\
    int kBlockWarpIdx = (blockIdx.x / blockSplitKFactor) * (BLOCK_DIM / 32) + localWarpIdx;                             \n\
#pragma unroll                                                                                                          \n\
    for (int i = 0; i < kFragsPerWarp; i++)                                                                             \n\
    {                                                                                                                   \n\
#pragma unroll                                                                                                          \n\
        for (int k = 0; k < bidirectionFactor; k++)                                                                     \n\
        {                                                                                                               \n\
#pragma unroll                                                                                                          \n\
            for (int j = 0; j < mFragsPerWarp; j++)                                                                     \n\
            {                                                                                                           \n\
                // For WARPS_PER_BLOCK_K == 2 => warps 0 and 1 co-operate on rows.                                      \n\
                // For WARPS_PER_BLOCK_K == 4 => warps 0, 1, 2 and 3 co-operate on rows.                                \n\
                int fragFirstCol                                                                                        \n\
                    = ((kBlock * WARPS_PER_BLOCK_K + localWarpIdx % WARPS_PER_BLOCK_K) * kFragsPerWarp + i) * FRAG_K;   \n\
                int fragFirstRow = ((kBlockWarpIdx / WARPS_PER_BLOCK_K) * mFragsPerWarp + j) * FRAG_M;                  \n\
                if (i < kFragsPerWarpInRF)                                                                              \n\
                {                                                                                                       \n\
                    // Load the FRAG_M*FRAG_K values                                                                    \n\
                    for (int startIdx = 0; startIdx < FRAG_M * FRAG_K; startIdx += 32)                                  \n\
                    {                                                                                                   \n\
                        int smemCol = (startIdx + laneIdx) % FRAG_K;                                                    \n\
                        int smemRow = (startIdx + laneIdx) / FRAG_K;                                                    \n\
                        int col = fragFirstCol + smemCol;                                                               \n\
                        int row = fragFirstRow + smemRow;                                                               \n\
                        if (col < hiddenSize && row < 4 * hiddenSize)                                                   \n\
                        {                                                                                               \n\
                            tmpSmem[localWarpIdx * FRAG_M * FRAG_K + smemRow * FRAG_K + smemCol]                        \n\
                                = (half)(rMat[k * 4 * hiddenSize * hiddenSize + row * hiddenSize + col]);               \n\
                        }                                                                                               \n\
                        else                                                                                            \n\
                        {                                                                                               \n\
                            tmpSmem[localWarpIdx * FRAG_M * FRAG_K + smemRow * FRAG_K + smemCol] = (half)(0.f);         \n\
                        }                                                                                               \n\
                    }                                                                                                   \n\
                    __syncthreads();                                                                                    \n\
                    wmma::load_matrix_sync(rMat_frags[k][i][j], &(tmpSmem[localWarpIdx * FRAG_M * FRAG_K]), FRAG_K);    \n\
                }                                                                                                       \n\
                else                                                                                                    \n\
                {                                                                                                       \n\
                    int localkFragsIdx                                                                                  \n\
                        = k * (kFragsPerWarpInSM * mFragsPerWarp) + (i - kFragsPerWarpInRF) * mFragsPerWarp + j;        \n\
                    const int smemPerK = FRAG_M * FRAG_K * (BLOCK_DIM / 32);                                            \n\
                    int fragFirstCol                                                                                    \n\
                        = ((kBlock * WARPS_PER_BLOCK_K + localWarpIdx % WARPS_PER_BLOCK_K) * kFragsPerWarp + i)         \n\
                        * FRAG_K;                                                                                       \n\
                    int fragFirstRow = ((kBlockWarpIdx / WARPS_PER_BLOCK_K) * mFragsPerWarp + j) * FRAG_M;              \n\
                    // Load the FRAG_M*FRAG_K values                                                                    \n\
                    for (int startIdx = 0; startIdx < FRAG_M * FRAG_K; startIdx += 32)                                  \n\
                    {                                                                                                   \n\
                        int smemCol = (startIdx + laneIdx) % FRAG_K;                                                    \n\
                        int smemRow = (startIdx + laneIdx) / FRAG_K;                                                    \n\
                        int col = fragFirstCol + smemCol;                                                               \n\
                        int row = fragFirstRow + smemRow;                                                               \n\
                        if (col < hiddenSize && row < 4 * hiddenSize)                                                   \n\
                        {                                                                                               \n\
                            tmpSmem[localkFragsIdx * smemPerK + localWarpIdx * FRAG_M * FRAG_K + smemRow * FRAG_K       \n\
                                + smemCol]                                                                              \n\
                                = (half)(rMat[k * 4 * hiddenSize * hiddenSize + row * hiddenSize + col]);               \n\
                        }                                                                                               \n\
                        else                                                                                            \n\
                        {                                                                                               \n\
                            tmpSmem[localkFragsIdx * smemPerK + localWarpIdx * FRAG_M * FRAG_K + smemRow * FRAG_K       \n\
                                + smemCol]                                                                              \n\
                                = (half)(0.f);                                                                          \n\
                        }                                                                                               \n\
                    }                                                                                                   \n\
                    __syncthreads();                                                                                    \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        __syncthreads();                                                                                                \n\
    }                                                                                                                   \n\
}                                                                                                                       \n\
template <int kFragsPerWarp, int mFragsPerWarp, int kFragsPerWarpInRF, int kFragsPerWarpInSM, int WARPS_PER_BLOCK_K,    \n\
    int WARPS_PER_BLOCK_M, int bidirectionFactor, int bidirectionK>                                                     \n\
__device__ __forceinline__ void PLSTM_load_rMat_tcores_sm(                                                              \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major>& rMat_frags_sm, char* smem,           \n\
    const int mFragIdx, const int kFragIdx)                                                                             \n\
{                                                                                                                       \n\
    half* tmpSmem = (half*) smem;                                                                                       \n\
    int localWarpIdx = threadIdx.x / 32;                                                                                \n\
    int localkFragsIdx = bidirectionK * (kFragsPerWarpInSM * mFragsPerWarp)                                             \n\
        + (kFragIdx - kFragsPerWarpInRF) * mFragsPerWarp + mFragIdx;                                                    \n\
    const int smemPerK = FRAG_M * FRAG_K * (BLOCK_DIM / 32);                                                            \n\
    wmma::load_matrix_sync(                                                                                             \n\
        rMat_frags_sm, &(tmpSmem[localkFragsIdx * smemPerK + localWarpIdx * FRAG_M * FRAG_K]), FRAG_K);                 \n\
}                                                                                                                       \n\
template <int kFragsPerWarp, int mFragsPerWarp, int kFragsPerWarpInRF, int kFragsPerWarpInSM, int WARPS_PER_BLOCK_K,    \n\
    int WARPS_PER_BLOCK_M, int innerStepSize, int unrollSplitK, int unrollGemmBatch, int hiddenSize, int ldh,           \n\
    int bidirectionFactor, int bidirectionK>                                                                            \n\
__device__ __forceinline__ void PLSTM_GEMM_tcores_inner(                                                                \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> rMat_frags[bidirectionFactor]         \n\
                                                                                            [kFragsPerWarpInRF]         \n\
                                                                                            [mFragsPerWarp],            \n\
    T_DATA* __restrict__ h_out, const int blockSplitKFactor, const int minibatch, const int batchStart, half* smemh,    \n\
    T_ACCUMULATE* smemAccumulate, char* smemr)                                                                          \n\
{                                                                                                                       \n\
    int kBlock = blockIdx.x % blockSplitKFactor;                                                                        \n\
    int localWarpIdx = threadIdx.x / 32;                                                                                \n\
    int kBlockWarpIdx = (blockIdx.x / blockSplitKFactor) * (BLOCK_DIM / 32) + localWarpIdx;                             \n\
    const int numInnerSteps = (innerStepSize + FRAG_N - 1) / FRAG_N;                                                    \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> rMat_frags_sm;                        \n\
    wmma::fragment<wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, half, wmma::col_major> b_frag[numInnerSteps];                \n\
    wmma::fragment<wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, T_ACCUMULATE> c_frag[mFragsPerWarp][numInnerSteps];       \n\
#pragma unroll                                                                                                          \n\
    for (int innerStep = 0; innerStep < numInnerSteps; innerStep++)                                                     \n\
    {                                                                                                                   \n\
#pragma unroll                                                                                                          \n\
        for (int i = 0; i < mFragsPerWarp; i++)                                                                         \n\
        {                                                                                                               \n\
            wmma::fill_fragment(c_frag[i][innerStep], (T_ACCUMULATE) 0.0f);                                             \n\
        }                                                                                                               \n\
#pragma unroll                                                                                                          \n\
        for (int i = 0; i < kFragsPerWarp; i++)                                                                         \n\
        {                                                                                                               \n\
            int Acol = innerStep * ldh * FRAG_N + ((localWarpIdx % WARPS_PER_BLOCK_K) * kFragsPerWarp + i) * FRAG_K;    \n\
            wmma::load_matrix_sync(b_frag[innerStep], smemh + Acol, ldh);                                               \n\
            if (i < kFragsPerWarpInRF)                                                                                  \n\
            {                                                                                                           \n\
#pragma unroll                                                                                                          \n\
                for (int j = 0; j < mFragsPerWarp; j++)                                                                 \n\
                {                                                                                                       \n\
                    wmma::mma_sync(                                                                                     \n\
                        c_frag[j][innerStep], rMat_frags[bidirectionK][i][j], b_frag[innerStep], c_frag[j][innerStep]); \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
            else                                                                                                        \n\
            {                                                                                                           \n\
#pragma unroll                                                                                                          \n\
                for (int j = 0; j < mFragsPerWarp; j++)                                                                 \n\
                {                                                                                                       \n\
                    PLSTM_load_rMat_tcores_sm<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM,       \n\
                        WARPS_PER_BLOCK_K, WARPS_PER_BLOCK_M, bidirectionFactor, bidirectionK>(                         \n\
                        rMat_frags_sm, smemr, j, i);                                                                    \n\
                    wmma::mma_sync(c_frag[j][innerStep], rMat_frags_sm, b_frag[innerStep], c_frag[j][innerStep]);       \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
    // We're done with smemh                                                                                            \n\
    __syncthreads();                                                                                                    \n\
    // This is true if we have exactly the right number of M elements. Otherwise we have to do a runtime check.         \n\
    bool compileTimeBoundsCheck                                                                                         \n\
        = ((GRID_DIM / blockSplitKFactor) * WARPS_PER_BLOCK_M * mFragsPerWarp) * FRAG_M <= 4 * hiddenSize;              \n\
    // Write out to shared memory so we can reduce across warps and/or do a type conversion.                            \n\
    if (WARPS_PER_BLOCK_K > 1 || sizeof(T_DATA) != sizeof(T_ACCUMULATE))                                                \n\
    {                                                                                                                   \n\
// Tiling is non-trivial. Dimensions:                                                                                   \n\
//   1) The fragments in a warp. We've accumulated the k-fragments, so this spans M.                                    \n\
//   2) The warps. This is k-major, so if WARPS_PER_BLOCK_K == 2, warps 0 and 1 hold values to contribute to the same   \n\
//      output M.                                                                                                       \n\
//   3) The steps across N.                                                                                             \n\
#pragma unroll                                                                                                          \n\
        for (int innerStep = 0; innerStep < numInnerSteps; innerStep++)                                                 \n\
        {                                                                                                               \n\
#pragma unroll                                                                                                          \n\
            for (int i = 0; i < mFragsPerWarp; i++)                                                                     \n\
            {                                                                                                           \n\
                wmma::store_matrix_sync(smemAccumulate                                                                  \n\
                        + (((BLOCK_DIM / 32) * innerStep + localWarpIdx) * mFragsPerWarp + i) * FRAG_M * FRAG_N,        \n\
                    c_frag[i][innerStep], FRAG_M, wmma::mem_col_major);                                                 \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        __syncthreads();                                                                                                \n\
#pragma unroll                                                                                                          \n\
        for (int innerStep = 0; innerStep < numInnerSteps; innerStep++)                                                 \n\
        {                                                                                                               \n\
            size_t hOutOffset;                                                                                          \n\
            if (blockSplitKFactor > 1)                                                                                  \n\
            {                                                                                                           \n\
                hOutOffset = (kBlock * roundUp(minibatch, 8) + (batchStart + innerStep * FRAG_N)) * hiddenSize * 4;     \n\
                hOutOffset += bidirectionK * blockSplitKFactor * hiddenSize * 4 * roundUp(minibatch, 8);                \n\
            }                                                                                                           \n\
            else                                                                                                        \n\
            {                                                                                                           \n\
                hOutOffset = (batchStart + innerStep * FRAG_N) * hiddenSize * 4;                                        \n\
                hOutOffset += bidirectionK * hiddenSize * 4 * roundUp(minibatch, 8);                                    \n\
            }                                                                                                           \n\
            // Sometimes it's advantageous to unroll this loop, sometimes not.                                          \n\
            // We can make it a tuneable parameter and autotune for our situation.                                      \n\
            const int unrollFactor                                                                                      \n\
                = unrollSplitK ? (WARPS_PER_BLOCK_M * mFragsPerWarp * FRAG_N * FRAG_M + BLOCK_DIM - 1) / BLOCK_DIM : 1; \n\
#pragma unroll unrollFactor                                                                                             \n\
            for (int startIdx = 0; startIdx < WARPS_PER_BLOCK_M * mFragsPerWarp * FRAG_N * FRAG_M;                      \n\
                 startIdx += BLOCK_DIM)                                                                                 \n\
            {                                                                                                           \n\
                int idx = startIdx + threadIdx.x;                                                                       \n\
                // If the block is larger than the total work we need to check if our index is valid.                   \n\
                // This is a compile time check followed by a runtime check so is optimized out if                      \n\
                // we don't have to do the runtime check.                                                               \n\
                if ((WARPS_PER_BLOCK_M * mFragsPerWarp * FRAG_N * FRAG_M) % BLOCK_DIM != 0                              \n\
                    && idx >= WARPS_PER_BLOCK_M * mFragsPerWarp * FRAG_N * FRAG_M)                                      \n\
                {                                                                                                       \n\
                    break;                                                                                              \n\
                }                                                                                                       \n\
                // If FRAG_N * FRAG_M == BLOCK_DIM then a lot of these values become independent of                     \n\
                // threadIdx, and we remove quite a few integer operations.                                             \n\
                int matrixElement;                                                                                      \n\
                int mTile;                                                                                              \n\
                int mFragId; // 0->mFragsPerWarp                                                                        \n\
                int mWarpId; // 0->WARPS_PER_BLOCK_M                                                                    \n\
                if (FRAG_N * FRAG_M == BLOCK_DIM)                                                                       \n\
                {                                                                                                       \n\
                    matrixElement = threadIdx.x;                                                                        \n\
                    mTile = startIdx / BLOCK_DIM;                                                                       \n\
                    mFragId = mTile % mFragsPerWarp;                                                                    \n\
                    mWarpId = mTile / mFragsPerWarp;                                                                    \n\
                }                                                                                                       \n\
                else                                                                                                    \n\
                {                                                                                                       \n\
                    matrixElement = idx % (FRAG_M * FRAG_N);                                                            \n\
                    mTile = idx / (FRAG_N * FRAG_M);                                                                    \n\
                    mFragId = mTile % mFragsPerWarp;                                                                    \n\
                    mWarpId = mTile / mFragsPerWarp;                                                                    \n\
                }                                                                                                       \n\
                int innerStepOffset = innerStep * (BLOCK_DIM / 32) * mFragsPerWarp * FRAG_M * FRAG_N;                   \n\
                T_DATA res = 0;                                                                                         \n\
#pragma unroll                                                                                                          \n\
                for (int warpK = 0; warpK < WARPS_PER_BLOCK_K; warpK++)                                                 \n\
                {                                                                                                       \n\
                    res += smemAccumulate[innerStepOffset                                                               \n\
                        + (mWarpId * WARPS_PER_BLOCK_K * mFragsPerWarp + mFragId + warpK * mFragsPerWarp) * FRAG_N      \n\
                            * FRAG_M                                                                                    \n\
                        + matrixElement];                                                                               \n\
                }                                                                                                       \n\
                int col = matrixElement / FRAG_M;                                                                       \n\
                int row = ((blockIdx.x / blockSplitKFactor) * WARPS_PER_BLOCK_M * mFragsPerWarp + mTile) * FRAG_M       \n\
                    + matrixElement % FRAG_M;                                                                           \n\
                if (compileTimeBoundsCheck || row < 4 * hiddenSize)                                                     \n\
                {                                                                                                       \n\
                    h_out[hOutOffset + row + col * 4 * hiddenSize] = res;                                               \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        __syncthreads();                                                                                                \n\
    }                                                                                                                   \n\
    else                                                                                                                \n\
    {                                                                                                                   \n\
#pragma unroll                                                                                                          \n\
        for (int innerStep = 0; innerStep < numInnerSteps; innerStep++)                                                 \n\
        {                                                                                                               \n\
            size_t hOutOffset;                                                                                          \n\
            if (blockSplitKFactor > 1)                                                                                  \n\
            {                                                                                                           \n\
                hOutOffset = (kBlock * roundUp(minibatch, 8) + (batchStart + innerStep * FRAG_N)) * hiddenSize * 4;     \n\
                hOutOffset += bidirectionK * blockSplitKFactor * hiddenSize * 4 * roundUp(minibatch, 8);                \n\
            }                                                                                                           \n\
            else                                                                                                        \n\
            {                                                                                                           \n\
                hOutOffset = (batchStart + innerStep * FRAG_N) * hiddenSize * 4;                                        \n\
                hOutOffset += bidirectionK * hiddenSize * 4 * roundUp(minibatch, 8);                                    \n\
            }                                                                                                           \n\
#pragma unroll                                                                                                          \n\
            for (int i = 0; i < mFragsPerWarp; i++)                                                                     \n\
            {                                                                                                           \n\
                if ((compileTimeBoundsCheck || kBlockWarpIdx * mFragsPerWarp + i) * FRAG_M < 4 * hiddenSize)            \n\
                {                                                                                                       \n\
                    wmma::store_matrix_sync(                                                                            \n\
                        (T_ACCUMULATE*) h_out + hOutOffset + (kBlockWarpIdx * mFragsPerWarp + i) * FRAG_M,              \n\
                        c_frag[i][innerStep], 4 * hiddenSize, wmma::mem_col_major);                                     \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
}                                                                                                                       \n\
template <int kFragsPerWarp, int mFragsPerWarp, int kFragsPerWarpInRF, int kFragsPerWarpInSM, int WARPS_PER_BLOCK_K,    \n\
    int WARPS_PER_BLOCK_M, int innerStepSize, int unrollSplitK, int unrollGemmBatch, int minibatch, int hiddenSize,     \n\
    int bidirectionFactor, int bidirectionK>                                                                            \n\
__device__ __forceinline__ void PLSTM_GEMM_tcores_piped(                                                                \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> rMat_frags[bidirectionFactor]         \n\
                                                                                            [kFragsPerWarpInRF]         \n\
                                                                                            [mFragsPerWarp],            \n\
    T_DATA* __restrict__ init_h, T_DATA* __restrict__ h_in, T_DATA* __restrict__ h_out, const int step,                 \n\
    const int seqLength, const int blockSplitKFactor, const int* miniBatchArray, const int hInOffset, char* smem,       \n\
    char* smemr)                                                                                                        \n\
{                                                                                                                       \n\
    half* smemh = (half*) smem;                                                                                         \n\
    T_ACCUMULATE* smemAccumulate = (T_ACCUMULATE*) (smemh);                                                             \n\
    int kBlock = blockIdx.x % blockSplitKFactor;                                                                        \n\
    T_DATA registerBuffer[(kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K + BLOCK_DIM - 1) / BLOCK_DIM][innerStepSize];     \n\
    // It's useful to be able to determine at compile time whether conditionals within the data loading loops           \n\
    // need to be executed. This saves registers and allows the compiler to do extra optimizations.                     \n\
    bool compileTimeGmemBoundsCheck1 = (kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K) % BLOCK_DIM == 0                    \n\
        && kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K == hiddenSize / blockSplitKFactor;                                \n\
    bool compileTimeGmemBoundsCheck2 = minibatch % innerStepSize == 0;                                                  \n\
    bool compileTimeSmemBoundsCheck = (kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K) % BLOCK_DIM == 0;                    \n\
    // To avoid bank conflicts inside wmma::load_matrix_sync we want to ensure that consecutive columns                 \n\
    // of the loaded submatrix start at different shared memory banks. We would also like to use vectorized load        \n\
    // instructions, which are 128 bit wide.                                                                            \n\
    // V100 has 32 banks of 32 bit words (See Programming Guide Appendix H). To avoid bank conflicts                    \n\
    // we should therefore offset eah column by 128 bits, or 8 fp16 values.                                             \n\
    const int ldh                                                                                                       \n\
        = kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K + ((kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K) % 64 == 0 ? 8 : 0);\n\
    int batchStart = 0;                                                                                                 \n\
// Load the B matrix from global to shared. Whole block co-operates.                                                    \n\
// This loop looks more complicated than it is due to the nested bounds checking.                                       \n\
// If the if statements are true it becomes a lot simpler.                                                              \n\
#pragma unroll                                                                                                          \n\
    for (int i = 0; i < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K; i += BLOCK_DIM)                                     \n\
    {                                                                                                                   \n\
        int idx = i + threadIdx.x;                                                                                      \n\
        int row = idx + (blockSplitKFactor > 1 ? kBlock * kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K : 0);              \n\
        if (compileTimeGmemBoundsCheck1 || (idx < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K && row < hiddenSize))      \n\
        {                                                                                                               \n\
#pragma unroll                                                                                                          \n\
            for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                             \n\
            {                                                                                                           \n\
                int subExample = batchStart + innerStep;                                                                \n\
                if (compileTimeGmemBoundsCheck2 || (subExample < minibatch))                                            \n\
                {                                                                                                       \n\
                    registerBuffer[i / BLOCK_DIM][innerStep]                                                            \n\
                        = init_h[bidirectionK * miniBatchArray[0] * hiddenSize + subExample * hiddenSize + row];        \n\
                }                                                                                                       \n\
                else                                                                                                    \n\
                {                                                                                                       \n\
                    registerBuffer[i / BLOCK_DIM][innerStep] = 0.f;                                                     \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        else                                                                                                            \n\
        {                                                                                                               \n\
#pragma unroll                                                                                                          \n\
            for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                             \n\
            {                                                                                                           \n\
                registerBuffer[i / BLOCK_DIM][innerStep] = 0.f;                                                         \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
#pragma unroll                                                                                                          \n\
    for (int i = 0; i < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K; i += BLOCK_DIM)                                     \n\
    {                                                                                                                   \n\
        int idx = i + threadIdx.x;                                                                                      \n\
        if (compileTimeSmemBoundsCheck || idx < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K)                             \n\
        {                                                                                                               \n\
            for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                             \n\
            {                                                                                                           \n\
                smemh[innerStep * ldh + idx] = registerBuffer[i / BLOCK_DIM][innerStep];                                \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
    __syncthreads();                                                                                                    \n\
    // Sometimes it's advantageous to unroll this loop, sometimes not.                                                  \n\
    // We can make it a tuneable parameter and autotune for our situation.                                              \n\
    // Now that minibatch size is not constant, not sure what to put here                                               \n\
    const int unrollFactor = unrollGemmBatch ? (minibatch - 1) / innerStepSize : 1;                                     \n\
#pragma unroll unrollFactor                                                                                             \n\
    for (; batchStart < minibatch - innerStepSize; batchStart += innerStepSize)                                         \n\
    {                                                                                                                   \n\
// Same as above, but load the next one in preparation for the next loop iteration.                                     \n\
#pragma unroll                                                                                                          \n\
        for (int i = 0; i < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K; i += BLOCK_DIM)                                 \n\
        {                                                                                                               \n\
            int idx = i + threadIdx.x;                                                                                  \n\
            int row = idx + (blockSplitKFactor > 1 ? kBlock * kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K : 0);          \n\
            if (compileTimeGmemBoundsCheck1 || (idx < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K && row < hiddenSize))  \n\
            {                                                                                                           \n\
#pragma unroll                                                                                                          \n\
                for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                         \n\
                {                                                                                                       \n\
                    int subExample = batchStart + innerStepSize + innerStep;                                            \n\
                    if (compileTimeGmemBoundsCheck2 || (subExample < minibatch))                                        \n\
                    {                                                                                                   \n\
                        registerBuffer[i / BLOCK_DIM][innerStep]                                                        \n\
                            = init_h[bidirectionK * miniBatchArray[0] * hiddenSize + subExample * hiddenSize + row];    \n\
                    }                                                                                                   \n\
                    else                                                                                                \n\
                    {                                                                                                   \n\
                        registerBuffer[i / BLOCK_DIM][innerStep] = 0.f;                                                 \n\
                    }                                                                                                   \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
            else                                                                                                        \n\
            {                                                                                                           \n\
#pragma unroll                                                                                                          \n\
                for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                         \n\
                {                                                                                                       \n\
                    registerBuffer[i / BLOCK_DIM][innerStep] = 0.f;                                                     \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        PLSTM_GEMM_tcores_inner<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM, WARPS_PER_BLOCK_K,  \n\
            WARPS_PER_BLOCK_M, innerStepSize, unrollSplitK, unrollGemmBatch, hiddenSize, ldh, bidirectionFactor,        \n\
            bidirectionK>(rMat_frags, h_out, blockSplitKFactor, minibatch, batchStart, smemh, smemAccumulate, smemr);   \n\
// Because the above call has a __syncthreads after reading from smemh we don't                                         \n\
// need one before writing.                                                                                             \n\
#pragma unroll                                                                                                          \n\
        for (int i = 0; i < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K; i += BLOCK_DIM)                                 \n\
        {                                                                                                               \n\
            int idx = i + threadIdx.x;                                                                                  \n\
            if (compileTimeSmemBoundsCheck || idx < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K)                         \n\
            {                                                                                                           \n\
                for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                         \n\
                {                                                                                                       \n\
                    smemh[innerStep * ldh + idx] = registerBuffer[i / BLOCK_DIM][innerStep];                            \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        __syncthreads();                                                                                                \n\
    }                                                                                                                   \n\
    PLSTM_GEMM_tcores_inner<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM, WARPS_PER_BLOCK_K,      \n\
        WARPS_PER_BLOCK_M, innerStepSize, unrollSplitK, unrollGemmBatch, hiddenSize, ldh, bidirectionFactor,            \n\
        bidirectionK>(rMat_frags, h_out, blockSplitKFactor, minibatch, batchStart, smemh, smemAccumulate, smemr);       \n\
}                                                                                                                       \n\
__launch_bounds__(BLOCK_DIM, BLOCKS_PER_SM) __global__                                                                  \n\
    void PLSTM_tcores(T_DATA* tmp_i, T_DATA* tmp_h, T_DATA* y, T_DATA* rMat, T_DATA* bias, T_DATA* init_h,              \n\
        T_DATA* init_c, T_DATA* final_h, T_DATA* final_c, int* miniBatchArray, int seqLength, int numElementsTotal)     \n\
{                                                                                                                       \n\
    const int kFragsPerWarp = _kFragsPerWarp;                                                                           \n\
    const int kFragsPerWarpInSM = kFragsPerWarp * (rfSplitFactor > 1 ? 1 : 0) / rfSplitFactor;                          \n\
    const int kFragsPerWarpInRF = kFragsPerWarp - kFragsPerWarpInSM;                                                    \n\
    const int mFragsPerWarp = _mFragsPerWarp;                                                                           \n\
    const int WARPS_PER_BLOCK_K = _WARPS_PER_BLOCK_K;                                                                   \n\
    const int WARPS_PER_BLOCK_M = _WARPS_PER_BLOCK_M;                                                                   \n\
    const int hiddenSize = _hiddenSize;                                                                                 \n\
    const int minibatch = _minibatch;                                                                                   \n\
    const int blockSplitKFactor = _blockSplitKFactor;                                                                   \n\
    const int innerStepSize = _innerStepSize;                                                                           \n\
    const int unrollSplitK = _unrollSplitK;                                                                             \n\
    const int unrollGemmBatch = _unrollGemmBatch;                                                                       \n\
    const int bidirectionFactor = _bidirectionFactor;                                                                   \n\
    const int ldh                                                                                                       \n\
        = kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K + ((kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K) % 64 == 0 ? 8 : 0);\n\
    const size_t smemhRequired = ldh * innerStepSize * sizeof(half);                                                    \n\
    const size_t smemAccumulateRequired                                                                                 \n\
        = ((BLOCK_DIM / 32) * mFragsPerWarp * FRAG_M * innerStepSize) * sizeof(T_ACCUMULATE);                           \n\
    const size_t sharedMemoryRequired = smemhRequired > smemAccumulateRequired ? smemhRequired : smemAccumulateRequired;\n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> rMat_frags[bidirectionFactor]         \n\
                                                                                            [kFragsPerWarpInRF]         \n\
                                                                                            [mFragsPerWarp];            \n\
    char* dynamic_smem = dsmem;                                                                                         \n\
    char* smemr = dsmem;                                                                                                \n\
    PLSTM_load_rMat_tcores<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM, WARPS_PER_BLOCK_K,       \n\
        WARPS_PER_BLOCK_M, bidirectionFactor>(rMat, rMat_frags, blockSplitKFactor, hiddenSize, smemr);                  \n\
    const size_t smemPerK = FRAG_M * FRAG_K * (BLOCK_DIM / 32) * bidirectionFactor * mFragsPerWarp * kFragsPerWarpInSM; \n\
    dynamic_smem += smemPerK * sizeof(T_DATA);                                                                          \n\
    grid_group grid;                                                                                                    \n\
    grid = this_grid();                                                                                                 \n\
    // For storing c timestep-to-timestep.                                                                              \n\
    T_DATA* smemc = (T_DATA*) dynamic_smem;                                                                             \n\
    int tid = blockIdx.x * BLOCK_DIM + threadIdx.x;                                                                     \n\
    int idx;                                                                                                            \n\
    int c_idx;                                                                                                          \n\
    int cPerThread = (hiddenSize * minibatch + BLOCK_DIM * GRID_DIM - 1) / (BLOCK_DIM * GRID_DIM);                      \n\
    for (int k = 0; k < bidirectionFactor; k++)                                                                         \n\
    {                                                                                                                   \n\
        c_idx = threadIdx.x;                                                                                            \n\
        for (idx = tid; idx < minibatch * hiddenSize; idx += GRID_DIM * BLOCK_DIM, c_idx += BLOCK_DIM)                  \n\
        {                                                                                                               \n\
            smemc[k * cPerThread * BLOCK_DIM + c_idx] = init_c[k * minibatch * hiddenSize + idx];                       \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
    dynamic_smem += bidirectionFactor * cPerThread * BLOCK_DIM * sizeof(T_DATA);                                        \n\
    __syncthreads();                                                                                                    \n\
    bool compileTimeBidirectionalCheck = bidirectionFactor == 2;                                                        \n\
    size_t hInForWardOffset = 0;                                                                                        \n\
    size_t hInBackWardOffset = numElementsTotal * hiddenSize;                                                           \n\
    if (seqLength > 0)                                                                                                  \n\
    {                                                                                                                   \n\
        hInBackWardOffset -= hiddenSize * miniBatchArray[seqLength - 1];                                                \n\
    }                                                                                                                   \n\
    for (int step = 0; step < seqLength; step++)                                                                        \n\
    {                                                                                                                   \n\
        // Recurrent GEMM                                                                                               \n\
        // forward pass                                                                                                 \n\
        PLSTM_GEMM_tcores_piped<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM, WARPS_PER_BLOCK_K,  \n\
            WARPS_PER_BLOCK_M, innerStepSize, unrollSplitK, unrollGemmBatch, minibatch, hiddenSize, bidirectionFactor,  \n\
            0>(rMat_frags, final_h, y, tmp_h, step, seqLength, blockSplitKFactor, miniBatchArray, hInForWardOffset,     \n\
            dynamic_smem, smemr);                                                                                       \n\
        if (compileTimeBidirectionalCheck)                                                                              \n\
        {                                                                                                               \n\
            PLSTM_GEMM_tcores_piped<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM,                 \n\
                WARPS_PER_BLOCK_K, WARPS_PER_BLOCK_M, innerStepSize, unrollSplitK, unrollGemmBatch, minibatch,          \n\
                hiddenSize, bidirectionFactor, 1>(rMat_frags, final_h, y, tmp_h, step, seqLength, blockSplitKFactor,    \n\
                miniBatchArray, hInBackWardOffset, dynamic_smem + sharedMemoryRequired, smemr);                         \n\
        }                                                                                                               \n\
        // Wait for tmp_h to be written                                                                                 \n\
        grid.sync();                                                                                                    \n\
        if (step > 0)                                                                                                   \n\
        {                                                                                                               \n\
            hInForWardOffset += hiddenSize * miniBatchArray[step - 1];                                                  \n\
        }                                                                                                               \n\
        // Add non-recurrent part and perform pointwise ops                                                             \n\
        PLSTM_pointwise<bidirectionFactor, 0>(step, tmp_h, bias, y, tmp_i, smemc, final_h, blockSplitKFactor,           \n\
            hiddenSize, minibatch, hInForWardOffset, miniBatchArray[step], miniBatchArray);                             \n\
        if (compileTimeBidirectionalCheck)                                                                              \n\
        {                                                                                                               \n\
            if (step > 0)                                                                                               \n\
            {                                                                                                           \n\
                hInBackWardOffset -= hiddenSize * miniBatchArray[seqLength - step - 1];                                 \n\
            }                                                                                                           \n\
            PLSTM_pointwise<bidirectionFactor, 1>(step, tmp_h, bias, y, tmp_i, smemc, final_h, blockSplitKFactor,       \n\
                hiddenSize, minibatch, hInBackWardOffset, miniBatchArray[seqLength - step - 1], miniBatchArray);        \n\
        }                                                                                                               \n\
        // Wait for y to be written                                                                                     \n\
        grid.sync();                                                                                                    \n\
    }                                                                                                                   \n\
    // copy c back to final c                                                                                           \n\
    if (final_c != NULL)                                                                                                \n\
    {                                                                                                                   \n\
        for (int k = 0; k < bidirectionFactor; k++)                                                                     \n\
        {                                                                                                               \n\
            c_idx = threadIdx.x;                                                                                        \n\
            for (idx = tid; idx < minibatch * hiddenSize; idx += GRID_DIM * BLOCK_DIM, c_idx += BLOCK_DIM)              \n\
            {                                                                                                           \n\
                final_c[k * minibatch * hiddenSize + idx] = smemc[k * cPerThread * BLOCK_DIM + c_idx];                  \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
}";

const char* tCoreSourceSeparate
= "#include <cooperative_groups.h>                                                                                      \n\
using namespace cooperative_groups;                                                                                     \n\
#include <mma.h>                                                                                                        \n\
using namespace nvcuda;                                                                                                 \n\
__device__ __forceinline__ float sigmoidf(float in)                                                                     \n\
{                                                                                                                       \n\
    if (in > 0.f)                                                                                                       \n\
    {                                                                                                                   \n\
        return 1.f / (1.f + expf(-in));                                                                                 \n\
    }                                                                                                                   \n\
    else                                                                                                                \n\
    {                                                                                                                   \n\
        float z = expf(in);                                                                                             \n\
        return z / (1.f + z);                                                                                           \n\
    }                                                                                                                   \n\
}                                                                                                                       \n\
__device__ inline int roundUp(int x, int y)                                                                             \n\
{                                                                                                                       \n\
    return ((x - 1) / y + 1) * y;                                                                                       \n\
}                                                                                                                       \n\
template <int bidirectionFactor, int bidirectionK>                                                                      \n\
__device__ __forceinline__ void PLSTM_pointwise(int step, T_DATA* tmp_h, T_DATA* bias, T_DATA* y, T_DATA* tmp_i,        \n\
    T_DATA* smemc, T_DATA* final_h, int blockSplitKFactor, int hiddenSize, int minibatch, int hInOffset, int validBatch,\n\
    int* miniBatchArray)                                                                                                \n\
{                                                                                                                       \n\
    int tid = blockIdx.x * BLOCK_DIM + threadIdx.x;                                                                     \n\
    int idx;                                                                                                            \n\
    int c_idx;                                                                                                          \n\
    bool compileTimeCopyCheck = (bidirectionK == 1);                                                                    \n\
    for (idx = tid, c_idx = threadIdx.x; idx < validBatch * hiddenSize; idx += GRID_DIM * BLOCK_DIM, c_idx += BLOCK_DIM)\n\
    {                                                                                                                   \n\
        float g[4];                                                                                                     \n\
        int batch = idx / hiddenSize;                                                                                   \n\
        int gateIndex = (idx % hiddenSize) + 4 * batch * hiddenSize;                                                    \n\
        int biasOffset = bidirectionK * 2 * 4 * hiddenSize;                                                             \n\
        for (int i = 0; i < 4; i++)                                                                                     \n\
        {                                                                                                               \n\
            g[i] = (float) tmp_i[(hInOffset * 4 + 4 * batch * hiddenSize) * bidirectionFactor                           \n\
                       + bidirectionK * 4 * hiddenSize + i * hiddenSize + idx % hiddenSize]                             \n\
                + (float) bias[biasOffset + i * hiddenSize + idx % hiddenSize]                                          \n\
                + (float) bias[biasOffset + 4 * hiddenSize + i * hiddenSize + idx % hiddenSize];                        \n\
            for (int k = 0; k < blockSplitKFactor; k++)                                                                 \n\
            {                                                                                                           \n\
                g[i] += (float) tmp_h[k * hiddenSize * 4 * roundUp(minibatch, 8) + i * hiddenSize + gateIndex];         \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        float in_gate = sigmoidf(g[0]);                                                                                 \n\
        float forget_gate = sigmoidf(g[2]);                                                                             \n\
        float in_gate2 = tanhf(g[1]);                                                                                   \n\
        float out_gate = sigmoidf(g[3]);                                                                                \n\
        int cPerThread = (hiddenSize * minibatch + BLOCK_DIM * GRID_DIM - 1) / (BLOCK_DIM * GRID_DIM);                  \n\
        float res = in_gate * in_gate2;                                                                                 \n\
        res += forget_gate * (float) smemc[bidirectionK * cPerThread * BLOCK_DIM + c_idx];                              \n\
        smemc[bidirectionK * cPerThread * BLOCK_DIM + c_idx] = res;                                                     \n\
        res = out_gate * tanhf(res);                                                                                    \n\
        size_t yIdx = hInOffset * bidirectionFactor + batch * bidirectionFactor * hiddenSize + bidirectionK * hiddenSize\n\
            + idx % hiddenSize;                                                                                         \n\
        y[yIdx] = res;                                                                                                  \n\
        if (final_h != NULL)                                                                                            \n\
        {                                                                                                               \n\
            final_h[bidirectionK * miniBatchArray[0] * hiddenSize + idx] = res;                                         \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
}                                                                                                                       \n\
extern __shared__ char dsmem[];                                                                                         \n\
template <int kFragsPerWarp, int mFragsPerWarp, int kFragsPerWarpInRF, int kFragsPerWarpInSM, int WARPS_PER_BLOCK_K,    \n\
    int WARPS_PER_BLOCK_M, int bidirectionFactor, int bidirectionK>                                                     \n\
__device__ __forceinline__ void PLSTM_load_rMat_tcores(const T_DATA* rMat,                                              \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> rMat_frags[kFragsPerWarpInRF]         \n\
                                                                                            [mFragsPerWarp],            \n\
    const int blockSplitKFactor, const int hiddenSize, char* smem)                                                      \n\
{                                                                                                                       \n\
    half* tmpSmem = (half*) smem;                                                                                       \n\
    int kBlock = blockIdx.x % blockSplitKFactor;                                                                        \n\
    int laneIdx = (threadIdx.x % 32);                                                                                   \n\
    int localWarpIdx = threadIdx.x / 32;                                                                                \n\
    int kBlockWarpIdx = (blockIdx.x / blockSplitKFactor) * (BLOCK_DIM / 32) + localWarpIdx;                             \n\
#pragma unroll                                                                                                          \n\
    for (int i = 0; i < kFragsPerWarp; i++)                                                                             \n\
    {                                                                                                                   \n\
#pragma unroll                                                                                                          \n\
        for (int k = 0; k < bidirectionFactor; k++)                                                                     \n\
        {                                                                                                               \n\
#pragma unroll                                                                                                          \n\
            for (int j = 0; j < mFragsPerWarp; j++)                                                                     \n\
            {                                                                                                           \n\
                // For WARPS_PER_BLOCK_K == 2 => warps 0 and 1 co-operate on rows.                                      \n\
                // For WARPS_PER_BLOCK_K == 4 => warps 0, 1, 2 and 3 co-operate on rows.                                \n\
                int fragFirstCol                                                                                        \n\
                    = ((kBlock * WARPS_PER_BLOCK_K + localWarpIdx % WARPS_PER_BLOCK_K) * kFragsPerWarp + i) * FRAG_K;   \n\
                int fragFirstRow = ((kBlockWarpIdx / WARPS_PER_BLOCK_K) * mFragsPerWarp + j) * FRAG_M;                  \n\
                if (i < kFragsPerWarpInRF)                                                                              \n\
                {                                                                                                       \n\
                    // Load the FRAG_M*FRAG_K values                                                                    \n\
                    for (int startIdx = 0; startIdx < FRAG_M * FRAG_K; startIdx += 32)                                  \n\
                    {                                                                                                   \n\
                        int smemCol = (startIdx + laneIdx) % FRAG_K;                                                    \n\
                        int smemRow = (startIdx + laneIdx) / FRAG_K;                                                    \n\
                        int col = fragFirstCol + smemCol;                                                               \n\
                        int row = fragFirstRow + smemRow;                                                               \n\
                        if (col < hiddenSize && row < 4 * hiddenSize)                                                   \n\
                        {                                                                                               \n\
                            tmpSmem[localWarpIdx * FRAG_M * FRAG_K + smemRow * FRAG_K + smemCol]                        \n\
                                = (half)(rMat[bidirectionK * 4 * hiddenSize * hiddenSize + row * hiddenSize + col]);    \n\
                        }                                                                                               \n\
                        else                                                                                            \n\
                        {                                                                                               \n\
                            tmpSmem[localWarpIdx * FRAG_M * FRAG_K + smemRow * FRAG_K + smemCol] = (half)(0.f);         \n\
                        }                                                                                               \n\
                    }                                                                                                   \n\
                    __syncthreads();                                                                                    \n\
                    wmma::load_matrix_sync(rMat_frags[i][j], &(tmpSmem[localWarpIdx * FRAG_M * FRAG_K]), FRAG_K);       \n\
                }                                                                                                       \n\
                else                                                                                                    \n\
                {                                                                                                       \n\
                    int localkFragsIdx = (i - kFragsPerWarpInRF) * mFragsPerWarp + j;                                   \n\
                    const int smemPerK = FRAG_M * FRAG_K * (BLOCK_DIM / 32);                                            \n\
                    int fragFirstCol                                                                                    \n\
                        = ((kBlock * WARPS_PER_BLOCK_K + localWarpIdx % WARPS_PER_BLOCK_K) * kFragsPerWarp + i)         \n\
                        * FRAG_K;                                                                                       \n\
                    int fragFirstRow = ((kBlockWarpIdx / WARPS_PER_BLOCK_K) * mFragsPerWarp + j) * FRAG_M;              \n\
                    // Load the FRAG_M*FRAG_K values                                                                    \n\
                    for (int startIdx = 0; startIdx < FRAG_M * FRAG_K; startIdx += 32)                                  \n\
                    {                                                                                                   \n\
                        int smemCol = (startIdx + laneIdx) % FRAG_K;                                                    \n\
                        int smemRow = (startIdx + laneIdx) / FRAG_K;                                                    \n\
                        int col = fragFirstCol + smemCol;                                                               \n\
                        int row = fragFirstRow + smemRow;                                                               \n\
                        if (col < hiddenSize && row < 4 * hiddenSize)                                                   \n\
                        {                                                                                               \n\
                            tmpSmem[localkFragsIdx * smemPerK + localWarpIdx * FRAG_M * FRAG_K + smemRow * FRAG_K       \n\
                                + smemCol]                                                                              \n\
                                = (half)(rMat[bidirectionK * 4 * hiddenSize * hiddenSize + row * hiddenSize + col]);    \n\
                        }                                                                                               \n\
                        else                                                                                            \n\
                        {                                                                                               \n\
                            tmpSmem[localkFragsIdx * smemPerK + localWarpIdx * FRAG_M * FRAG_K + smemRow * FRAG_K       \n\
                                + smemCol]                                                                              \n\
                                = (half)(0.f);                                                                          \n\
                        }                                                                                               \n\
                    }                                                                                                   \n\
                    __syncthreads();                                                                                    \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        __syncthreads();                                                                                                \n\
    }                                                                                                                   \n\
}                                                                                                                       \n\
template <int kFragsPerWarp, int mFragsPerWarp, int kFragsPerWarpInRF, int kFragsPerWarpInSM, int WARPS_PER_BLOCK_K,    \n\
    int WARPS_PER_BLOCK_M, int bidirectionFactor, int bidirectionK>                                                     \n\
__device__ __forceinline__ void PLSTM_load_rMat_tcores_sm(                                                              \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major>& rMat_frags_sm, char* smem,           \n\
    const int mFragIdx, const int kFragIdx)                                                                             \n\
{                                                                                                                       \n\
    half* tmpSmem = (half*) smem;                                                                                       \n\
    int localWarpIdx = threadIdx.x / 32;                                                                                \n\
    int localkFragsIdx = (kFragIdx - kFragsPerWarpInRF) * mFragsPerWarp + mFragIdx;                                     \n\
    const int smemPerK = FRAG_M * FRAG_K * (BLOCK_DIM / 32);                                                            \n\
    wmma::load_matrix_sync(                                                                                             \n\
        rMat_frags_sm, &(tmpSmem[localkFragsIdx * smemPerK + localWarpIdx * FRAG_M * FRAG_K]), FRAG_K);                 \n\
}                                                                                                                       \n\
template <int kFragsPerWarp, int mFragsPerWarp, int kFragsPerWarpInRF, int kFragsPerWarpInSM, int WARPS_PER_BLOCK_K,    \n\
    int WARPS_PER_BLOCK_M, int innerStepSize, int unrollSplitK, int unrollGemmBatch, int hiddenSize, int ldh,           \n\
    int bidirectionFactor, int bidirectionK>                                                                            \n\
__device__ __forceinline__ void PLSTM_GEMM_tcores_inner(                                                                \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> rMat_frags[kFragsPerWarpInRF]         \n\
                                                                                            [mFragsPerWarp],            \n\
    T_DATA* __restrict__ h_out, const int blockSplitKFactor, const int minibatch, const int batchStart, half* smemh,    \n\
    T_ACCUMULATE* smemAccumulate, char* smemr)                                                                          \n\
{                                                                                                                       \n\
    int kBlock = blockIdx.x % blockSplitKFactor;                                                                        \n\
    int localWarpIdx = threadIdx.x / 32;                                                                                \n\
    int kBlockWarpIdx = (blockIdx.x / blockSplitKFactor) * (BLOCK_DIM / 32) + localWarpIdx;                             \n\
    const int numInnerSteps = (innerStepSize + FRAG_N - 1) / FRAG_N;                                                    \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> rMat_frags_sm;                        \n\
    wmma::fragment<wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, half, wmma::col_major> b_frag[numInnerSteps];                \n\
    wmma::fragment<wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, T_ACCUMULATE> c_frag[mFragsPerWarp][numInnerSteps];       \n\
#pragma unroll                                                                                                          \n\
    for (int innerStep = 0; innerStep < numInnerSteps; innerStep++)                                                     \n\
    {                                                                                                                   \n\
#pragma unroll                                                                                                          \n\
        for (int i = 0; i < mFragsPerWarp; i++)                                                                         \n\
        {                                                                                                               \n\
            wmma::fill_fragment(c_frag[i][innerStep], (T_ACCUMULATE) 0.0f);                                             \n\
        }                                                                                                               \n\
#pragma unroll                                                                                                          \n\
        for (int i = 0; i < kFragsPerWarp; i++)                                                                         \n\
        {                                                                                                               \n\
            int Acol = innerStep * ldh * FRAG_N + ((localWarpIdx % WARPS_PER_BLOCK_K) * kFragsPerWarp + i) * FRAG_K;    \n\
            wmma::load_matrix_sync(b_frag[innerStep], smemh + Acol, ldh);                                               \n\
            if (i < kFragsPerWarpInRF)                                                                                  \n\
            {                                                                                                           \n\
#pragma unroll                                                                                                          \n\
                for (int j = 0; j < mFragsPerWarp; j++)                                                                 \n\
                {                                                                                                       \n\
                    wmma::mma_sync(c_frag[j][innerStep], rMat_frags[i][j], b_frag[innerStep], c_frag[j][innerStep]);    \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
            else                                                                                                        \n\
            {                                                                                                           \n\
#pragma unroll                                                                                                          \n\
                for (int j = 0; j < mFragsPerWarp; j++)                                                                 \n\
                {                                                                                                       \n\
                    PLSTM_load_rMat_tcores_sm<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM,       \n\
                        WARPS_PER_BLOCK_K, WARPS_PER_BLOCK_M, bidirectionFactor, bidirectionK>(                         \n\
                        rMat_frags_sm, smemr, j, i);                                                                    \n\
                    wmma::mma_sync(c_frag[j][innerStep], rMat_frags_sm, b_frag[innerStep], c_frag[j][innerStep]);       \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
    // We're done with smemh                                                                                            \n\
    __syncthreads();                                                                                                    \n\
    // This is true if we have exactly the right number of M elements. Otherwise we have to do a runtime check.         \n\
    bool compileTimeBoundsCheck                                                                                         \n\
        = ((GRID_DIM / blockSplitKFactor) * WARPS_PER_BLOCK_M * mFragsPerWarp) * FRAG_M <= 4 * hiddenSize;              \n\
    // Write out to shared memory so we can reduce across warps and/or do a type conversion.                            \n\
    if (WARPS_PER_BLOCK_K > 1 || sizeof(T_DATA) != sizeof(T_ACCUMULATE))                                                \n\
    {                                                                                                                   \n\
// Tiling is non-trivial. Dimensions:                                                                                   \n\
//   1) The fragments in a warp. We've accumulated the k-fragments, so this spans M.                                    \n\
//   2) The warps. This is k-major, so if WARPS_PER_BLOCK_K == 2, warps 0 and 1 hold values to contribute to the same   \n\
//      output M.                                                                                                       \n\
//   3) The steps across N.                                                                                             \n\
#pragma unroll                                                                                                          \n\
        for (int innerStep = 0; innerStep < numInnerSteps; innerStep++)                                                 \n\
        {                                                                                                               \n\
#pragma unroll                                                                                                          \n\
            for (int i = 0; i < mFragsPerWarp; i++)                                                                     \n\
            {                                                                                                           \n\
                wmma::store_matrix_sync(smemAccumulate                                                                  \n\
                        + (((BLOCK_DIM / 32) * innerStep + localWarpIdx) * mFragsPerWarp + i) * FRAG_M * FRAG_N,        \n\
                    c_frag[i][innerStep], FRAG_M, wmma::mem_col_major);                                                 \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        __syncthreads();                                                                                                \n\
#pragma unroll                                                                                                          \n\
        for (int innerStep = 0; innerStep < numInnerSteps; innerStep++)                                                 \n\
        {                                                                                                               \n\
            size_t hOutOffset;                                                                                          \n\
            if (blockSplitKFactor > 1)                                                                                  \n\
            {                                                                                                           \n\
                hOutOffset = (kBlock * roundUp(minibatch, 8) + (batchStart + innerStep * FRAG_N)) * hiddenSize * 4;     \n\
            }                                                                                                           \n\
            else                                                                                                        \n\
            {                                                                                                           \n\
                hOutOffset = (batchStart + innerStep * FRAG_N) * hiddenSize * 4;                                        \n\
            }                                                                                                           \n\
            // Sometimes it's advantageous to unroll this loop, sometimes not.                                          \n\
            // We can make it a tuneable parameter and autotune for our situation.                                      \n\
            const int unrollFactor                                                                                      \n\
                = unrollSplitK ? (WARPS_PER_BLOCK_M * mFragsPerWarp * FRAG_N * FRAG_M + BLOCK_DIM - 1) / BLOCK_DIM : 1; \n\
#pragma unroll unrollFactor                                                                                             \n\
            for (int startIdx = 0; startIdx < WARPS_PER_BLOCK_M * mFragsPerWarp * FRAG_N * FRAG_M;                      \n\
                 startIdx += BLOCK_DIM)                                                                                 \n\
            {                                                                                                           \n\
                int idx = startIdx + threadIdx.x;                                                                       \n\
                // If the block is larger than the total work we need to check if our index is valid.                   \n\
                // This is a compile time check followed by a runtime check so is optimized out if                      \n\
                // we don't have to do the runtime check.                                                               \n\
                if ((WARPS_PER_BLOCK_M * mFragsPerWarp * FRAG_N * FRAG_M) % BLOCK_DIM != 0                              \n\
                    && idx >= WARPS_PER_BLOCK_M * mFragsPerWarp * FRAG_N * FRAG_M)                                      \n\
                {                                                                                                       \n\
                    break;                                                                                              \n\
                }                                                                                                       \n\
                // If FRAG_N * FRAG_M == BLOCK_DIM then a lot of these values become independent of                     \n\
                // threadIdx, and we remove quite a few integer operations.                                             \n\
                int matrixElement;                                                                                      \n\
                int mTile;                                                                                              \n\
                int mFragId; // 0->mFragsPerWarp                                                                        \n\
                int mWarpId; // 0->WARPS_PER_BLOCK_M                                                                    \n\
                if (FRAG_N * FRAG_M == BLOCK_DIM)                                                                       \n\
                {                                                                                                       \n\
                    matrixElement = threadIdx.x;                                                                        \n\
                    mTile = startIdx / BLOCK_DIM;                                                                       \n\
                    mFragId = mTile % mFragsPerWarp;                                                                    \n\
                    mWarpId = mTile / mFragsPerWarp;                                                                    \n\
                }                                                                                                       \n\
                else                                                                                                    \n\
                {                                                                                                       \n\
                    matrixElement = idx % (FRAG_M * FRAG_N);                                                            \n\
                    mTile = idx / (FRAG_N * FRAG_M);                                                                    \n\
                    mFragId = mTile % mFragsPerWarp;                                                                    \n\
                    mWarpId = mTile / mFragsPerWarp;                                                                    \n\
                }                                                                                                       \n\
                int innerStepOffset = innerStep * (BLOCK_DIM / 32) * mFragsPerWarp * FRAG_M * FRAG_N;                   \n\
                T_DATA res = 0;                                                                                         \n\
#pragma unroll                                                                                                          \n\
                for (int warpK = 0; warpK < WARPS_PER_BLOCK_K; warpK++)                                                 \n\
                {                                                                                                       \n\
                    res += smemAccumulate[innerStepOffset                                                               \n\
                        + (mWarpId * WARPS_PER_BLOCK_K * mFragsPerWarp + mFragId + warpK * mFragsPerWarp) * FRAG_N      \n\
                            * FRAG_M                                                                                    \n\
                        + matrixElement];                                                                               \n\
                }                                                                                                       \n\
                int col = matrixElement / FRAG_M;                                                                       \n\
                int row = ((blockIdx.x / blockSplitKFactor) * WARPS_PER_BLOCK_M * mFragsPerWarp + mTile) * FRAG_M       \n\
                    + matrixElement % FRAG_M;                                                                           \n\
                if (compileTimeBoundsCheck || row < 4 * hiddenSize)                                                     \n\
                {                                                                                                       \n\
                    h_out[hOutOffset + row + col * 4 * hiddenSize] = res;                                               \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        __syncthreads();                                                                                                \n\
    }                                                                                                                   \n\
    else                                                                                                                \n\
    {                                                                                                                   \n\
#pragma unroll                                                                                                          \n\
        for (int innerStep = 0; innerStep < numInnerSteps; innerStep++)                                                 \n\
        {                                                                                                               \n\
            size_t hOutOffset;                                                                                          \n\
            if (blockSplitKFactor > 1)                                                                                  \n\
            {                                                                                                           \n\
                hOutOffset = (kBlock * roundUp(minibatch, 8) + (batchStart + innerStep * FRAG_N)) * hiddenSize * 4;     \n\
            }                                                                                                           \n\
            else                                                                                                        \n\
            {                                                                                                           \n\
                hOutOffset = (batchStart + innerStep * FRAG_N) * hiddenSize * 4;                                        \n\
            }                                                                                                           \n\
#pragma unroll                                                                                                          \n\
            for (int i = 0; i < mFragsPerWarp; i++)                                                                     \n\
            {                                                                                                           \n\
                if ((compileTimeBoundsCheck || kBlockWarpIdx * mFragsPerWarp + i) * FRAG_M < 4 * hiddenSize)            \n\
                {                                                                                                       \n\
                    wmma::store_matrix_sync(                                                                            \n\
                        (T_ACCUMULATE*) h_out + hOutOffset + (kBlockWarpIdx * mFragsPerWarp + i) * FRAG_M,              \n\
                        c_frag[i][innerStep], 4 * hiddenSize, wmma::mem_col_major);                                     \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
}                                                                                                                       \n\
template <int kFragsPerWarp, int mFragsPerWarp, int kFragsPerWarpInRF, int kFragsPerWarpInSM, int WARPS_PER_BLOCK_K,    \n\
    int WARPS_PER_BLOCK_M, int innerStepSize, int unrollSplitK, int unrollGemmBatch, int minibatch, int hiddenSize,     \n\
    int bidirectionFactor, int bidirectionK>                                                                            \n\
__device__ __forceinline__ void PLSTM_GEMM_tcores_piped(                                                                \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> rMat_frags[kFragsPerWarpInRF]         \n\
                                                                                            [mFragsPerWarp],            \n\
    T_DATA* __restrict__ init_h, T_DATA* __restrict__ h_in, T_DATA* __restrict__ h_out, const int step,                 \n\
    const int seqLength, const int blockSplitKFactor, const int* miniBatchArray, const int hInOffset, char* smem,       \n\
    char* smemr)                                                                                                        \n\
{                                                                                                                       \n\
    half* smemh = (half*) smem;                                                                                         \n\
    T_ACCUMULATE* smemAccumulate = (T_ACCUMULATE*) (smemh);                                                             \n\
    int kBlock = blockIdx.x % blockSplitKFactor;                                                                        \n\
    T_DATA registerBuffer[(kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K + BLOCK_DIM - 1) / BLOCK_DIM][innerStepSize];     \n\
    // It's useful to be able to determine at compile time whether conditionals within the data loading loops           \n\
    // need to be executed. This saves registers and allows the compiler to do extra optimizations.                     \n\
    bool compileTimeGmemBoundsCheck1 = (kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K) % BLOCK_DIM == 0                    \n\
        && kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K == hiddenSize / blockSplitKFactor;                                \n\
    bool compileTimeGmemBoundsCheck2 = minibatch % innerStepSize == 0;                                                  \n\
    bool compileTimeSmemBoundsCheck = (kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K) % BLOCK_DIM == 0;                    \n\
    // To avoid bank conflicts inside wmma::load_matrix_sync we want to ensure that consecutive columns                 \n\
    // of the loaded submatrix start at different shared memory banks. We would also like to use vectorized load        \n\
    // instructions, which are 128 bit wide.                                                                            \n\
    // V100 has 32 banks of 32 bit words (See Programming Guide Appendix H). To avoid bank conflicts                    \n\
    // we should therefore offset eah column by 128 bits, or 8 fp16 values.                                             \n\
    const int ldh                                                                                                       \n\
        = kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K + ((kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K) % 64 == 0 ? 8 : 0);\n\
    int batchStart = 0;                                                                                                 \n\
// Load the B matrix from global to shared. Whole block co-operates.                                                    \n\
// This loop looks more complicated than it is due to the nested bounds checking.                                       \n\
// If the if statements are true it becomes a lot simpler.                                                              \n\
#pragma unroll                                                                                                          \n\
    for (int i = 0; i < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K; i += BLOCK_DIM)                                     \n\
    {                                                                                                                   \n\
        int idx = i + threadIdx.x;                                                                                      \n\
        int row = idx + (blockSplitKFactor > 1 ? kBlock * kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K : 0);              \n\
        if (compileTimeGmemBoundsCheck1 || (idx < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K && row < hiddenSize))      \n\
        {                                                                                                               \n\
#pragma unroll                                                                                                          \n\
            for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                             \n\
            {                                                                                                           \n\
                int subExample = batchStart + innerStep;                                                                \n\
                if (compileTimeGmemBoundsCheck2 || (subExample < minibatch))                                            \n\
                {                                                                                                       \n\
                    registerBuffer[i / BLOCK_DIM][innerStep]                                                            \n\
                        = init_h[bidirectionK * miniBatchArray[0] * hiddenSize + subExample * hiddenSize + row];        \n\
                }                                                                                                       \n\
                else                                                                                                    \n\
                {                                                                                                       \n\
                    registerBuffer[i / BLOCK_DIM][innerStep] = 0.f;                                                     \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        else                                                                                                            \n\
        {                                                                                                               \n\
#pragma unroll                                                                                                          \n\
            for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                             \n\
            {                                                                                                           \n\
                registerBuffer[i / BLOCK_DIM][innerStep] = 0.f;                                                         \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
#pragma unroll                                                                                                          \n\
    for (int i = 0; i < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K; i += BLOCK_DIM)                                     \n\
    {                                                                                                                   \n\
        int idx = i + threadIdx.x;                                                                                      \n\
        if (compileTimeSmemBoundsCheck || idx < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K)                             \n\
        {                                                                                                               \n\
            for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                             \n\
            {                                                                                                           \n\
                smemh[innerStep * ldh + idx] = registerBuffer[i / BLOCK_DIM][innerStep];                                \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
    __syncthreads();                                                                                                    \n\
    // Sometimes it's advantageous to unroll this loop, sometimes not.                                                  \n\
    // We can make it a tuneable parameter and autotune for our situation.                                              \n\
    // Now that minibatch size is not constant, not sure what to put here                                               \n\
    const int unrollFactor = unrollGemmBatch ? (minibatch - 1) / innerStepSize : 1;                                     \n\
#pragma unroll unrollFactor                                                                                             \n\
    for (; batchStart < minibatch - innerStepSize; batchStart += innerStepSize)                                         \n\
    {                                                                                                                   \n\
// Same as above, but load the next one in preparation for the next loop iteration.                                     \n\
#pragma unroll                                                                                                          \n\
        for (int i = 0; i < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K; i += BLOCK_DIM)                                 \n\
        {                                                                                                               \n\
            int idx = i + threadIdx.x;                                                                                  \n\
            int row = idx + (blockSplitKFactor > 1 ? kBlock * kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K : 0);          \n\
            if (compileTimeGmemBoundsCheck1 || (idx < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K && row < hiddenSize))  \n\
            {                                                                                                           \n\
#pragma unroll                                                                                                          \n\
                for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                         \n\
                {                                                                                                       \n\
                    int subExample = batchStart + innerStepSize + innerStep;                                            \n\
                    if (compileTimeGmemBoundsCheck2 || (subExample < minibatch))                                        \n\
                    {                                                                                                   \n\
                        registerBuffer[i / BLOCK_DIM][innerStep]                                                        \n\
                            = init_h[bidirectionK * miniBatchArray[0] * hiddenSize + subExample * hiddenSize + row];    \n\
                    }                                                                                                   \n\
                    else                                                                                                \n\
                    {                                                                                                   \n\
                        registerBuffer[i / BLOCK_DIM][innerStep] = 0.f;                                                 \n\
                    }                                                                                                   \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
            else                                                                                                        \n\
            {                                                                                                           \n\
#pragma unroll                                                                                                          \n\
                for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                         \n\
                {                                                                                                       \n\
                    registerBuffer[i / BLOCK_DIM][innerStep] = 0.f;                                                     \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        PLSTM_GEMM_tcores_inner<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM, WARPS_PER_BLOCK_K,  \n\
            WARPS_PER_BLOCK_M, innerStepSize, unrollSplitK, unrollGemmBatch, hiddenSize, ldh, bidirectionFactor,        \n\
            bidirectionK>(rMat_frags, h_out, blockSplitKFactor, minibatch, batchStart, smemh, smemAccumulate, smemr);   \n\
// Because the above call has a __syncthreads after reading from smemh we don't                                         \n\
// need one before writing.                                                                                             \n\
#pragma unroll                                                                                                          \n\
        for (int i = 0; i < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K; i += BLOCK_DIM)                                 \n\
        {                                                                                                               \n\
            int idx = i + threadIdx.x;                                                                                  \n\
            if (compileTimeSmemBoundsCheck || idx < kFragsPerWarp * WARPS_PER_BLOCK_K * FRAG_K)                         \n\
            {                                                                                                           \n\
                for (int innerStep = 0; innerStep < innerStepSize; innerStep++)                                         \n\
                {                                                                                                       \n\
                    smemh[innerStep * ldh + idx] = registerBuffer[i / BLOCK_DIM][innerStep];                            \n\
                }                                                                                                       \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
        __syncthreads();                                                                                                \n\
    }                                                                                                                   \n\
    PLSTM_GEMM_tcores_inner<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM, WARPS_PER_BLOCK_K,      \n\
        WARPS_PER_BLOCK_M, innerStepSize, unrollSplitK, unrollGemmBatch, hiddenSize, ldh, bidirectionFactor,            \n\
        bidirectionK>(rMat_frags, h_out, blockSplitKFactor, minibatch, batchStart, smemh, smemAccumulate, smemr);       \n\
}                                                                                                                       \n\
__launch_bounds__(BLOCK_DIM, BLOCKS_PER_SM) __global__                                                                  \n\
    void PLSTM_tcores(T_DATA* tmp_i, T_DATA* tmp_h, T_DATA* y, T_DATA* rMat, T_DATA* bias, T_DATA* init_h,              \n\
        T_DATA* init_c, T_DATA* final_h, T_DATA* final_c, int* miniBatchArray, int seqLength, int numElementsTotal)     \n\
{                                                                                                                       \n\
    const int kFragsPerWarp = _kFragsPerWarp;                                                                           \n\
    const int kFragsPerWarpInSM = kFragsPerWarp * (rfSplitFactor > 1 ? 1 : 0) / rfSplitFactor;                          \n\
    const int kFragsPerWarpInRF = kFragsPerWarp - kFragsPerWarpInSM;                                                    \n\
    const int mFragsPerWarp = _mFragsPerWarp;                                                                           \n\
    const int WARPS_PER_BLOCK_K = _WARPS_PER_BLOCK_K;                                                                   \n\
    const int WARPS_PER_BLOCK_M = _WARPS_PER_BLOCK_M;                                                                   \n\
    const int hiddenSize = _hiddenSize;                                                                                 \n\
    const int minibatch = _minibatch;                                                                                   \n\
    const int blockSplitKFactor = _blockSplitKFactor;                                                                   \n\
    const int innerStepSize = _innerStepSize;                                                                           \n\
    const int unrollSplitK = _unrollSplitK;                                                                             \n\
    const int unrollGemmBatch = _unrollGemmBatch;                                                                       \n\
    const int bidirectionFactor = _bidirectionFactor;                                                                   \n\
    wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, half, wmma::row_major> rMat_frags[kFragsPerWarpInRF]         \n\
                                                                                            [mFragsPerWarp];            \n\
    char* dynamic_smem = dsmem;                                                                                         \n\
    char* smemr = dsmem;                                                                                                \n\
    PLSTM_load_rMat_tcores<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM, WARPS_PER_BLOCK_K,       \n\
        WARPS_PER_BLOCK_M, bidirectionFactor, 0>(rMat, rMat_frags, blockSplitKFactor, hiddenSize, smemr);               \n\
    if (rfSplitFactor > 1)                                                                                              \n\
    {                                                                                                                   \n\
        const size_t smemPerK = FRAG_M * FRAG_K * (BLOCK_DIM / 32) * mFragsPerWarp * kFragsPerWarpInSM;                 \n\
        const size_t smemPerKTemp = FRAG_M * FRAG_K * (BLOCK_DIM / 32);                                                 \n\
        const size_t smemPerKFinal = smemPerK > smemPerKTemp ? smemPerK : smemPerKTemp;                                 \n\
        dynamic_smem += smemPerKFinal * sizeof(T_DATA);                                                                 \n\
    }                                                                                                                   \n\
    else                                                                                                                \n\
    {                                                                                                                   \n\
        const size_t smemPerK = FRAG_M * FRAG_K * (BLOCK_DIM / 32);                                                     \n\
        dynamic_smem += smemPerK * sizeof(T_DATA);                                                                      \n\
    }                                                                                                                   \n\
    grid_group grid;                                                                                                    \n\
    grid = this_grid();                                                                                                 \n\
    // For storing c timestep-to-timestep.                                                                              \n\
    T_DATA* smemc = (T_DATA*) dynamic_smem;                                                                             \n\
    int tid = blockIdx.x * BLOCK_DIM + threadIdx.x;                                                                     \n\
    int idx;                                                                                                            \n\
    int c_idx;                                                                                                          \n\
    int cPerThread = (hiddenSize * minibatch + BLOCK_DIM * GRID_DIM - 1) / (BLOCK_DIM * GRID_DIM);                      \n\
    for (int k = 0; k < bidirectionFactor; k++)                                                                         \n\
    {                                                                                                                   \n\
        c_idx = threadIdx.x;                                                                                            \n\
        for (idx = tid; idx < minibatch * hiddenSize; idx += GRID_DIM * BLOCK_DIM, c_idx += BLOCK_DIM)                  \n\
        {                                                                                                               \n\
            smemc[k * cPerThread * BLOCK_DIM + c_idx] = init_c[k * minibatch * hiddenSize + idx];                       \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
    dynamic_smem += bidirectionFactor * cPerThread * BLOCK_DIM * sizeof(T_DATA);                                        \n\
    __syncthreads();                                                                                                    \n\
    bool compileTimeBidirectionalCheck = bidirectionFactor == 2;                                                        \n\
    size_t hInForWardOffset = 0;                                                                                        \n\
    size_t hInBackWardOffset = numElementsTotal * hiddenSize;                                                           \n\
    if (seqLength > 0)                                                                                                  \n\
    {                                                                                                                   \n\
        hInBackWardOffset -= hiddenSize * miniBatchArray[seqLength - 1];                                                \n\
    }                                                                                                                   \n\
    for (int step = 0; step < seqLength; step++)                                                                        \n\
    {                                                                                                                   \n\
        // Recurrent GEMM                                                                                               \n\
        // forward pass                                                                                                 \n\
        PLSTM_GEMM_tcores_piped<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM, WARPS_PER_BLOCK_K,  \n\
            WARPS_PER_BLOCK_M, innerStepSize, unrollSplitK, unrollGemmBatch, minibatch, hiddenSize, bidirectionFactor,  \n\
            0>(rMat_frags, final_h, y, tmp_h, step, seqLength, blockSplitKFactor, miniBatchArray, hInForWardOffset,     \n\
            dynamic_smem, smemr);                                                                                       \n\
        // Wait for tmp_h to be written                                                                                 \n\
        grid.sync();                                                                                                    \n\
        if (step > 0)                                                                                                   \n\
        {                                                                                                               \n\
            hInForWardOffset += hiddenSize * miniBatchArray[step - 1];                                                  \n\
        }                                                                                                               \n\
        // Add non-recurrent part and perform pointwise ops                                                             \n\
        PLSTM_pointwise<bidirectionFactor, 0>(step, tmp_h, bias, y, tmp_i, smemc, final_h, blockSplitKFactor,           \n\
            hiddenSize, minibatch, hInForWardOffset, miniBatchArray[step], miniBatchArray);                             \n\
        // Wait for y to be written                                                                                     \n\
        grid.sync();                                                                                                    \n\
    }                                                                                                                   \n\
    if (compileTimeBidirectionalCheck)                                                                                  \n\
    {                                                                                                                   \n\
        PLSTM_load_rMat_tcores<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM, WARPS_PER_BLOCK_K,   \n\
            WARPS_PER_BLOCK_M, bidirectionFactor, 1>(rMat, rMat_frags, blockSplitKFactor, hiddenSize, smemr);           \n\
        for (int step = 0; step < seqLength; step++)                                                                    \n\
        {                                                                                                               \n\
            PLSTM_GEMM_tcores_piped<kFragsPerWarp, mFragsPerWarp, kFragsPerWarpInRF, kFragsPerWarpInSM,                 \n\
                WARPS_PER_BLOCK_K, WARPS_PER_BLOCK_M, innerStepSize, unrollSplitK, unrollGemmBatch, minibatch,          \n\
                hiddenSize, bidirectionFactor, 1>(rMat_frags, final_h, y, tmp_h, step, seqLength, blockSplitKFactor,    \n\
                miniBatchArray, hInBackWardOffset, dynamic_smem, smemr);                                                \n\
            grid.sync();                                                                                                \n\
            if (step > 0)                                                                                               \n\
            {                                                                                                           \n\
                hInBackWardOffset -= hiddenSize * miniBatchArray[seqLength - step - 1];                                 \n\
            }                                                                                                           \n\
            PLSTM_pointwise<bidirectionFactor, 1>(step, tmp_h, bias, y, tmp_i, smemc, final_h, blockSplitKFactor,       \n\
                hiddenSize, minibatch, hInBackWardOffset, miniBatchArray[seqLength - step - 1], miniBatchArray);        \n\
            grid.sync();                                                                                                \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
    // copy c back to final c                                                                                           \n\
    if (final_c != NULL)                                                                                                \n\
    {                                                                                                                   \n\
        for (int k = 0; k < bidirectionFactor; k++)                                                                     \n\
        {                                                                                                               \n\
            c_idx = threadIdx.x;                                                                                        \n\
            for (idx = tid; idx < miniBatchArray[0] * hiddenSize; idx += GRID_DIM * BLOCK_DIM, c_idx += BLOCK_DIM)      \n\
            {                                                                                                           \n\
                final_c[k * miniBatchArray[0] * hiddenSize + idx] = smemc[k * cPerThread * BLOCK_DIM + c_idx];          \n\
            }                                                                                                           \n\
        }                                                                                                               \n\
    }                                                                                                                   \n\
}";
