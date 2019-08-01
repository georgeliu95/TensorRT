#ifndef SINGLE_STEP_LSTM_KERNEL_H
#define SINGLE_STEP_LSTM_KERNEL_H

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <stdio.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define BATCHED_GEMM 0
#define NEW_GEMM 1
#define USE_INTERLEAVED_OUTPUT 0
#define CONCAT_IN_GEMM 1
 
#ifndef AUTOTUNING
#define NUM_SPLIT_K_STREAMS 2
#define M_STEPS1(x) 4
#define K_WARPS1(x) 4
#define M_STEPS2(x) 4
#define K_WARPS2(x) 4
#endif

template<typename T_GEMM_IN, cudaDataType_t dataTypeIn, typename T_GEMM_OUT, cudaDataType_t dataTypeOut>
void singleStepLSTMKernel(int hiddenSize, 
                            int inputSize,
                            int miniBatch, 
                            int seqLength, 
                            int numLayers,
                            cublasHandle_t cublasHandle,
                            T_GEMM_IN *x, 
                            T_GEMM_IN **hx, 
                            T_GEMM_IN **cx, 
                            T_GEMM_IN **w, 
                            T_GEMM_IN **bias,
                            T_GEMM_IN *y, 
                            T_GEMM_IN **hy, 
                            T_GEMM_IN **cy,
                            T_GEMM_IN *concatData,
                            T_GEMM_IN *tmp_io,
                            T_GEMM_OUT *tmp_i,
                            T_GEMM_OUT *tmp_h,
#if (BATCHED_GEMM)
                            T_GEMM_IN **aPtrs,
                            T_GEMM_IN **bPtrs,
                            T_GEMM_IN **cPtrs,
#endif
                            cudaStream_t streami,
                            cudaStream_t* splitKStreams,
                            cudaEvent_t* splitKEvents,
                            int numSplitKStreams,
                            cudaStream_t streamh);

#endif // SINGLE_STEP_LSTM_KERNEL_H

