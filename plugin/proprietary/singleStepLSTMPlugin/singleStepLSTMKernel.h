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
                            cudaStream_t streamh);

#endif // SINGLE_STEP_LSTM_KERNEL_H
