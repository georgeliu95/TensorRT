#ifndef TRT_SSD_MACROS_H
#define TRT_SSD_MACROS_H
#include "ssd.h"
#include <cassert>
#include <cstdio>

const char* cublasGetErrorString(cublasStatus_t error);

#ifndef DEBUG

#define SSD_ASSERT_PARAM(exp)        \
    do                               \
    {                                \
        if (!(exp))                  \
            return STATUS_BAD_PARAM; \
    } while (0)

#define SSD_ASSERT_FAILURE(exp)    \
    do                             \
    {                              \
        if (!(exp))                \
            return STATUS_FAILURE; \
    } while (0)

#define CSC(call, err)                 \
    do                                 \
    {                                  \
        cudaError_t cudaStatus = call; \
        if (cudaStatus != cudaSuccess) \
        {                              \
            return err;                \
        }                              \
    } while (0)

#define DEBUG_PRINTF(...) \
    do                    \
    {                     \
    } while (0)

#else

#define SSD_ASSERT_PARAM(exp)                                                     \
    do                                                                            \
    {                                                                             \
        if (!(exp))                                                               \
        {                                                                         \
            fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__); \
            return STATUS_BAD_PARAM;                                              \
        }                                                                         \
    } while (0)

#define SSD_ASSERT_FAILURE(exp)                                                 \
    do                                                                          \
    {                                                                           \
        if (!(exp))                                                             \
        {                                                                       \
            fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__); \
            return STATUS_FAILURE;                                              \
        }                                                                       \
    } while (0)

#define CSC(call, err)                                                                          \
    do                                                                                          \
    {                                                                                           \
        cudaError_t cudaStatus = call;                                                          \
        if (cudaStatus != cudaSuccess)                                                          \
        {                                                                                       \
            printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
            return err;                                                                         \
        }                                                                                       \
    } while (0)

#define DEBUG_PRINTF(...)    \
    do                       \
    {                        \
        printf(__VA_ARGS__); \
    } while (0)

#endif

#endif // TRT_SSD_MACROS_H