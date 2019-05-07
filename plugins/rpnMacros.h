#ifndef TRT_RPN_MACROS_H
#define TRT_RPN_MACROS_H
#include "rpnlayer.h"
#include <cstdio>

// Whether to print debug messages for RPN
#define DEBUG_RPN_ENABLE 0

#define DEBUG_PRINTF(...)        \
    do                           \
    {                            \
        if (DEBUG_RPN_ENABLE)    \
        {                        \
            printf(__VA_ARGS__); \
        }                        \
    } while (0)

#define DEBUG_FPRINTF(...)        \
    do                            \
    {                             \
        if (DEBUG_RPN_ENABLE)     \
        {                         \
            fprintf(__VA_ARGS__); \
        }                         \
    } while (0)

#define FRCNN_ASSERT_PARAM(exp)                                                         \
    do                                                                                  \
    {                                                                                   \
        if (!(exp))                                                                     \
        {                                                                               \
            DEBUG_FPRINTF(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__); \
            return STATUS_BAD_PARAM;                                                    \
        }                                                                               \
    } while (0)

#define FRCNN_ASSERT_FAILURE(exp)                                                     \
    do                                                                                \
    {                                                                                 \
        if (!(exp))                                                                   \
        {                                                                             \
            DEBUG_FPRINTF(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__); \
            return STATUS_FAILURE;                                                    \
        }                                                                             \
    } while (0)

#define CSC(call, err)                                                                                \
    do                                                                                                \
    {                                                                                                 \
        cudaError_t cudaStatus = call;                                                                \
        if (cudaStatus != cudaSuccess)                                                                \
        {                                                                                             \
            DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
            return err;                                                                               \
        }                                                                                             \
    } while (0)

#define CHECK(status)                                                                             \
    {                                                                                             \
        if ((status) != 0)                                                                        \
        {                                                                                         \
            DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
            abort();                                                                              \
        }                                                                                         \
    }

#endif // TRT_RPN_MACROS_H
