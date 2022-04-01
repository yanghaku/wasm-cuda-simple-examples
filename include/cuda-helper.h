#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * \brief log error to stderr and exit
 */
#define LOG_ERROR(fmt, ...)                                                                        \
    do {                                                                                           \
        fprintf(stderr, fmt, ##__VA_ARGS__);                                                       \
        exit(-1);                                                                                  \
    } while (0)

/**
 * \brief check the CUResult.
 * if error, print the error to stderr and exit.
 */
#define cuError(err)                                                                               \
    do {                                                                                           \
        char *str;                                                                                 \
        if ((err) != CUDA_SUCCESS) {                                                               \
            cuGetErrorName(err, (const char **)&str);                                              \
            LOG_ERROR("CUDA driver API error = %04d \"%s\" line %d\n", err, str, __LINE__);        \
        }                                                                                          \
    } while (0)

/**
 * the max size of GPU device name is 256
 * \sa cudaDeviceProp in cuda_runtime.h
 */
#define GPU_DEVICE_NAME_SIZE 256

#ifdef _MSC_VER

#include <Windows.h>

#define SET_TIME(t0)                                                                               \
    long long(t0);                                                                                 \
    GetSystemTimePreciseAsFileTime(&(t0));

#define GET_DURING(t1, t0) (((double)((t1) - (t0))) / 10000.0)

#else

#include <sys/time.h>

#define SET_TIME(t0)                                                                               \
    struct timeval(t0);                                                                            \
    gettimeofday(&(t0), NULL);

#define GET_DURING(t1, t0)                                                                         \
    ((double)((t1).tv_sec - (t0).tv_sec) * 1000 + (double)((t1).tv_usec - (t0).tv_usec) / 1000.0)

#endif

#endif // CUDA_HELPER_H
