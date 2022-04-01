#include "cuda-helper.h"
#include <time.h>

/**
 * \brief this ptx string is compiled from NVCC (10.2.89)
 * \note source is:
 * ```cu
 *   __global__ void sumArray(const float *A, const float *B, float *C) {
 *       C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
 *   }
 * ```
 */
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ".visible .entry _Z8sumArrayPKfS0_Pf(\n"
                                "   .param .u64 _Z8sumArrayPKfS0_Pf_param_0,\n"
                                "   .param .u64 _Z8sumArrayPKfS0_Pf_param_1,\n"
                                "   .param .u64 _Z8sumArrayPKfS0_Pf_param_2\n"
                                "){"
                                "   .reg .f32    %f<4>;\n"
                                "   .reg .b32    %r<2>;\n"
                                "   .reg .b64    %rd<11>;\n"
                                ""
                                "   ld.param.u64    %rd1, [_Z8sumArrayPKfS0_Pf_param_0];\n"
                                "   ld.param.u64    %rd2, [_Z8sumArrayPKfS0_Pf_param_1];\n"
                                "   ld.param.u64    %rd3, [_Z8sumArrayPKfS0_Pf_param_2];\n"
                                "   cvta.to.global.u64    %rd4, %rd3;\n"
                                "   cvta.to.global.u64    %rd5, %rd2;\n"
                                "   cvta.to.global.u64    %rd6, %rd1;\n"
                                "   mov.u32    %r1, %tid.x;\n"
                                "   mul.wide.u32    %rd7, %r1, 4;\n"
                                "   add.s64    %rd8, %rd6, %rd7;\n"
                                "   ld.global.f32    %f1, [%rd8];\n"
                                "   add.s64    %rd9, %rd5, %rd7;\n"
                                "   ld.global.f32    %f2, [%rd9];\n"
                                "   add.f32    %f3, %f1, %f2;\n"
                                "   add.s64    %rd10, %rd4, %rd7;\n"
                                "   st.global.f32    [%rd10], %f3;\n"
                                "   ret;"
                                "}";


void initRandomArray(float *a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = (float) (time(NULL) * 0x789 + 0x3456 % 0x12345678);
    }
}


void sumArray(CUdevice device, const float *h_A, const float *h_B, float *h_C, size_t nElem) {
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func = NULL;
    CUdeviceptr d_A, d_B, d_C;
    size_t nBytes = nElem * sizeof(float);


    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&d_A, nBytes));
    cuError(cuMemAlloc(&d_B, nBytes));
    cuError(cuMemAlloc(&d_C, nBytes));
    cuError(cuMemcpyHtoD(d_A, h_A, nBytes));
    cuError(cuMemcpyHtoD(d_B, h_B, nBytes));

    printf("  Loading module from ptx ...\n");
    cuError(cuModuleLoadData(&module, KERNEL_PTX));
    printf("  Load module from ptx Success\n");

    printf("  Loading function from module ...\n");
    cuError(cuModuleGetFunction(&func, module, "_Z8sumArrayPKfS0_Pf"));
    printf("  Load function from module Success\n");

    printf("  Launching kernel function ...\n");
    void *args[] = {&d_A, &d_B, &d_C, &nElem, NULL};
    cuError(cuLaunchKernel(func, 1, 1, 1, nElem, 1, 1, 0, NULL, args, NULL));
    printf("  Launch kernel function Success\n");

    cuError(cuMemcpyDtoH(h_C, d_C, nBytes));
    cuError(cuMemFree(d_A));
    cuError(cuMemFree(d_B));
    cuError(cuMemFree(d_C));
    cuError(cuModuleUnload(module));
    cuError(cuCtxDestroy(context));
}


int main() {
    static const size_t nElem = 1024;
    float *h_a, *h_b, *gpu_sum, *cpu_sum;
    char name[GPU_DEVICE_NAME_SIZE];
    int deviceCount = 0;
    CUdevice device = 0;

    h_a = (float *) malloc(nElem * sizeof(float));
    h_b = (float *) malloc(nElem * sizeof(float));
    cpu_sum = (float *) malloc(nElem * sizeof(float));
    gpu_sum = (float *) malloc(nElem * sizeof(float));
    initRandomArray(h_a, nElem);
    initRandomArray(h_b, nElem);
    for (size_t i = 0; i < nElem; ++i) {
        cpu_sum[i] = h_a[i] + h_b[i];
    }

    printf("Initialize array Success. The array size = %zu\n", nElem);

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    printf("GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        printf("\nTesting array sum on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        printf("  GPU device name is: '%s'\n", name);

        sumArray(device, h_a, h_b, gpu_sum, nElem);

        printf("  Checking the sum of two array ...\n");
        for (size_t j = 0; j < nElem; ++j) {
            float diff = gpu_sum[j] - cpu_sum[j];
            if (diff > 1e-6 || diff < -1e-6) {
                LOG_ERROR("  Test Fail! Index %zu expect %f, but got %f\n", j, cpu_sum[j], gpu_sum[j]);
            }
        }
        printf("  Check the sum of two array OK, calculate the sum Success\n");

        printf("Test array sum on GPU device %d Success\n", i);
    }
    return 0;
}
