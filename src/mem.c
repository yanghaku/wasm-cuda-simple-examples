#include <cuda-helper.h>

void memoryTest(CUdevice device) {
    CUcontext context = NULL;
    size_t free_size = 0, total_size = 0;
    size_t alloc_size, element_number;
    CUdeviceptr d_a, d_b;
    int *h_a, *h_b;

    printf("  Testing 'cuCtxCreate' ...\n");
    cuError(cuCtxCreate(&context, 0, device));
    printf("  Test 'cuCtxCreate' Success\n");

    printf("  Testing 'cuMemAlloc' ...\n");

    cuError(cuMemGetInfo(&free_size, &total_size));
    printf("  Before 'cuMemAlloc', memory info: { free=%zu, total=%zu }\n", free_size, total_size);

    alloc_size = free_size / 10;
    if (alloc_size < sizeof(int)) {
        LOG_ERROR("  No more free memory to alloc\n  Test Fail\n");
    }
    element_number = alloc_size / sizeof(int);
    if (alloc_size % sizeof(int) != 0) {
        alloc_size = element_number * sizeof(int);
    }

    printf("  Memory alloc size = %zu bytes\n", alloc_size * 2);
    cuError(cuMemAlloc(&d_a, alloc_size));
    cuError(cuMemAlloc(&d_b, alloc_size));

    cuError(cuMemGetInfo(&free_size, &total_size));
    printf("  After 'cuMemAlloc', memory info: { free=%zu, total=%zu }\n", free_size, total_size);

    printf("  Test 'cuMemAlloc' Success\n");

    // alloc host memory
    h_a = (int *) malloc(alloc_size);
    h_b = (int *) malloc(alloc_size);
    if (h_a == NULL || h_b == NULL) {
        LOG_ERROR("  Alloc host memory fail\n  Test Fail\n");
    }

    printf("  Testing 'cuMemcpyHtoD', 'cuMemcpyDtoH', 'cuMemcpyDtoD' ...\n");
    for (size_t i = 0; i < element_number; ++i) {
        h_a[i] = 0x12365498;
        h_b[i] = 0;
    }

    // h_a -> d_a -> d_b -> h_b
    cuError(cuMemcpyHtoD(d_a, h_a, alloc_size));
    cuError(cuMemcpyDtoD(d_b, d_a, alloc_size));
    cuError(cuMemcpyDtoH(h_b, d_b, alloc_size));

    for (size_t i = 0; i < element_number; ++i) {
        if (h_b[i] != 0x12365498) {
            LOG_ERROR("  Test 'cuMemcpyHtoD', 'cuMemcpyDtoH', 'cuMemcpyDtoD' Fail\n");
        }
    }
    printf("  Test 'cuMemcpyHtoD', 'cuMemcpyDtoH', 'cuMemcpyDtoD' Success\n");


    printf("  Testing 'cuMemsetD8', 'cuMemsetD16', 'cuMemsetD32' ...\n");
    cuError(cuMemsetD8(d_b, 0x67, alloc_size));
    cuError(cuMemcpyDtoH(h_b, d_b, alloc_size));
    for (size_t i = 0; i < element_number; ++i) {
        if (h_b[i] != 0x67676767) {
            LOG_ERROR("  Test 'cuMemsetD8' Fail\n");
        }
    }
    cuError(cuMemsetD16(d_a, 0x1234, alloc_size / 2));
    cuError(cuMemcpyDtoH(h_a, d_a, alloc_size));
    for (size_t i = 0; i < element_number; ++i) {
        if (h_a[i] != 0x12341234) {
            LOG_ERROR("  Test 'cuMemsetD16' Fail\n");
        }
    }
    cuError(cuMemsetD32(d_b, 0x19283746, alloc_size / 4));
    cuError(cuMemcpyDtoH(h_b, d_b, alloc_size));
    for (size_t i = 0; i < element_number; ++i) {
        if (h_b[i] != 0x19283746) {
            LOG_ERROR("  Test 'cuMemsetD32' Fail\n");
        }
    }
    printf("  Test 'cuMemsetD8', 'cuMemsetD16', 'cuMemsetD32' Success\n");

    printf("  Testing 'cuMemfree' ...\n");
    cuError(cuMemFree(d_a));
    cuError(cuMemFree(d_b));
    printf("  Test 'cuMemfree' Success\n");

    printf("  Testing 'cuCtxDestroy' ...\n");
    cuError(cuCtxDestroy(context));
    printf("  Test 'cuCtxDestroy' Success\n");

    free(h_a);
    free(h_b);
}

int main() {
    char name[GPU_DEVICE_NAME_SIZE];
    int deviceCount = 0;
    CUdevice device = 0;
    size_t totalGlobalMem = 0;

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    printf("GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        printf("\nTesting Memory Management Operations on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        printf("  GPU device name is: '%s'\n", name);

        cuError(cuDeviceTotalMem(&totalGlobalMem, device));
        printf("  Total amount of global memory: %zu MB (%zu bytes)\n", (totalGlobalMem >> 20), totalGlobalMem);

        memoryTest(device);
        printf("Test Memory Management Operations on GPU device %d Success\n", i);
    }
    return 0;
}
