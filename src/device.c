#include <cuda-helper.h>

void printDevice() {
    char name[GPU_DEVICE_NAME_SIZE];
    int deviceCount = 0;
    CUdevice device = 0;
    size_t totalGlobalMem = 0;
    int major = 0, minor = 0;

    cuError(cuDeviceGetCount(&deviceCount));
    printf("GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        printf("\nInformation for GPU device %d: \n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        printf("  GPU device name is: '%s'\n", name);

        cuError(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        cuError(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        printf("  GPU device has SM %d.%d compute capability\n", major, minor);

        cuError(cuDeviceTotalMem(&totalGlobalMem, device));
        printf("  Total amount of global memory: %zu MB (%zu bytes)\n", (totalGlobalMem >> 20), totalGlobalMem);
        printf("  64-bit Memory Address: %s\n",
               ((unsigned long long) totalGlobalMem > ((unsigned long long) (1LL) << 32)) ? "YES" : "NO");
    }
}

int main() {
    cuError(cuInit(0));

    printDevice();

    return 0;
}
