#include "cuda-helper.h"
#include <time.h>
#include <math.h>


/**
 * \brief this ptx string is compiled from NVCC (10.2.89)
 * \note source is:
 * ```cu
 * __global__ void mulMatrix(const int *A, const int *B, int *C, unsigned nRow) {
 *   unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
 *   unsigned line = blockIdx.y * blockDim.y + threadIdx.y;
 *
 *   if (column >= nRow || line >= nRow)
 *       return;
 *
 *   int sum = 0;
 *   unsigned beginA = nRow * line;
 *   unsigned beginB = column;
 *
 *   for (unsigned i = 0; i < nRow; i++) {
 *       sum += A[beginA + i] * B[i * nRow + beginB];
 *   }
 *   C[line * nRow + column] = sum;
 * }
 * ```
 */
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ".visible .entry _Z9mulMatrixPKiS0_Pij(\n"
                                "    .param .u64 _Z9mulMatrixPKiS0_Pij_param_0,\n"
                                "    .param .u64 _Z9mulMatrixPKiS0_Pij_param_1,\n"
                                "    .param .u64 _Z9mulMatrixPKiS0_Pij_param_2,\n"
                                "    .param .u32 _Z9mulMatrixPKiS0_Pij_param_3\n"
                                "){\n"
                                "    .reg .pred     %p<10>;\n"
                                "    .reg .b32     %r<80>;\n"
                                "    .reg .b64     %rd<37>;\n"
                                ""
                                "    ld.param.u64     %rd4, [_Z9mulMatrixPKiS0_Pij_param_0];\n"
                                "    ld.param.u64     %rd5, [_Z9mulMatrixPKiS0_Pij_param_1];\n"
                                "    ld.param.u64     %rd3, [_Z9mulMatrixPKiS0_Pij_param_2];\n"
                                "    ld.param.u32     %r20, [_Z9mulMatrixPKiS0_Pij_param_3];\n"
                                "    cvta.to.global.u64     %rd1, %rd5;\n"
                                "    cvta.to.global.u64     %rd2, %rd4;\n"
                                "    mov.u32     %r21, %ntid.x;\n"
                                "    mov.u32     %r22, %ctaid.x;\n"
                                "    mov.u32     %r23, %tid.x;\n"
                                "    mad.lo.s32     %r1, %r21, %r22, %r23;\n"
                                "    mov.u32     %r24, %ntid.y;\n"
                                "    mov.u32     %r25, %ctaid.y;\n"
                                "    mov.u32     %r26, %tid.y;\n"
                                "    mad.lo.s32     %r2, %r24, %r25, %r26;\n"
                                "    setp.ge.u32    %p1, %r2, %r20;\n"
                                "    setp.ge.u32    %p2, %r1, %r20;\n"
                                "    or.pred      %p3, %p1, %p2;\n"
                                "    @%p3 bra     BB0_13;\n"
                                ""
                                "    mul.lo.s32     %r3, %r2, %r20;\n"
                                "    setp.eq.s32    %p4, %r20, 0;\n"
                                "    mov.u32     %r79, 0;\n"
                                "    @%p4 bra     BB0_12;\n"
                                ""
                                "    and.b32      %r34, %r20, 3;\n"
                                "    mov.u32     %r75, 0;\n"
                                "    setp.eq.s32    %p5, %r34, 0;\n"
                                "    @%p5 bra     BB0_3;\n"
                                ""
                                "    setp.eq.s32    %p6, %r34, 1;\n"
                                "    @%p6 bra     BB0_5;\n"
                                "    bra.uni     BB0_6;\n"
                                ""
                                "BB0_5:\n"
                                "    mov.u32     %r74, %r75;\n"
                                "    bra.uni     BB0_9;\n"
                                ""
                                "BB0_3:\n"
                                "    mov.u32     %r79, %r75;\n"
                                "    bra.uni     BB0_10;\n"
                                ""
                                "BB0_6:\n"
                                "    setp.eq.s32    %p7, %r34, 2;\n"
                                "    mov.u32     %r72, %r75;\n"
                                "    @%p7 bra     BB0_8;\n"
                                ""
                                "    mul.wide.u32     %rd6, %r3, 4;\n"
                                "    add.s64     %rd7, %rd2, %rd6;\n"
                                "    mul.wide.u32     %rd8, %r1, 4;\n"
                                "    add.s64     %rd9, %rd1, %rd8;\n"
                                "    ld.global.u32     %r36, [%rd9];\n"
                                "    ld.global.u32     %r37, [%rd7];\n"
                                "    mul.lo.s32     %r72, %r36, %r37;\n"
                                "    mov.u32     %r75, 1;\n"
                                ""
                                "BB0_8:\n"
                                "    add.s32     %r38, %r75, %r3;\n"
                                "    mul.wide.u32     %rd10, %r38, 4;\n"
                                "    add.s64     %rd11, %rd2, %rd10;\n"
                                "    neg.s32     %r39, %r75;\n"
                                "    and.b32      %r40, %r39, %r20;\n"
                                "    add.s32     %r41, %r40, %r1;\n"
                                "    mul.wide.u32     %rd12, %r41, 4;\n"
                                "    add.s64     %rd13, %rd1, %rd12;\n"
                                "    ld.global.u32     %r42, [%rd13];\n"
                                "    ld.global.u32     %r43, [%rd11];\n"
                                "    mad.lo.s32     %r74, %r42, %r43, %r72;\n"
                                "    add.s32     %r75, %r75, 1;\n"
                                ""
                                "BB0_9:\n"
                                "    add.s32     %r44, %r75, %r3;\n"
                                "    mul.wide.u32     %rd14, %r44, 4;\n"
                                "    add.s64     %rd15, %rd2, %rd14;\n"
                                "    mad.lo.s32     %r45, %r75, %r20, %r1;\n"
                                "    mul.wide.u32     %rd16, %r45, 4;\n"
                                "    add.s64     %rd17, %rd1, %rd16;\n"
                                "    ld.global.u32     %r46, [%rd17];\n"
                                "    ld.global.u32     %r47, [%rd15];\n"
                                "    mad.lo.s32     %r79, %r46, %r47, %r74;\n"
                                "    add.s32     %r75, %r75, 1;\n"
                                ""
                                "BB0_10:\n"
                                "    setp.lt.u32    %p8, %r20, 4;\n"
                                "    @%p8 bra     BB0_12;\n"
                                ""
                                "BB0_11:\n"
                                "    add.s32     %r48, %r75, %r3;\n"
                                "    mul.wide.u32     %rd18, %r48, 4;\n"
                                "    add.s64     %rd19, %rd2, %rd18;\n"
                                "    mad.lo.s32     %r49, %r75, %r20, %r1;\n"
                                "    mul.wide.u32     %rd20, %r49, 4;\n"
                                "    add.s64     %rd21, %rd1, %rd20;\n"
                                "    ld.global.u32     %r50, [%rd21];\n"
                                "    ld.global.u32     %r51, [%rd19];\n"
                                "    mad.lo.s32     %r52, %r50, %r51, %r79;\n"
                                "    add.s32     %r53, %r75, 1;\n"
                                "    add.s32     %r54, %r53, %r3;\n"
                                "    mul.wide.u32     %rd22, %r54, 4;\n"
                                "    add.s64     %rd23, %rd2, %rd22;\n"
                                "    mad.lo.s32     %r55, %r53, %r20, %r1;\n"
                                "    mul.wide.u32     %rd24, %r55, 4;\n"
                                "    add.s64     %rd25, %rd1, %rd24;\n"
                                "    ld.global.u32     %r56, [%rd25];\n"
                                "    ld.global.u32     %r57, [%rd23];\n"
                                "    mad.lo.s32     %r58, %r56, %r57, %r52;\n"
                                "    add.s32     %r59, %r75, 2;\n"
                                "    add.s32     %r60, %r59, %r3;\n"
                                "    mul.wide.u32     %rd26, %r60, 4;\n"
                                "    add.s64     %rd27, %rd2, %rd26;\n"
                                "    mad.lo.s32     %r61, %r59, %r20, %r1;\n"
                                "    mul.wide.u32     %rd28, %r61, 4;\n"
                                "    add.s64     %rd29, %rd1, %rd28;\n"
                                "    ld.global.u32     %r62, [%rd29];\n"
                                "    ld.global.u32     %r63, [%rd27];\n"
                                "    mad.lo.s32     %r64, %r62, %r63, %r58;\n"
                                "    add.s32     %r65, %r75, 3;\n"
                                "    add.s32     %r66, %r65, %r3;\n"
                                "    mul.wide.u32     %rd30, %r66, 4;\n"
                                "    add.s64     %rd31, %rd2, %rd30;\n"
                                "    mad.lo.s32     %r67, %r65, %r20, %r1;\n"
                                "    mul.wide.u32     %rd32, %r67, 4;\n"
                                "    add.s64     %rd33, %rd1, %rd32;\n"
                                "    ld.global.u32     %r68, [%rd33];\n"
                                "    ld.global.u32     %r69, [%rd31];\n"
                                "    mad.lo.s32     %r79, %r68, %r69, %r64;\n"
                                "    add.s32     %r75, %r75, 4;\n"
                                "    setp.lt.u32    %p9, %r75, %r20;\n"
                                "    @%p9 bra     BB0_11;\n"
                                ""
                                "BB0_12:\n"
                                "    add.s32     %r70, %r3, %r1;\n"
                                "    cvta.to.global.u64     %rd34, %rd3;\n"
                                "    mul.wide.u32     %rd35, %r70, 4;\n"
                                "    add.s64     %rd36, %rd34, %rd35;\n"
                                "    st.global.u32     [%rd36], %r79;\n"
                                ""
                                "BB0_13:\n"
                                "    ret;\n"
                                "}";

//32x32
#define NTHREADS_X 32
#define NTHREADS_Y 32

void initMatrix(int *a, const size_t row, const size_t col) {
    size_t nElem = row * col;
    for (size_t i = 0; i < nElem; ++i) {
        // use 1000 as mod to avoid overflow
        a[i] = (int) (time(NULL) * 0x789 + 0x3456 % 1000);
    }
}


void mulMatrix(CUdevice device, const int *h_A, const int *h_B, int *h_C, size_t nRow) {
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func = NULL;
    CUdeviceptr d_A, d_B, d_C;
    size_t nBytes = nRow * nRow * sizeof(float);

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
    cuError(cuModuleGetFunction(&func, module, "_Z9mulMatrixPKiS0_Pij"));
    printf("  Load function from module Success\n");

    printf("  Launching kernel function ...\n");

    unsigned block_x = ceil((double) nRow / NTHREADS_X);
    unsigned block_y = ceil((double) nRow / NTHREADS_Y);
    void *args[] = {&d_A, &d_B, &d_C, &nRow, NULL};
    cuError(cuLaunchKernel(func, block_x, block_y, 1, NTHREADS_X, NTHREADS_Y, 1, 0, NULL, args, NULL));

    printf("  Launch kernel function Success\n");

    cuError(cuMemcpyDtoH(h_C, d_C, nBytes));
    cuError(cuMemFree(d_A));
    cuError(cuMemFree(d_B));
    cuError(cuMemFree(d_C));
    cuError(cuModuleUnload(module));
    cuError(cuCtxDestroy(context));
}

int main() {
    static const size_t nRow = 1024; // in this program, we assume that rows=cols, and sizeof(matrix_1)=sizeof(matrix_2)
    int *matrix_1, *matrix_2, *gpu_mul, *cpu_mul;
    char name[GPU_DEVICE_NAME_SIZE];
    int deviceCount = 0;
    CUdevice device = 0;
    size_t nBytes = nRow * nRow * sizeof(int);

    matrix_1 = (int *) malloc(nBytes);
    matrix_2 = (int *) malloc(nBytes);
    cpu_mul = (int *) malloc(nBytes);
    gpu_mul = (int *) malloc(nBytes);

    initMatrix(matrix_1, nRow, nRow);
    initMatrix(matrix_2, nRow, nRow);

    SET_TIME(time_0)
    for (size_t i = 0; i < nRow; ++i) {
        for (size_t j = 0; j < nRow; ++j) {
            int sum = 0;
            for (size_t k = 0; k < nRow; ++k) {
                sum += matrix_1[i * nRow + k] * matrix_2[k * nRow + j];
            }
            cpu_mul[i * nRow + j] = sum;
        }
    }
    SET_TIME(time_1)

    printf("Initialize matrix Success. The matrix size = %zux%zu\n", nRow, nRow);
    printf("Calculate on CPU token %lf ms\n", GET_DURING(time_1, time_0));

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    printf("GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        printf("\nTesting matrix multiply on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        printf("  GPU device name is: '%s'\n", name);

        SET_TIME(t2)
        mulMatrix(device, matrix_1, matrix_2, gpu_mul, nRow);
        SET_TIME(t3)

        printf("  Calculate on GPU device %d token %lf ms\n", i, GET_DURING(t3, t2));
        printf("  Checking the multiply of two matrix ...\n");
        for (size_t j = 0; j < nRow * nRow; ++j) {
            if (gpu_mul[j] != cpu_mul[j]) {
                LOG_ERROR("  Test Fail! Index %zu expect %d, but got %d\n", j, cpu_mul[j], gpu_mul[j]);
            }
        }
        printf("  Check the the multiply of two matrix OK, calculate the multiply Success\n");

        printf("Test matrix multiply on GPU device %d Success\n", i);
    }
    return 0;
}
