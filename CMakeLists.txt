cmake_minimum_required(VERSION 3.19)

if (NOT DEFINED WASI_SDK_PREFIX)
    set(WASI_SDK_PREFIX /opt/wasi-sdk)
endif ()
if (NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT /usr/local/cuda)
endif ()

message("CUDAToolkit path = ${CUDAToolkit_ROOT}")
message("WASI-SDK path=${WASI_SDK_PREFIX}")

# set wasi-sdk toolchain
set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)

project(wasm-cuda-simple-examples)


include_directories(${CUDAToolkit_ROOT}/include)
include_directories(include)

add_compile_options(-fno-exceptions -O3 -Wextra -Werror)
add_link_options(-Wl,--allow-undefined,--export=main,--strip-all,--no-entry)


add_executable(device.wasm src/device.c)
add_executable(mem.wasm src/mem.c)
add_executable(sumArray.wasm src/sumArray.c)
add_executable(mulMatrix.wasm src/mulMatrix.c)
