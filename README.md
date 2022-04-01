# CUDA Simple Examples for WebAssembly

This project is some simple **cuda programs** (use cuda driver API) that can be compiled into **WebAssembly** and run on
a wasm runtime that supports cuda access, such as **wasmer-gpu**.

### The current examples

1. device.cpp: some simple device query operations, output device information.
2. mem.cpp: some simple memory management operations, such as alloc, free, copy, etc. of device memory.
3. sumArray.cpp: Use cuda to perform the sum operation of a one-dimensional array.
4. mulMatrix.cpp: Use cuda to perform the multiplication of two-dimensional matrices.

### requirements

* wasi-sdk
* cmake *(and build tools such as make, ninja, etc.)*
* cuda toolkit (only use the header file ```cuda.h```, if your environment has no gpu, you can also copy a header file
  to compile)

### running results
