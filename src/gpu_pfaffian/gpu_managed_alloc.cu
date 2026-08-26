#include "gpu_managed_alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" void *gpu_managed_alloc(size_t nbytes) {
  void *ptr = NULL;
  cudaError_t err = cudaMallocManaged(&ptr, nbytes, cudaMemAttachGlobal);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "gpu_managed_alloc: cudaMallocManaged(%zu bytes) failed: %s\n",
            nbytes, cudaGetErrorString(err));
    exit(1);
  }
  return ptr;
}

extern "C" void gpu_managed_free(void *ptr) {
  if (ptr != NULL) {
    cudaFree(ptr);
  }
}
