#ifndef GPU_MANAGED_ALLOC_H
#define GPU_MANAGED_ALLOC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate nbytes of CUDA Unified Memory (cudaMallocManaged), visible to
 * both host and device with no explicit copy. Aborts on failure -- this
 * mirrors mVMC's own malloc() call sites, none of which check for NULL
 * either. Used by setmemory.c under -DUSE_GPU_PFAFFIAN so the buffers the
 * GPU Pfaffian kernels touch (EleIdx, SlaterElm_real, InvM_real/PfM_real)
 * are resident wherever they're last used, with no host/device mirror
 * and no enclosing OpenACC data region. See HANDOFF.md / the
 * mvmc-pfaffian-gpu-port skill for the integration plan this is step 2
 * of: CMake scaffolding, then this malloc swap (verify nothing breaks
 * against a CPU-identical run), then the kernel dispatch itself. */
void *gpu_managed_alloc(size_t nbytes);

/* Counterpart to gpu_managed_alloc; a NULL-safe cudaFree. */
void gpu_managed_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* GPU_MANAGED_ALLOC_H */
