#ifndef DSKTRF_CUDA_H
#define DSKTRF_CUDA_H

#include <cublas_v2.h>

/* Batched GPU factorization P*A*P^T = U*T*U^T (Parlett-Reid, upper
 * triangle, mode='N' full tridiagonalization) -- the exact path real
 * mVMC's calculateMAll_child_real drives through PFAPACK's dsktrf.
 *
 * This is a direct port of dsktrf.f + dlasktrf.f (UPLO='U', MODE='N',
 * STEP=1): the blocked-panel driver calls the same panel kernel for every
 * block, including the final (< 2*nb) one, since PFAPACK's own unblocked
 * DSKTF2 fallback is an equivalent-but-differently-ordered Parlett-Reid
 * step for that case -- see HANDOFF.md for the derivation.
 *
 * Layout: A is a batch of n-by-n column-major matrices (lda = n, one
 * matrix per stride n*n), upper triangle populated on entry (as PFAPACK
 * expects). On exit A holds T's off-diagonal (A(i,i+1)) and U's columns
 * (A(1:i-1,i+1)), matching dsktrf_'s own in-place convention exactly, so
 * the existing CPU utu2pfa_d / utu2inv_d can consume it unchanged.
 *
 * ipiv is a batch of n-length arrays holding PFAPACK's 1-indexed IPIV
 * convention (row/col i interchanged with IPIV(i), applied i=n..1).
 * info is a batch of per-matrix flags: 0 = ok, >0 = first zero pivot
 * column (1-indexed), matching dsktrf_'s INFO for UPLO='U'.
 *
 * w is scratch, batch of n-by-(2*nb) doubles. Every *blocked* call uses a
 * panel exactly nb wide, but n is rarely a multiple of nb: the final call
 * (when the remaining size K < 2*nb) factorizes the whole K-wide remainder
 * in one unblocked-equivalent panel call, and K can be as wide as 2*nb-1
 * -- so the W workspace must be sized for that, not for nb. nb must be
 * <= 128 (so 2*nb-1 <= DSKTRF_CUDA_MAX_NB) -- the panel kernel keeps the
 * WY-style basis vectors for a block in fixed-size shared memory.
 *
 * scratch is tightly-packed GEMM scratch for the trailing rank-2k update,
 * batch of (n-nb)-by-(n-nb) doubles (the largest trailing block needed,
 * from the very first block's update -- only genuinely-blocked calls,
 * whose width is always exactly nb, ever run the trailing update).
 *
 * All device pointers; the call is asynchronous on the handle's stream. */
#define DSKTRF_CUDA_MAX_NB 256

cublasStatus_t dsktrf_cuda_upper_n(cublasHandle_t handle, int n, int nb,
                                   int batch, double *a, int *ipiv,
                                   int *info, double *w, double *scratch);

#endif
