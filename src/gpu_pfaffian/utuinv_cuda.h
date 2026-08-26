#ifndef UTUINV_CUDA_H
#define UTUINV_CUDA_H

#include <cublas_v2.h>

/* Batched GPU port of ltl2inv's utu2pfa_d / utu2inv_d -- Pfaffian
 * extraction and inverse reconstruction from PFAPACK's U*T*U^T
 * factorization (dsktrf_cuda_upper_n's output). See HANDOFF.md. */

/* Pfaffian: pfa = prod_{i=0,2,4,...} A(i,i+1) * sign(pivots). Trivial
 * O(n) work per matrix -- one thread per batch matrix. n must be even
 * (mVMC's electron count always is). */
cublasStatus_t utu2pfa_cuda(int n, int batch, const double *a,
                            const int *ipiv, double *pfaff);

/* Inverse: overwrites `a` in place with PFAPACK's inv(P*A_orig*P^T)
 * reconstruction (mVMC then negates it for its InvM = -M^{-1}
 * convention -- not done here, matching utu2inv_d's own scope).
 *
 * This always takes ltl2inv's "full" TRMM path (a plain batched GEMM
 * against M, whose lower triangle is genuinely zero, rather than the
 * blocked trmmt+antisymmetrize alternative LAPACK's ILAENV block-size
 * heuristic would pick on CPU for these sizes) -- both compute the same
 * M^T*A product, just via different routes; see HANDOFF.md for why this
 * substitution is legitimate (the same reasoning as DSKR2K's GEMM
 * reformulation) and the validation that confirms it numerically.
 *
 * a: in/out, batch of n-by-n column-major matrices, lda=n, holding the
 *    dsktrf_cuda_upper_n factorization on entry.
 * ipiv: in, batch of n-length 1-indexed pivot arrays from dsktrf.
 * m, uinv, vt, out: scratch, sized n*n, (n-1)*(n-1), n, and n*n doubles
 *    per batch member respectively. */
cublasStatus_t utu2inv_cuda_upper(cublasHandle_t handle, int n, int batch,
                                  double *a, const int *ipiv, double *m,
                                  double *uinv, double *vt, double *out);

#endif
