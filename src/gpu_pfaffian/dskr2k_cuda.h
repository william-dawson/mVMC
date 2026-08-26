#ifndef DSKR2K_CUDA_H
#define DSKR2K_CUDA_H

#include <cublas_v2.h>

/* Device-resident upper-triangle skew rank-2k update. A and B are n-by-k
 * column-major panels; C and scratch T are n-by-n. All strides are in
 * doubles. The call is asynchronous on the handle's stream. */
cublasStatus_t dskr2k_cuda_upper(cublasHandle_t handle, int n, int k,
                                 int batch, double alpha, const double *a,
                                 long long stride_a, const double *b,
                                 long long stride_b, double beta, double *c,
                                 long long stride_c, double *t);

/* Same update, but A, B, and C are all submatrix blocks of a larger
 * lda-by-lda resident matrix (the panel factorization's use case: the A/B
 * panels and the C trailing block are slices of one n_total-by-n_total
 * working matrix, not standalone tightly-packed buffers). `t` is scratch,
 * tightly packed n-by-n (stride n*n). */
cublasStatus_t dskr2k_cuda_upper_lda(cublasHandle_t handle, int n, int k,
                                     int batch, double alpha, const double *a,
                                     int lda, long long stride_a,
                                     const double *b, long long stride_b,
                                     double beta, double *c,
                                     long long stride_c, double *t);

#endif
