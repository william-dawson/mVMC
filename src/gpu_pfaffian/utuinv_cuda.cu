/* Batched GPU port of ltl2inv's utu2pfa_d / utu2inv_d. Transliterated
 * directly from pfaffian.tcc / invert.tcc (0-indexed C++ already, unlike
 * PFAPACK's 1-indexed Fortran, so no index-shift layer is needed here).
 * See utuinv_cuda.h for the one deliberate algorithmic substitution
 * (always the "full" TRMM path, reformulated as a batched GEMM).
 */
#include "utuinv_cuda.h"

#include <cuda_runtime.h>

#define BLOCK_THREADS 256

/* 0-indexed accessor, matching pfaffian.tcc/invert.tcc's own colmaj. */
#define AT(buf, i, j, ld) (buf)[(long long)(i) + (long long)(j) * (ld)]

__global__ static void utu2pfa_kernel(int n, int batch, const double *a,
                                      const int *ipiv, double *pfaff) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= batch) return;
  const double *A = a + (long long)q * n * n;
  const int *ip = ipiv + (long long)q * n;
  double pfa = 1.0;
  for (int i = 0; i < n; i += 2) {
    pfa *= AT(A, i, i + 1, n);
    if (ip[i] - 1 != i) pfa = -pfa;
    if (ip[i + 1] - 1 != i + 1) pfa = -pfa;
  }
  pfaff[q] = pfa;
}

cublasStatus_t utu2pfa_cuda(int n, int batch, const double *a,
                            const int *ipiv, double *pfaff) {
  const int threads = 128;
  const int blocks = (batch + threads - 1) / threads;
  utu2pfa_kernel<<<blocks, threads>>>(n, batch, a, ipiv, pfaff);
  return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                            : CUBLAS_STATUS_EXECUTION_FAILED;
}

__global__ static void set_identity_kernel(double *buf, int dim, int batch,
                                           long long stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int q = blockIdx.z;
  if (i >= dim || j >= dim) return;
  AT(buf + (long long)q * stride, i, j, dim) = (i == j) ? 1.0 : 0.0;
}

/* Device pointer arrays for cublasDtrsmBatched: A's U-panel (the
 * (n-1)x(n-1) block at column offset 1, lda=n) and uinv (lda=n-1,
 * initialized to identity by the caller before this). */
__global__ static void build_trsm_ptrs_kernel(int n, int batch, double *a,
                                              double *uinv,
                                              double **ptrs_a,
                                              double **ptrs_b) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= batch) return;
  ptrs_a[q] = a + (long long)q * n * n + (long long)1 * n; /* A(0,1) */
  ptrs_b[q] = uinv + (long long)q * (n - 1) * (n - 1);
}

/* lacpy(UPPER, n-2, n-2, uinv(0,1), n-1, m(0,1), n): copy the upper
 * triangle of uinv's (n-2)x(n-2) block (offset row 0, col 1 within
 * uinv's own (n-1)-sized indexing) into m's block at the same offset
 * within m's n-sized indexing. */
__global__ static void lacpy_upper_kernel(int n, int batch, const double *uinv,
                                          double *m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; /* 0..n-3 */
  int j = blockIdx.y * blockDim.y + threadIdx.y; /* 0..n-3 */
  int q = blockIdx.z;
  const int dim = n - 2;
  if (i >= dim || j >= dim || i > j) return;
  const double *U = uinv + (long long)q * (n - 1) * (n - 1);
  double *M = m + (long long)q * n * n;
  AT(M, i, j + 1, n) = AT(U, i, j + 1, n - 1);
}

__global__ static void extract_vt_kernel(int n, int batch, const double *a,
                                         double *vt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; /* 0..n-2 */
  int q = blockIdx.y;
  if (i >= n - 1) return;
  const double *A = a + (long long)q * n * n;
  vt[(long long)q * n + i] = -AT(A, i, i + 1, n);
}

/* sktdsmx: solves the skew-tridiagonal system, one column (j) per
 * thread -- each column's forward/backward sweep is independent. */
__global__ static void sktdsmx_kernel(int n, int batch, const double *vt,
                                      const double *b, double *c) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int q = blockIdx.y;
  if (j >= n) return;
  const double *B = b + (long long)q * n * n;
  double *C = c + (long long)q * n * n;
  const double *VT = vt + (long long)q * n;

  AT(C, 1, j, n) = AT(B, 0, j, n) / -VT[0];
  for (int i = 2; i < n; i += 2)
    AT(C, i + 1, j, n) = (AT(B, i, j, n) - AT(C, i - 1, j, n) * VT[i - 1]) / -VT[i];

  AT(C, n - 2, j, n) = AT(B, n - 1, j, n) / VT[n - 2];
  for (int i = n - 3; i >= 0; i -= 2)
    AT(C, i - 1, j, n) = (AT(B, i, j, n) + AT(C, i + 1, j, n) * VT[i]) / VT[i - 1];
}

/* Sequential-in-j column permutation (LAPACK IPIV application order
 * matters -- see utuinv_cuda.h). One block per matrix. */
__global__ static void permute_columns_kernel(int n, int batch, double *a,
                                              const int *ipiv) {
  int q = blockIdx.x;
  double *A = a + (long long)q * n * n;
  const int *ip = ipiv + (long long)q * n;
  const int tid = threadIdx.x, nt = blockDim.x;

  for (int j = 0; j < n; ++j) {
    int kp = ip[j] - 1;
    if (kp != j) {
      for (int i = tid; i < n; i += nt) {
        double t = AT(A, i, j, n);
        AT(A, i, j, n) = AT(A, i, kp, n);
        AT(A, i, kp, n) = t;
      }
    }
    __syncthreads();
  }
}

__global__ static void permute_rows_kernel(int n, int batch, double *a,
                                           const int *ipiv) {
  int q = blockIdx.x;
  double *A = a + (long long)q * n * n;
  const int *ip = ipiv + (long long)q * n;
  const int tid = threadIdx.x, nt = blockDim.x;

  for (int i = 0; i < n; ++i) {
    int kp = ip[i] - 1;
    if (kp != i) {
      for (int j = tid; j < n; j += nt) {
        double t = AT(A, i, j, n);
        AT(A, i, j, n) = AT(A, kp, j, n);
        AT(A, kp, j, n) = t;
      }
    }
    __syncthreads();
  }
}

#undef AT

cublasStatus_t utu2inv_cuda_upper(cublasHandle_t handle, int n, int batch,
                                  double *a, const int *ipiv, double *m,
                                  double *uinv, double *vt, double *out) {
  if (n < 4 || batch < 1) return CUBLAS_STATUS_INVALID_VALUE;

  double **ptrs_a = nullptr, **ptrs_b = nullptr;
  if (cudaMalloc(&ptrs_a, static_cast<size_t>(batch) * sizeof(double *)) != cudaSuccess)
    return CUBLAS_STATUS_ALLOC_FAILED;
  if (cudaMalloc(&ptrs_b, static_cast<size_t>(batch) * sizeof(double *)) != cudaSuccess) {
    cudaFree(ptrs_a);
    return CUBLAS_STATUS_ALLOC_FAILED;
  }

  dim3 threads2(16, 16);

  /* M = identity(n). */
  dim3 blocks_m((n + 15) / 16, (n + 15) / 16, batch);
  set_identity_kernel<<<blocks_m, threads2>>>(m, n, batch, (long long)n * n);

  /* uinv = identity(n-1), then TRSM: U * uinv = uinv  =>  uinv = U^{-1}. */
  dim3 blocks_u((n - 1 + 15) / 16, (n - 1 + 15) / 16, batch);
  set_identity_kernel<<<blocks_u, threads2>>>(uinv, n - 1, batch,
                                              (long long)(n - 1) * (n - 1));

  const int ptr_threads = 128;
  const int ptr_blocks = (batch + ptr_threads - 1) / ptr_threads;
  build_trsm_ptrs_kernel<<<ptr_blocks, ptr_threads>>>(n, batch, a, uinv, ptrs_a, ptrs_b);
  if (cudaGetLastError() != cudaSuccess) {
    cudaFree(ptrs_a); cudaFree(ptrs_b);
    return CUBLAS_STATUS_EXECUTION_FAILED;
  }

  const double one = 1.0;
  cublasStatus_t st = cublasDtrsmBatched(
      handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
      CUBLAS_DIAG_UNIT, n - 1, n - 1, &one, ptrs_a, n, ptrs_b, n - 1, batch);
  cudaFree(ptrs_a);
  cudaFree(ptrs_b);
  if (st != CUBLAS_STATUS_SUCCESS) return st;

  /* lacpy the (n-2)x(n-2) upper triangle of uinv's own column-1 block
   * into M's column-1 block. */
  dim3 blocks_lacpy((n - 2 + 15) / 16, (n - 2 + 15) / 16, batch);
  lacpy_upper_kernel<<<blocks_lacpy, threads2>>>(n, batch, uinv, m);

  /* vT[i] = -A(i,i+1), i=0..n-2. */
  dim3 blocks_vt((n - 1 + 127) / 128, batch);
  extract_vt_kernel<<<blocks_vt, 128>>>(n, batch, a, vt);

  /* A := sktdsmx(vT, B=M) -- overwrites A in place, one thread per
   * (matrix, column). */
  dim3 blocks_skt((n + 127) / 128, batch);
  sktdsmx_kernel<<<blocks_skt, 128>>>(n, batch, vt, m, a);

  /* Column permutation (sequential in j, matching LAPACK IPIV order). */
  permute_columns_kernel<<<batch, BLOCK_THREADS>>>(n, batch, a, ipiv);

  /* A := M^T * A -- the "full" TRMM path reformulated as a plain batched
   * GEMM (M's lower triangle is genuinely zero in memory, so this is
   * numerically identical to a true TRMM, just 2x the FLOPs -- see
   * utuinv_cuda.h). Needs a separate output buffer (GEMM can't alias
   * input/output), copied back into `a` afterward. */
  const double zero = 0.0;
  st = cublasDgemmStridedBatched(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, m, n,
      (long long)n * n, a, n, (long long)n * n, &zero, out, n,
      (long long)n * n, batch);
  if (st != CUBLAS_STATUS_SUCCESS) return st;
  if (cudaMemcpyAsync(a, out, static_cast<size_t>(batch) * n * n * sizeof(double),
                      cudaMemcpyDeviceToDevice) != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  /* Row permutation. */
  permute_rows_kernel<<<batch, BLOCK_THREADS>>>(n, batch, a, ipiv);
  return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                            : CUBLAS_STATUS_EXECUTION_FAILED;
}
