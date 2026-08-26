#include "dskr2k_cuda.h"

#include <cuda_runtime.h>

__global__ static void antisymmetrize_upper(double *c, const double *t, int n,
                                            double beta, long long stride_t,
                                            long long stride_c, int ld_t,
                                            int ld_c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;
  if (i > j || j >= n) return;
  double *cb = c + static_cast<long long>(b) * stride_c;
  const double *tb = t + static_cast<long long>(b) * stride_t;
  cb[i + static_cast<long long>(j) * ld_c] =
      i == j ? 0.0 : tb[i + static_cast<long long>(j) * ld_t]
                    - tb[j + static_cast<long long>(i) * ld_t]
                    + beta * cb[i + static_cast<long long>(j) * ld_c];
}

cublasStatus_t dskr2k_cuda_upper(cublasHandle_t handle, int n, int k,
                                  int batch, double alpha, const double *a,
                                  long long stride_a, const double *b,
                                  long long stride_b, double beta, double *c,
                                  long long stride_c, double *t) {
  const double zero = 0.0;
  cublasStatus_t status = cublasDgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, a, n, stride_a,
      b, n, stride_b, &zero, t, n, stride_c, batch);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  dim3 threads(16, 16);
  dim3 blocks((n + threads.x - 1) / threads.x,
              (n + threads.y - 1) / threads.y, batch);
  antisymmetrize_upper<<<blocks, threads>>>(c, t, n, beta, stride_c, stride_c,
                                            n, n);
  return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                            : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t dskr2k_cuda_upper_lda(cublasHandle_t handle, int n, int k,
                                     int batch, double alpha, const double *a,
                                     int lda, long long stride_a,
                                     const double *b, long long stride_b,
                                     double beta, double *c,
                                     long long stride_c, double *t) {
  const double zero = 0.0;
  const long long stride_t = static_cast<long long>(n) * n;
  cublasStatus_t status = cublasDgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, a, lda, stride_a,
      b, lda, stride_b, &zero, t, n, stride_t, batch);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  dim3 threads(16, 16);
  dim3 blocks((n + threads.x - 1) / threads.x,
              (n + threads.y - 1) / threads.y, batch);
  antisymmetrize_upper<<<blocks, threads>>>(c, t, n, beta, stride_t, stride_c,
                                            n, lda);
  return cudaGetLastError() == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                            : CUBLAS_STATUS_EXECUTION_FAILED;
}
