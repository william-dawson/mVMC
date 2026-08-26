/* Batched GPU panel factorization for PFAPACK's DSKTRF (UPLO='U',
 * MODE='N'), the exact path real mVMC drives.
 *
 * This is a direct transliteration of dlasktrf.f's upper-triangle,
 * STEP=1 branch, plus its "IF (N-NPANEL+1.GT.2)" trailing prep, batched
 * over matrices (one CUDA block per matrix). The one deliberate deviation
 * from dsktrf.f: every block -- including the final, sub-panel-width one
 * that dsktrf.f hands to the unblocked DSKTF2 -- goes through this same
 * panel kernel, called with nb_block = the remaining size. DSKTF2 and
 * DLASKTRF compute the identical Parlett-Reid factorization via a
 * differently-ordered sequence of equivalent rank updates (immediate
 * rank-2 SKR2 vs. delayed WY-style accumulation); substituting one for
 * the other changes the order floating-point roundoff accumulates in but
 * not the mathematical result, exactly as blocked and unblocked LAPACK
 * routines are already interchangeable. This lets one kernel cover both
 * of dsktrf.f's branches.
 *
 * A second deviation folds dsktrf.f's own post-block "missing row
 * interchange" fixup (applied by the outer DSKTRF loop to columns
 * beyond the active block) into the panel kernel's pivot swap directly,
 * by using the *total* matrix width for that one swap's span instead of
 * the active block's width -- the two swaps touch disjoint column
 * ranges for the same row pair, so merging them is exact. See
 * HANDOFF.md for the full derivation of both equivalences.
 */
#include "dsktrf_cuda.h"
#include "dskr2k_cuda.h"

#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_THREADS 256

/* 1-indexed accessor into a total_n-by-total_n (or total_n-by-NB_alloc)
 * column-major buffer, matching PFAPACK's own 1-indexed A(i,j) notation
 * so this kernel can be checked line-by-line against dlasktrf.f. */
#define IA(i, j) A[(long long)((i) - 1) + (long long)((j) - 1) * total_n]
#define IW(i, j) W[(long long)((i) - 1) + (long long)((j) - 1) * total_n]

__global__ static void dsktrf_init_kernel(int total_n, int batch,
                                          int *ipiv_batch, int *info_batch) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= batch) return;
  ipiv_batch[(long long)q * total_n + (total_n - 1)] = total_n; /* IPIV(N)=N */
  info_batch[q] = 0;
}

/* One CUDA block factorizes one matrix's current panel (columns
 * M-NB+1..M of its active M-by-M leading submatrix), plus -- if a
 * leading submatrix >= 2x2 remains below it -- prepares the W panel
 * for the caller's trailing rank-2k update (dskr2k_cuda_upper_lda). */
__global__ static void dsktrf_panel_kernel(int total_n, int M, int NB,
                                           double *A_batch, double *W_batch,
                                           int NB_alloc, int *ipiv_batch,
                                           int *info_batch,
                                           double *saved_t_batch) {
  const int q = blockIdx.x;
  const long long mat_stride = (long long)total_n * total_n;
  const long long w_stride = (long long)total_n * NB_alloc;
  double *A = A_batch + q * mat_stride;
  double *W = W_batch + q * w_stride;
  int *ipiv = ipiv_batch + (long long)q * total_n;
  int *info = info_batch + q;

  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  __shared__ double v1[DSKTRF_CUDA_MAX_NB];
  __shared__ double v2[DSKTRF_CUDA_MAX_NB];
  __shared__ double sred_val[BLOCK_THREADS];
  __shared__ int sred_idx[BLOCK_THREADS];
  __shared__ int s_kp;
  __shared__ double s_colmax;
  __shared__ double s_t;

  int WK = 0;
  const int NPANEL = NB; /* STEP=1 (mode='N') */
  int Kmin = M - NPANEL + 1;
  if (Kmin < 2) Kmin = 2;

  for (int K = M; K >= Kmin; --K) {
    const int KK = K - 1;

    if (K < M) {
      if (WK > 0) {
        if (tid == 0) IA(K, K) = 0.0;
        __syncthreads();
        if (tid < WK) {
          v1[tid] = IW(K, NB - WK + 1 + tid);
          v2[tid] = IA(K, M - WK + 1 + tid);
        }
        __syncthreads();
        for (int i = tid + 1; i <= K; i += nt) {
          double sum = IA(i, K);
          for (int l = 0; l < WK; ++l)
            sum += IA(i, M - WK + 1 + l) * v1[l] - IW(i, NB - WK + 1 + l) * v2[l];
          IA(i, K) = sum;
        }
        __syncthreads();
        if (tid == 0) IA(K, K) = 0.0;
        __syncthreads();
      }
      WK += 1;
      for (int i = tid + 1; i <= K; i += nt) IW(i, NB - WK + 1) = IA(i, K);
      __syncthreads();
    }

    /* pivot search: IDAMAX over A(1:K-1,K) */
    double best_val = -1.0;
    int best_idx = 1;
    for (int i = tid + 1; i <= K - 1; i += nt) {
      double v = fabs(IA(i, K));
      if (v > best_val) { best_val = v; best_idx = i; }
    }
    sred_val[tid] = best_val;
    sred_idx[tid] = best_idx;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
      if (tid < s && sred_val[tid + s] > sred_val[tid]) {
        sred_val[tid] = sred_val[tid + s];
        sred_idx[tid] = sred_idx[tid + s];
      }
      __syncthreads();
    }
    if (tid == 0) {
      int kp = sred_idx[0];
      double colmax = sred_val[0];
      if (colmax <= 0.0) {
        if (*info == 0) *info = K - 1;
        kp = KK;
      }
      s_kp = kp;
      s_colmax = colmax;
    }
    __syncthreads();
    const int KP = s_kp;
    const double COLMAX = s_colmax;

    if (KP != KK) {
      for (int i = tid + 1; i <= KP - 1; i += nt) {
        double t = IA(i, KK); IA(i, KK) = IA(i, KP); IA(i, KP) = t;
      }
      __syncthreads();
      for (int d = tid; d < KK - KP - 1; d += nt) {
        int i = KP + 1 + d;
        double t = IA(i, KK); IA(i, KK) = IA(KP, i); IA(KP, i) = t;
      }
      __syncthreads();
      /* total_n (not the active block width M) folds in dsktrf.f's
       * deferred trailing-column pivot fixup -- see file header. */
      for (int d = tid; d < total_n - K + 1; d += nt) {
        int j = K + d;
        double t = IA(KK, j); IA(KK, j) = IA(KP, j); IA(KP, j) = t;
      }
      __syncthreads();
      for (int d = tid; d < KK - KP; d += nt) {
        int i = KP + d;
        IA(i, KK) *= -1.0;
      }
      __syncthreads();
      for (int d = tid; d < KK - KP - 1; d += nt) {
        int j = KP + 1 + d;
        IA(KP, j) *= -1.0;
      }
      __syncthreads();
      if (WK > 0) {
        for (int d = tid; d < WK; d += nt) {
          int col = NB - WK + 1 + d;
          double t = IW(KK, col); IW(KK, col) = IW(KP, col); IW(KP, col) = t;
        }
        __syncthreads();
      }
    }

    if (COLMAX > 0.0) {
      double inv = 1.0 / IA(KK, K);
      for (int i = tid + 1; i <= K - 2; i += nt) IA(i, K) *= inv;
      __syncthreads();
    }

    if (tid == 0) ipiv[KK - 1] = KP;
    __syncthreads();
  }

  /* Trailing prep: only when a >=2x2 leading submatrix remains. Fills the
   * one W column the K-loop above never gets to (see file header), so
   * that W(1:size,1:NB) and A(1:size,M-NB+1:M) are ready for the
   * caller's dskr2k_cuda_upper_lda trailing rank-2k update. */
  const int size = M - NB;
  if (size >= 2) {
    if (tid == 0) { s_t = IA(size, size + 1); IA(size, size + 1) = 0.0; }
    __syncthreads();

    if (WK < NB) {
      for (int i = tid + 1; i <= size; i += nt) IW(i, 1) = IA(i, size);
      __syncthreads();
      if (tid == 0) IW(size, 1) = 0.0;
      __syncthreads();
      if (tid < WK) {
        v1[tid] = IW(size, NB - WK + 1 + tid);
        v2[tid] = IA(size, M - WK + 1 + tid);
      }
      __syncthreads();
      for (int i = tid + 1; i <= size; i += nt) {
        double sum = IW(i, 1);
        for (int l = 0; l < WK; ++l)
          sum += IA(i, M - WK + 1 + l) * v1[l] - IW(i, NB - WK + 1 + l) * v2[l];
        IW(i, 1) = sum;
      }
      __syncthreads();
      if (tid == 0) IW(size, 1) = 0.0;
      __syncthreads();
      WK += 1;
    }

    /* Leave A(size,size+1) zeroed -- PFAPACK's own dlasktrf.f keeps it
     * zero for the duration of the trailing DSKR2K call that follows and
     * restores it only afterward. The restore happens in a separate
     * kernel (dsktrf_restore_kernel) launched by the host driver after
     * its trailing dskr2k_cuda_upper_lda call, not here: this panel
     * kernel and that GEMM are two separate launches, so restoring
     * before the GEMM runs would feed it the wrong value. */
    if (tid == 0) saved_t_batch[q] = s_t;
    __syncthreads();
  }
}

/* Restores the one A(size,size+1) entry dsktrf_panel_kernel zeroed for
 * its trailing update, once that update (dskr2k_cuda_upper_lda) has
 * actually consumed the zeroed value. One thread per batch matrix. */
__global__ static void dsktrf_restore_kernel(int total_n, int size,
                                             double *A_batch,
                                             const double *saved_t_batch,
                                             int batch) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= batch) return;
  double *A = A_batch + (long long)q * total_n * total_n;
  A[(size - 1) + (long long)size * total_n] = saved_t_batch[q];
}

#undef IA
#undef IW

cublasStatus_t dsktrf_cuda_upper_n(cublasHandle_t handle, int n, int nb,
                                   int batch, double *a, int *ipiv, int *info,
                                   double *w, double *scratch) {
  /* The final (< 2*nb) block runs unblocked-equivalent at width up to
   * 2*nb-1 (see dsktrf_cuda.h); that must still fit the panel kernel's
   * fixed-size shared-memory basis vectors. */
  if (nb < 1 || 2 * nb - 1 > DSKTRF_CUDA_MAX_NB || n < 1 || batch < 1)
    return CUBLAS_STATUS_INVALID_VALUE;

  const int nb_alloc = 2 * nb; /* W's allocated panel width -- see dsktrf_cuda.h */

  const int init_threads = 128;
  const int init_blocks = (batch + init_threads - 1) / init_threads;
  dsktrf_init_kernel<<<init_blocks, init_threads>>>(n, batch, ipiv, info);
  if (cudaGetLastError() != cudaSuccess) return CUBLAS_STATUS_EXECUTION_FAILED;

  double *saved_t = nullptr;
  if (cudaMalloc(&saved_t, static_cast<size_t>(batch) * sizeof(double)) != cudaSuccess)
    return CUBLAS_STATUS_ALLOC_FAILED;

  const long long stride_a_mat = (long long)n * n;
  const long long stride_w_mat = (long long)n * nb_alloc;

  int K = n;
  while (K >= 1) {
    const int nb_block = (K >= 2 * nb) ? nb : K;

    dsktrf_panel_kernel<<<batch, BLOCK_THREADS>>>(n, K, nb_block, a, w,
                                                  nb_alloc, ipiv, info, saved_t);
    if (cudaGetLastError() != cudaSuccess) {
      cudaFree(saved_t);
      return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    const int size = K - nb_block;
    if (size >= 2) {
      /* A-panel: columns size+1..K (0-indexed column offset `size`).
       * W-panel: columns 1..nb_block (offset 0). C: leading size x size
       * block at offset 0. All three are slices of the same lda=n
       * resident matrix / workspace. A(size,size+1) is still zeroed
       * (dsktrf_panel_kernel left it that way) for this call, matching
       * dlasktrf.f -- restored right after by dsktrf_restore_kernel. */
      const double *a_panel = a + (long long)size * n;
      cublasStatus_t st = dskr2k_cuda_upper_lda(
          handle, size, nb_block, batch, 1.0, a_panel, n, stride_a_mat, w,
          stride_w_mat, 1.0, a, stride_a_mat, scratch);
      if (st != CUBLAS_STATUS_SUCCESS) { cudaFree(saved_t); return st; }

      const int restore_threads = 128;
      const int restore_blocks = (batch + restore_threads - 1) / restore_threads;
      dsktrf_restore_kernel<<<restore_blocks, restore_threads>>>(n, size, a,
                                                                  saved_t, batch);
      if (cudaGetLastError() != cudaSuccess) {
        cudaFree(saved_t);
        return CUBLAS_STATUS_EXECUTION_FAILED;
      }
    }
    K = size;
  }
  cudaFree(saved_t);
  return CUBLAS_STATUS_SUCCESS;
}
