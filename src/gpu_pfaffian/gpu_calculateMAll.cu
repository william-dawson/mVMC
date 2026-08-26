/* GPU dispatch for calculateMAll_child_real's Q=8 batch -- wired in for
 * A3's "recal PfM/InvM" call site only (vmcmake_real.c, `if (nAccept >
 * Nsite)`), not B1's per-sample call site in vmccal.c, which still needs
 * loop fission before GPU batching pays off (see HANDOFF.md / the
 * mvmc-pfaffian-gpu-port skill).
 *
 * Chains the validated CUDA kernels, originally proven out standalone in
 * mini-apps/mvmc-pfaffian-mini (see that repo's HANDOFF.md): assemble-
 * gather (new, this file) -> dsktrf_cuda_upper_n -> utu2pfa_cuda ->
 * utu2inv_cuda_upper -> negate (new, this file).
 *
 * Operates directly on mVMC's own global arrays (SlaterElm_real,
 * InvM_real, PfM_real -- CUDA Unified Memory since setmemory.c's malloc
 * swap under USE_GPU_PFAFFIAN), so there is no host<->device copy of
 * walker state here: this function's own job is orchestration plus two
 * small new kernels. Scratch buffers (ipiv/info/w/scratch/m/uinv/vt/out)
 * are plain device memory, allocated and freed each call -- buffer reuse
 * across calls is deferred (see HANDOFF.md's "Not yet ported" list);
 * this step is about correctness, not yet performance. */

#include "dsktrf_cuda.h"
#include "utuinv_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/* mVMC's own globals (defined once, in the single vmcmain.c translation
 * unit that #includes every .c file in this codebase -- see global.h).
 * Declared extern "C" here so these C++-compiled references link against
 * the C-compiled definitions with no name mangling. */
extern "C" {
  extern int Nsite;
  extern int Nsite2;
  extern int Ne;
  extern int Nsize;
  extern double *SlaterElm_real;
  extern double *InvM_real;
  extern double *PfM_real;
}

/* Gather: a[b][msi][msj] (col-major, lda=n) = -slaterElm[b][rsi][rsj],
 * rsi = eleIdx[msi] + (msi/ne)*nsite, rsj = eleIdx[msj] + (msj/ne)*nsite --
 * exactly matrix.c's calculateMAll_child_real assemble loop, batched. One
 * eleIdx (a single sample's electron configuration) is shared across the
 * whole batch: A3 recomputes all Q projections of the same sample. */
__global__ void k_gather_slater_real(int n, int ne, int nsite, int nsite2, int batch,
                                      const double *slaterElm, const int *eleIdx,
                                      double *a) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)batch * n * n;
  if (idx >= total) return;
  long b = idx / ((long)n * n);
  long rem = idx % ((long)n * n);
  int msi = (int)(rem / n);   /* column */
  int msj = (int)(rem % n);   /* row */
  int rsi = eleIdx[msi] + (msi / ne) * nsite;
  int rsj = eleIdx[msj] + (msj / ne) * nsite;
  const double *sltE = slaterElm + b * (long)nsite2 * nsite2;
  a[b * (long)n * n + (long)msi * n + msj] = -sltE[(long)rsi * nsite2 + rsj];
}

__global__ void k_negate(long n, double *a) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) a[idx] = -a[idx];
}

static cublasHandle_t g_handle;
static int g_handle_ready = 0;

static cublasHandle_t gpu_pfaffian_handle() {
  if (!g_handle_ready) {
    cublasStatus_t st = cublasCreate(&g_handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "CalculateMAll_real_gpu: cublasCreate failed (status %d)\n", (int)st);
      exit(1);
    }
    g_handle_ready = 1;
  }
  return g_handle;
}

static void check_cuda(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CalculateMAll_real_gpu: %s: %s\n", what, cudaGetErrorString(e));
    exit(1);
  }
}
static void check_cublas(cublasStatus_t s, const char *what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CalculateMAll_real_gpu: %s failed (status %d)\n", what, (int)s);
    exit(1);
  }
}

/* Scratch buffers, cached across calls. Nsize and the batch width
 * (always NQPFull today) are both constant for the lifetime of one
 * mVMC run, so there is nothing to gain from the malloc/free-per-call
 * pattern step 2 shipped with -- it just paid 8 cudaMalloc + 8 cudaFree
 * calls on every single A3 firing (dozens+ per SR step) for no reason.
 * Allocate once, on first use, and keep the buffers for the process's
 * lifetime (CUDA reclaims them at exit). If the shape genuinely changes
 * (it shouldn't, mid-run) the buffers are freed and resized rather than
 * silently reused at the wrong size.
 *
 * This is call-to-call reuse, not the *within-one-call* aliasing HANDOFF.md
 * describes for B1's much larger batch (e.g. dsktrf's trailing-update
 * scratch becoming utu2inv's m/out buffer) -- that only matters once B1's
 * loop fission makes peak memory the binding constraint; at A3's Q=8
 * scale the whole footprint here is a few hundred MiB even unaliased, so
 * the allocation *count*, not size, was the actual cost. */
static int g_buf_n = -1, g_buf_nb = -1, g_buf_batch = -1;
static int *g_ipiv = NULL, *g_info = NULL;
static double *g_w = NULL, *g_scratch = NULL;
static double *g_m = NULL, *g_uinv = NULL, *g_vt = NULL, *g_out = NULL;

static void gpu_pfaffian_free_buffers() {
  if (g_ipiv) cudaFree(g_ipiv);
  if (g_info) cudaFree(g_info);
  if (g_w) cudaFree(g_w);
  if (g_scratch) cudaFree(g_scratch);
  if (g_m) cudaFree(g_m);
  if (g_uinv) cudaFree(g_uinv);
  if (g_vt) cudaFree(g_vt);
  if (g_out) cudaFree(g_out);
  g_ipiv = g_info = NULL;
  g_w = g_scratch = g_m = g_uinv = g_vt = g_out = NULL;
}

static void gpu_pfaffian_ensure_buffers(int n, int nb, int batch) {
  if (n == g_buf_n && nb == g_buf_nb && batch == g_buf_batch) return;
  gpu_pfaffian_free_buffers();
  check_cuda(cudaMalloc(&g_ipiv, sizeof(int) * (size_t)n * batch), "cudaMalloc ipiv");
  check_cuda(cudaMalloc(&g_info, sizeof(int) * (size_t)batch), "cudaMalloc info");
  check_cuda(cudaMalloc(&g_w, sizeof(double) * (size_t)n * (2 * nb) * batch), "cudaMalloc w");
  check_cuda(cudaMalloc(&g_scratch, sizeof(double) * (size_t)(n - nb) * (n - nb) * batch), "cudaMalloc scratch");
  check_cuda(cudaMalloc(&g_m, sizeof(double) * (size_t)n * n * batch), "cudaMalloc m");
  check_cuda(cudaMalloc(&g_uinv, sizeof(double) * (size_t)(n - 1) * (n - 1) * batch), "cudaMalloc uinv");
  check_cuda(cudaMalloc(&g_vt, sizeof(double) * (size_t)n * batch), "cudaMalloc vt");
  check_cuda(cudaMalloc(&g_out, sizeof(double) * (size_t)n * n * batch), "cudaMalloc out");
  g_buf_n = n;
  g_buf_nb = nb;
  g_buf_batch = batch;
}

/* Drop-in GPU replacement for CalculateMAll_real(eleIdx, qpStart, qpEnd),
 * same signature and same return convention (0 = ok; DSKTRF's raw info,
 * or qpidx+1 for a non-finite Pfaffian, matching
 * calculateMAll_child_real's two failure paths). Call only where the full
 * qpStart..qpEnd range is meant to run as one GPU batch -- true for every
 * real call site today (always Q=NQPFull=8 wide), but this function does
 * not itself split large ranges into memory-safe chunks. */
extern "C" int CalculateMAll_real_gpu(const int *eleIdx, int qpStart, int qpEnd) {
  const int n = Nsize;
  const int ne = Ne;
  const int nsite = Nsite;
  const int nsite2 = Nsite2;
  const int batch = qpEnd - qpStart;
  int nb = (n / 2 < 32) ? (n / 2) : 32;
  if (nb < 1) nb = 1;

  cublasHandle_t handle = gpu_pfaffian_handle();

  const double *sltE_base = SlaterElm_real + (long)qpStart * nsite2 * nsite2;
  double *invM = InvM_real;      /* batch always written starting at 0, like CPU's local qpidx */
  double *pfM = PfM_real;

  gpu_pfaffian_ensure_buffers(n, nb, batch);
  int *ipiv = g_ipiv, *info = g_info;
  double *w = g_w, *scratch = g_scratch, *m = g_m, *uinv = g_uinv, *vt = g_vt, *out = g_out;

  const int threads = 256;
  long total = (long)batch * n * n;
  int blocks = (int)((total + threads - 1) / threads);
  k_gather_slater_real<<<blocks, threads>>>(n, ne, nsite, nsite2, batch, sltE_base, eleIdx, invM);
  check_cuda(cudaGetLastError(), "k_gather_slater_real launch");

  check_cublas(dsktrf_cuda_upper_n(handle, n, nb, batch, invM, ipiv, info, w, scratch), "dsktrf_cuda_upper_n");
  check_cublas(utu2pfa_cuda(n, batch, invM, ipiv, pfM), "utu2pfa_cuda");
  check_cublas(utu2inv_cuda_upper(handle, n, batch, invM, ipiv, m, uinv, vt, out), "utu2inv_cuda_upper");

  blocks = (int)((total + threads - 1) / threads);
  k_negate<<<blocks, threads>>>(total, invM);
  check_cuda(cudaGetLastError(), "k_negate launch");
  check_cuda(cudaDeviceSynchronize(), "sync");

  /* Match calculateMAll_child_real's failure signalling: dsktrf's info
   * (>0 = first zero pivot column) or a non-finite Pfaffian both count as
   * failure. info is device-only scratch, so read it back; pfM is
   * PfM_real itself (Unified Memory), directly host-readable now that
   * the stream is synchronized. */
  int *h_info = (int *)malloc(sizeof(int) * (size_t)batch);
  check_cuda(cudaMemcpy(h_info, info, sizeof(int) * (size_t)batch, cudaMemcpyDeviceToHost), "memcpy info");
  int ret = 0;
  for (int b = 0; b < batch; b++) {
    if (h_info[b] != 0) { ret = h_info[b]; break; }
    if (!isfinite(pfM[b])) { ret = b + 1; break; }
  }
  free(h_info);

  return ret;
}

/* --- B1: multi-sample batched dispatch -------------------------------
 *
 * A3 above batches Q projections of ONE sample. B1's call site
 * (vmccal.c's VMCMainCal) instead calls CalculateMAll_real once per
 * sample, Q-wide, for every one of the S stored samples -- S*Q
 * independent factorizations, described since the very first version of
 * this port ("B1 batch=2000") but never actually exposed as one GPU
 * batch until now: the loop in vmccal.c calls CalculateMAll_real and
 * immediately consumes its result before moving to the next sample (see
 * HANDOFF.md's B1 finding). This section batches every sample's Q
 * projections into one GPU call; vmccal.c's loop then repoints
 * InvM_real/PfM_real at each sample's slice of the result instead of
 * computing it fresh. */

/* Gather for B1: batch index b = sampleIdx*q + qpidx. Each sample has
 * its OWN eleIdx (electron configuration); all samples share the SAME Q
 * SlaterElm_real slabs (SlaterElm_real is recomputed once per SR step,
 * not per sample -- see vmcmain.c's own comment to that effect). This
 * mirrors vmccal.c's VMCMainCal loop exactly: `eleIdx = EleIdx +
 * sample*Nsize; CalculateMAll_real(eleIdx, 0, NQPFull)`, batched across
 * every sample in the chunk instead of one sample at a time. */
__global__ void k_gather_slater_real_multisample(int n, int ne, int nsite, int nsite2,
                                                   int q, int nsamples,
                                                   const double *slaterElm,
                                                   const int *eleIdxBase, int sampleStride,
                                                   double *a) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)nsamples * q * n * n;
  if (idx >= total) return;
  long b = idx / ((long)n * n);
  long rem = idx % ((long)n * n);
  int msi = (int)(rem / n);
  int msj = (int)(rem % n);
  int sampleIdx = (int)(b / q);
  int qpidx = (int)(b % q);
  const int *eleIdx = eleIdxBase + (long)sampleIdx * sampleStride;
  int rsi = eleIdx[msi] + (msi / ne) * nsite;
  int rsj = eleIdx[msj] + (msj / ne) * nsite;
  const double *sltE = slaterElm + (long)qpidx * nsite2 * nsite2;
  a[b * (long)n * n + (long)msi * n + msj] = -sltE[(long)rsi * nsite2 + rsj];
}

/* Separate cache from A3's: B1's batch (nsamples*q) is a different size
 * class from A3's (q alone) -- sharing one cache would thrash on every
 * call as each site evicts the other's buffers. */
static int g_b1_buf_n = -1, g_b1_buf_nb = -1, g_b1_buf_batch = -1;
static int *g_b1_ipiv = NULL, *g_b1_info = NULL;
static double *g_b1_w = NULL, *g_b1_scratch = NULL;
static double *g_b1_m = NULL, *g_b1_uinv = NULL, *g_b1_vt = NULL, *g_b1_out = NULL;
/* Unified Memory (unlike the scratch buffers above): VMCMainCal's
 * per-sample consumption phase reads these directly as ordinary host
 * pointers (repointing InvM_real/PfM_real at a slice of them), so they
 * must be host-readable, not plain device memory. */
static double *g_b1_invM = NULL, *g_b1_pfM = NULL;
static int *g_b1_info_per_sample = NULL;
static int g_b1_nsamples_alloc = -1;

static void gpu_pfaffian_b1_free_buffers() {
  if (g_b1_ipiv) cudaFree(g_b1_ipiv);
  if (g_b1_info) cudaFree(g_b1_info);
  if (g_b1_w) cudaFree(g_b1_w);
  if (g_b1_scratch) cudaFree(g_b1_scratch);
  if (g_b1_m) cudaFree(g_b1_m);
  if (g_b1_uinv) cudaFree(g_b1_uinv);
  if (g_b1_vt) cudaFree(g_b1_vt);
  if (g_b1_out) cudaFree(g_b1_out);
  if (g_b1_invM) cudaFree(g_b1_invM);
  if (g_b1_pfM) cudaFree(g_b1_pfM);
  if (g_b1_info_per_sample) cudaFree(g_b1_info_per_sample);
  g_b1_ipiv = g_b1_info = NULL;
  g_b1_w = g_b1_scratch = g_b1_m = g_b1_uinv = g_b1_vt = g_b1_out = NULL;
  g_b1_invM = g_b1_pfM = NULL;
  g_b1_info_per_sample = NULL;
}

static void gpu_pfaffian_b1_ensure_buffers(int n, int nb, int batch, int nsamples) {
  if (n == g_b1_buf_n && nb == g_b1_buf_nb && batch == g_b1_buf_batch && nsamples == g_b1_nsamples_alloc) return;
  gpu_pfaffian_b1_free_buffers();
  check_cuda(cudaMalloc(&g_b1_ipiv, sizeof(int) * (size_t)n * batch), "cudaMalloc b1 ipiv");
  check_cuda(cudaMalloc(&g_b1_info, sizeof(int) * (size_t)batch), "cudaMalloc b1 info");
  check_cuda(cudaMalloc(&g_b1_w, sizeof(double) * (size_t)n * (2 * nb) * batch), "cudaMalloc b1 w");
  check_cuda(cudaMalloc(&g_b1_scratch, sizeof(double) * (size_t)(n - nb) * (n - nb) * batch), "cudaMalloc b1 scratch");
  check_cuda(cudaMalloc(&g_b1_m, sizeof(double) * (size_t)n * n * batch), "cudaMalloc b1 m");
  check_cuda(cudaMalloc(&g_b1_uinv, sizeof(double) * (size_t)(n - 1) * (n - 1) * batch), "cudaMalloc b1 uinv");
  check_cuda(cudaMalloc(&g_b1_vt, sizeof(double) * (size_t)n * batch), "cudaMalloc b1 vt");
  check_cuda(cudaMalloc(&g_b1_out, sizeof(double) * (size_t)n * n * batch), "cudaMalloc b1 out");
  check_cuda(cudaMallocManaged(&g_b1_invM, sizeof(double) * (size_t)n * n * batch), "cudaMallocManaged b1 invM");
  check_cuda(cudaMallocManaged(&g_b1_pfM, sizeof(double) * (size_t)batch), "cudaMallocManaged b1 pfM");
  check_cuda(cudaMallocManaged(&g_b1_info_per_sample, sizeof(int) * (size_t)nsamples), "cudaMallocManaged b1 info_per_sample");
  g_b1_buf_n = n;
  g_b1_buf_nb = nb;
  g_b1_buf_batch = batch;
  g_b1_nsamples_alloc = nsamples;
}

/* B1's batched GPU dispatch: factorizes every (sample, qpidx) pair in
 * [0, nsamples) x [0, q) in one GPU batch of nsamples*q matrices --
 * exactly the "S*Q independent factorizations" B1 has always described,
 * finally exposed instead of run one sample (Q-wide) at a time.
 *
 * eleIdxBase/sampleStride let the caller pass a sub-range of mVMC's own
 * EleIdx array directly (EleIdx + sampleStart*Nsize, Nsize) -- no gather
 * or copy of the input side needed, since EleIdx is already contiguous
 * across all samples (see HANDOFF.md).
 *
 * Does NOT chunk by memory -- nsamples*q matrices of n*n doubles (times
 * several buffers) must fit at once. Correct for the sizes validated so
 * far; HANDOFF.md's memory-ceiling section already establishes the real
 * chunk size needed at production S/W, not yet wired in here.
 *
 * On return, *invM_batch_out / *pfM_batch_out point at cached Unified
 * Memory holding every result (layout: batch index b = sampleIdx*q +
 * qpidx, the same convention CalculateMAll_real's own qpidx loop uses;
 * matrices and Pfaffians are two SEPARATE uniform-stride arrays, not
 * contiguous with each other the way the global InvM_real/PfM_real are
 * -- callers must not assume "PfM_real = InvM_real + Q*Nsize*Nsize"
 * still holds for these batch buffers). *info_per_sample_out points at
 * an nsamples-length array, one aggregated status per sample (0 = ok,
 * matching CalculateMAll_real's return convention) -- the caller can
 * keep looping samples exactly as before and check info per sample
 * unchanged. Return value is the first nonzero per-sample status found
 * (0 if every sample succeeded), for convenience/logging only -- the
 * per-sample array is what callers should actually branch on. */
extern "C" int CalculateMAll_real_gpu_batch(const int *eleIdxBase, int sampleStride,
                                             int nsamples, int q,
                                             double **invM_batch_out, double **pfM_batch_out,
                                             int **info_per_sample_out) {
  const int n = Nsize;
  const int ne = Ne;
  const int nsite = Nsite;
  const int nsite2 = Nsite2;
  const int batch = nsamples * q;
  int nb = (n / 2 < 32) ? (n / 2) : 32;
  if (nb < 1) nb = 1;

  cublasHandle_t handle = gpu_pfaffian_handle();

  gpu_pfaffian_b1_ensure_buffers(n, nb, batch, nsamples);
  int *ipiv = g_b1_ipiv, *info = g_b1_info;
  double *w = g_b1_w, *scratch = g_b1_scratch, *m = g_b1_m, *uinv = g_b1_uinv, *vt = g_b1_vt, *out = g_b1_out;
  double *invM = g_b1_invM;
  double *pfM = g_b1_pfM;

  const int threads = 256;
  long total = (long)batch * n * n;
  int blocks = (int)((total + threads - 1) / threads);
  k_gather_slater_real_multisample<<<blocks, threads>>>(n, ne, nsite, nsite2, q, nsamples,
                                                          SlaterElm_real, eleIdxBase, sampleStride, invM);
  check_cuda(cudaGetLastError(), "k_gather_slater_real_multisample launch");

  check_cublas(dsktrf_cuda_upper_n(handle, n, nb, batch, invM, ipiv, info, w, scratch), "dsktrf_cuda_upper_n (B1)");
  check_cublas(utu2pfa_cuda(n, batch, invM, ipiv, pfM), "utu2pfa_cuda (B1)");
  check_cublas(utu2inv_cuda_upper(handle, n, batch, invM, ipiv, m, uinv, vt, out), "utu2inv_cuda_upper (B1)");

  blocks = (int)((total + threads - 1) / threads);
  k_negate<<<blocks, threads>>>(total, invM);
  check_cuda(cudaGetLastError(), "k_negate launch (B1)");
  check_cuda(cudaDeviceSynchronize(), "sync (B1)");

  int *h_info = (int *)malloc(sizeof(int) * (size_t)batch);
  check_cuda(cudaMemcpy(h_info, info, sizeof(int) * (size_t)batch, cudaMemcpyDeviceToHost), "memcpy b1 info");
  int overall = 0;
  for (int s = 0; s < nsamples; s++) {
    int sample_info = 0;
    for (int qi = 0; qi < q; qi++) {
      int b = s * q + qi;
      if (h_info[b] != 0) { sample_info = h_info[b]; break; }
      if (!isfinite(pfM[b])) { sample_info = qi + 1; break; }
    }
    g_b1_info_per_sample[s] = sample_info;
    if (sample_info != 0 && overall == 0) overall = sample_info;
  }
  free(h_info);

  *invM_batch_out = invM;
  *pfM_batch_out = pfM;
  *info_per_sample_out = g_b1_info_per_sample;
  return overall;
}
