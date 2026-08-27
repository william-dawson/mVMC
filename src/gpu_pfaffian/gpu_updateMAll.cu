/* GPU dispatch for UpdateMAll_real's Q=8 batch -- A2, the rank-1 fast
 * Pfaffian update fired on every accepted hopping move (vmcmake_real.c,
 * "if (w > genrand_real2())"). Measured 43% of the whole SR step at
 * W=42 with B1/A3 still CPU (see the mvmc-gpu-porting skill); with B1/A3
 * now GPU-dispatched (see gpu_calculateMAll.cu), A2 is what's left --
 * confirmed empirically: UpdateMAll [63] was 568.1s of a 600.3s total
 * (94.6%) in a real W=42 run with only A2 still CPU-only.
 *
 * Kernels are an unmodified port of mini-apps/mvmc-updatem-mini's CUDA
 * variant (kernel_cuda.cu), already validated there to 93% of achievable
 * bandwidth at 24 concurrent walkers, ~47% for one walker (the case
 * here -- this repo runs one walker per rank, see HANDOFF.md and the
 * mvmc-gpu-porting skill's "Walker granularity" section). Adapted the
 * same way gpu_calculateMAll.cu adapted the Pfaffian kernels: operates
 * on mVMC's own global SlaterElm_real/InvM_real/PfM_real directly
 * (Unified Memory since setmemory.c's step-1 malloc swap) via extern "C"
 * declarations, instead of the mini-app's params_t struct.
 *
 * Same skew-symmetry rewrite that mattered most in the mini-app (3.4x by
 * itself): invM[j][i] = -invM[i][j] turns the transposed matvec into a
 * row-major one, giving every thread a unit-stride read instead of a
 * gather. */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

extern "C" {
  extern int Nsite;
  extern int Nsite2;
  extern int Ne;
  extern int Nsize;
  extern double *SlaterElm_real;
  extern double *InvM_real;
  extern double *PfM_real;
  /* CPU reference, for the MVMC_A1_CHECK validation path only. */
  void CalculateNewPfM2_real(const int ma, const int s, double *pfMNew_real,
                             const int *eleIdx, const int qpStart, const int qpEnd);
}

#define A2_TPB 256

static void check_cuda(cudaError_t e, const char *what);

/* x[q][j] = SlaterElm_real[q][rsa][ eleIdx[j] + (j/ne)*nsite ] */
__global__ void k_a2_gather(const double* __restrict__ slt,
                             const int* __restrict__ eleIdx, double* __restrict__ xg,
                             int nsize, int ne, int nsite, int rsa) {
  int q = blockIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nsize) return;
  const long nsite2 = 2L * nsite;
  xg[(long)q * nsize + j] =
      slt[(long)q * nsite2 * nsite2 + (long)rsa * nsite2
          + eleIdx[j] + (j / ne) * nsite];
}

/* vec1[q][i] = sum_j invM[q][i][j] * xg[q][j] (skew form, row-major) */
__global__ void k_a2_matvec(const double* __restrict__ invM,
                             const double* __restrict__ xg,
                             double* __restrict__ vec1, double* __restrict__ rowa,
                             double* __restrict__ ivq, double* __restrict__ PfM,
                             int nsize, int msa) {
  int q = blockIdx.y, i = blockIdx.x;
  if (i >= nsize) return;
  const long nn = (long)nsize * nsize;
  const double* __restrict__ row = invM + (long)q * nn + (long)i * nsize;
  const double* __restrict__ x   = xg + (long)q * nsize;

  double acc = 0.0, red0 = 0.0;
  for (int j = threadIdx.x; j < nsize; j += blockDim.x) acc += row[j] * x[j];

  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  __shared__ double warp[A2_TPB / 32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) warp[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = (lane < A2_TPB / 32) ? warp[lane] : 0.0;
    #pragma unroll
    for (int o = (A2_TPB/32) >> 1; o > 0; o >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) { red0 = acc; vec1[(long)q * nsize + i] = acc; }
  }
  __shared__ double red0s;
  if (threadIdx.x == 0) red0s = red0;
  __syncthreads();

  if (i == msa) {
    for (int j = threadIdx.x; j < nsize; j += blockDim.x)
      rowa[(long)q * nsize + j] = row[j];
    if (threadIdx.x == 0) {
      double t = red0s == 0.0 ? 1e-300 : red0s;
      PfM[q] *= -t;
      ivq[q]  = -1.0 / t;
    }
  }
}

/* invM[q][i][j] += v1i*v2j - v1j*v2i, row/col msa folded in. */
__global__ void k_a2_rank1(double* __restrict__ invM, const double* __restrict__ vec1,
                            const double* __restrict__ rowa, const double* __restrict__ ivq,
                            int nsize, int msa) {
  int q = blockIdx.y, i = blockIdx.x;
  if (i >= nsize) return;
  const long nn = (long)nsize * nsize;
  const double iv  = ivq[q];
  const double v1i = vec1[(long)q * nsize + i];
  const double v2i = rowa[(long)q * nsize + i] * iv;
  const bool is_a  = (i == msa);
  double* __restrict__ row = invM + (long)q * nn + (long)i * nsize;
  const double* __restrict__ v1 = vec1 + (long)q * nsize;
  const double* __restrict__ ra = rowa + (long)q * nsize;

  int nv = nsize >> 1;
  double2* row2 = reinterpret_cast<double2*>(row);
  const double2* v12 = reinterpret_cast<const double2*>(v1);
  const double2* ra2 = reinterpret_cast<const double2*>(ra);
  for (int j = threadIdx.x; j < nv; j += blockDim.x) {
    double2 r = row2[j], a = ra2[j], b = v12[j];
    double v2x = a.x * iv, v2y = a.y * iv;
    double dx = v1i * v2x - b.x * v2i;
    double dy = v1i * v2y - b.y * v2i;
    if (is_a) { dx += v2x; dy += v2y; }
    if (2*j     == msa) dx -= v2i;
    if (2*j + 1 == msa) dy -= v2i;
    r.x += dx; r.y += dy;
    row2[j] = r;
  }
}

static int g_a2_buf_nsize = -1, g_a2_buf_q = -1;
static double *g_a2_scratch = NULL;   /* vec1, rowa, xg, ivq packed: 4*Q*nsize */

static void gpu_a2_ensure_buffers(int nsize, int q) {
  if (nsize == g_a2_buf_nsize && q == g_a2_buf_q) return;
  if (g_a2_scratch) cudaFree(g_a2_scratch);
  cudaError_t e = cudaMalloc(&g_a2_scratch, sizeof(double) * 4L * q * nsize);
  if (e != cudaSuccess) {
    fprintf(stderr, "UpdateMAll_real_gpu: cudaMalloc scratch: %s\n", cudaGetErrorString(e));
    exit(1);
  }
  g_a2_buf_nsize = nsize;
  g_a2_buf_q = q;
}

/* Drop-in GPU replacement for UpdateMAll_real(ma, s, eleIdx, qpStart,
 * qpEnd), same signature, same effect (updates InvM_real/PfM_real for
 * qpidx in [qpStart, qpEnd) in place). No return value to check --
 * UpdateMAll_real itself has none either; unlike B1/A3 there's no
 * dsktrf-style pivot failure mode here, just arithmetic. */
extern "C" void UpdateMAll_real_gpu(int ma, int s, const int *eleIdx, int qpStart, int qpEnd) {
  const int nsize = Nsize;
  const int ne = Ne;
  const int nsite = Nsite;
  const int q = qpEnd - qpStart;
  const int msa = ma + s * ne;
  const int rsa = eleIdx[msa] + s * nsite;

  gpu_a2_ensure_buffers(nsize, q);
  double *vec1 = g_a2_scratch;
  double *rowa = g_a2_scratch + (long)q * nsize;
  double *xg   = g_a2_scratch + 2L * q * nsize;
  double *ivq  = g_a2_scratch + 3L * q * nsize;

  const double *sltE_base = SlaterElm_real + (long)qpStart * Nsite2 * Nsite2;
  double *invM_base = InvM_real;   /* qpStart offset like calculateMAll_child_real's */
  double *pfM_base  = PfM_real;

  dim3 gg((nsize + A2_TPB - 1) / A2_TPB, q), gr(nsize, q);
  k_a2_gather<<<gg, A2_TPB>>>(sltE_base, eleIdx, xg, nsize, ne, nsite, rsa);
  k_a2_matvec<<<gr, A2_TPB>>>(invM_base, xg, vec1, rowa, ivq, pfM_base, nsize, msa);
  k_a2_rank1 <<<gr, A2_TPB>>>(invM_base, vec1, rowa, ivq, nsize, msa);

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "UpdateMAll_real_gpu: launch failed: %s\n", cudaGetErrorString(e));
    exit(1);
  }
  /* Deliberately NOT synchronized. The earlier version called
   * cudaDeviceSynchronize() here on every accepted move, on the reasoning
   * that the host reads InvM_real/PfM_real again almost immediately. That
   * reasoning was the bug: what read them was A1 (CalculateNewPfM2), on
   * the host, every trial -- so the sync was not the cost, it was the
   * symptom of a residency mistake that dragged 199 MB across C2C per
   * trial and cost A2 5.3x its own mini-app bandwidth.
   *
   * With A1 on the GPU (CalculateNewPfM2_real_gpu, below), nothing on the
   * host touches InvM_real inside the trial loop, and every consumer of
   * this kernel's output is a later launch on the same default stream --
   * which orders them for free. The only remaining host read is the Q
   * doubles of pfMNew, whose cudaMemcpy in A1 provides the one sync point
   * per trial that is actually needed.
   *
   * ...that was the theory, and it is WRONG as currently written. Dropping
   * this sync makes the run non-deterministic: two invocations of the same
   * binary with the same seed diverge in accept rate (0.36236 vs 0.35831 at
   * W=42/S=20), i.e. a real data race, bisected by toggling exactly this
   * call. With the sync in place the chain is reproducible and identical to
   * the pre-A1 reference (acc=0.35947, energy matching to ~4e-16, which is
   * the known B1 batched-cuBLAS reduction noise).
   *
   * So the sync stays ON by default until the race is root-caused. It is
   * not free -- A2 is 2.87 s of a 10.47 s run at W=42/S=20 -- but the
   * async version is not correct, and a fast wrong answer is worthless.
   * Set MVMC_A2_NOSYNC=1 to reproduce the race while investigating.
   *
   * Prime suspects, none yet confirmed: cuBLAS in the A3/B1 path not
   * actually sharing the legacy default stream with these raw launches;
   * per-thread default streams under OpenMP; or a host read of
   * InvM_real/PfM_real that this audit missed. */
  if (!getenv("MVMC_A2_NOSYNC"))
    check_cuda(cudaDeviceSynchronize(), "A2 sync");
}

/* --- A1: CalculateNewPfM2 ---------------------------------------------
 *
 * Ported for RESIDENCY, not for FLOPs. A1 is 0.6% of the CPU step and its
 * arithmetic is trivial -- Q dot products of length Nsize. But it is called
 * on every *trial* (52,920 per step at W=42/S=20, vs 19,023 accepts for
 * A2), and as the last host-side reader of InvM_real in the inner loop it
 * was dragging the whole 199 MB array back across NVLink-C2C every trial.
 * Measured cost of that: A1 itself 266 us/trial for 113 KB of reads
 * (0.42 GB/s), and A2 running 5.3x slower than the identical kernel does
 * in mini-apps/mvmc-updatem-mini (464 vs 2440 GB/s). Together 84.4% of the
 * GPU run. See the mvmc-gpu-porting skill, sections 2 and 5.
 *
 * With this on the GPU, the only per-trial host<->device traffic is the Q
 * doubles of pfMNew that the Metropolis test actually needs. */

/* Local to this TU; gpu_calculateMAll.cu has its own (internal linkage). */
static void check_cuda(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "gpu_updateMAll: %s: %s\n", what, cudaGetErrorString(e));
    exit(1);
  }
}

/* One block per qpidx; block-wide reduction over nsize.
 * ratio = sum_j invM[qpidx][msa][j] * sltE_a[ eleIdx[j] + (j/ne)*nsite ]
 * pfMNew[qpidx] = -ratio * PfM[qpidx]   -- exactly CalculateNewPfM2_real. */
__global__ void k_a1_ratio(int n, int ne, int nsite, int nsite2,
                            const double* __restrict__ slaterElm,
                            const double* __restrict__ invM,
                            const double* __restrict__ pfM,
                            const int* __restrict__ eleIdx,
                            int msa, int rsa,
                            double* __restrict__ pfMNew) {
  const int qpidx = blockIdx.x;
  const double* __restrict__ sltE_a = slaterElm + (long)qpidx * nsite2 * nsite2
                                                + (long)rsa * nsite2;
  const double* __restrict__ invM_a = invM + (long)qpidx * n * n + (long)msa * n;

  double acc = 0.0;
  for (int msj = threadIdx.x; msj < n; msj += blockDim.x) {
    const int rsj = eleIdx[msj] + (msj / ne) * nsite;
    acc += invM_a[msj] * sltE_a[rsj];
  }

  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  __shared__ double warp[A2_TPB / 32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) warp[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = (lane < A2_TPB / 32) ? warp[lane] : 0.0;
    #pragma unroll
    for (int o = (A2_TPB / 32) >> 1; o > 0; o >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) pfMNew[qpidx] = -acc * pfM[qpidx];
  }
}

static double *g_a1_pfMNew = NULL;   /* device, q doubles */
static int g_a1_q = -1;

/* Drop-in GPU replacement for CalculateNewPfM2_real. pfMNew_real is a host
 * array (a stack array at vmcmake_real.c:55), so the Q results are copied
 * back -- 64 bytes at Q=8. That cudaMemcpy is synchronous on the default
 * stream, which also orders this call behind the previous trial's
 * UpdateMAll_real_gpu without needing a device-wide sync anywhere. */
extern "C" void CalculateNewPfM2_real_gpu(int ma, int s, double *pfMNew_real,
                                          const int *eleIdx, int qpStart, int qpEnd) {
  const int n = Nsize, ne = Ne, nsite = Nsite, nsite2 = Nsite2;
  const int q = qpEnd - qpStart;
  const int msa = ma + s * ne;
  const int rsa = eleIdx[msa] + s * nsite;

  if (q != g_a1_q) {
    if (g_a1_pfMNew) cudaFree(g_a1_pfMNew);
    check_cuda(cudaMalloc(&g_a1_pfMNew, sizeof(double) * (size_t)q), "cudaMalloc a1 pfMNew");
    g_a1_q = q;
  }

  k_a1_ratio<<<q, A2_TPB>>>(n, ne, nsite, nsite2,
                            SlaterElm_real + (long)qpStart * nsite2 * nsite2,
                            InvM_real, PfM_real, eleIdx, msa, rsa, g_a1_pfMNew);
  check_cuda(cudaGetLastError(), "k_a1_ratio launch");
  check_cuda(cudaMemcpy(pfMNew_real, g_a1_pfMNew, sizeof(double) * (size_t)q,
                        cudaMemcpyDeviceToHost), "memcpy pfMNew");

  /* MVMC_A1_CHECK=1: recompute on the CPU and report the worst relative
   * difference seen so far. This exists because putting A1 on the GPU
   * changes the summation order of the Metropolis ratio, which can flip a
   * marginal accept and send the Markov chain down a different (equally
   * valid) path -- observed as a ~2e-3 energy shift at W=42/S=20. That is
   * indistinguishable, from the energy alone, from a wrong kernel. This
   * check separates the two: a reordering shows ~1e-16..1e-14 here, a bug
   * shows something much larger. The CPU path deliberately touches
   * InvM_real from the host, so this is a debug mode, not a run mode. */
  if (getenv("MVMC_A1_CHECK")) {
    static double worst = 0.0;
    static long ncall = 0;
    double *ref = (double *)malloc(sizeof(double) * (size_t)q);
    double *gpu = (double *)malloc(sizeof(double) * (size_t)q);
    memcpy(gpu, pfMNew_real, sizeof(double) * (size_t)q);
    CalculateNewPfM2_real(ma, s, ref, eleIdx, qpStart, qpEnd);
    for (int i = 0; i < q; i++) {
      double d = fabs(ref[i] - gpu[i]);
      double m = fabs(ref[i]) > fabs(gpu[i]) ? fabs(ref[i]) : fabs(gpu[i]);
      double rel = (m > 0.0) ? d / m : d;
      if (rel > worst) worst = rel;
    }
    /* MVMC_A1_CHECK=2 keeps the CPU value, so the Markov chain follows the
     * exact CPU decision sequence. That isolates the two things a diverged
     * energy could mean: if the energy then matches a CPU-only run to
     * machine precision, every GPU kernel is correct and the divergence
     * seen with =1 is purely A1's reordering flipping marginal Metropolis
     * accepts. If it still differs, something is actually wrong. */
    if (atoi(getenv("MVMC_A1_CHECK")) != 2)
      memcpy(pfMNew_real, gpu, sizeof(double) * (size_t)q);  /* follow the GPU path */
    free(ref); free(gpu);
    if (++ncall % 5000 == 0)
      fprintf(stderr, "A1-CHECK calls=%ld worst_rel=%.3e\n", ncall, worst);
  }
}
