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

extern "C" {
  extern int Nsite;
  extern int Nsite2;
  extern int Ne;
  extern int Nsize;
  extern double *SlaterElm_real;
  extern double *InvM_real;
  extern double *PfM_real;
}

#define A2_TPB 256

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
  /* Synchronous: the caller (vmcmake_real.c) reads PfM_real/InvM_real
   * again almost immediately (CalculateLogIP_real on the next trial, or
   * SlaterElmDiff_fcmp in VMCMainCal), so there is nothing to overlap
   * with in this single-walker integration -- unlike the mini-app's
   * concurrent-walker/async-stream variant, which this is not (yet). */
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    fprintf(stderr, "UpdateMAll_real_gpu: sync failed: %s\n", cudaGetErrorString(e));
    exit(1);
  }
}
