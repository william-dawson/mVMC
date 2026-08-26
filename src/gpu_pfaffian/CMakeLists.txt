# GPU Pfaffian integration library for A3 (recal PfM/InvM) and, once
# vmccal.c's per-sample loop is fissioned, B1 (CalculateMAll). Built
# whenever find_package(CUDAToolkit) succeeds (see top-level
# CMakeLists.txt) -- no project-specific ON/OFF flag; USE_GPU_PFAFFIAN
# there is a plain TRUE/FALSE derived from that detection.
#
# dsktrf_cuda.{h,cu} / dskr2k_cuda.{h,cu} / utuinv_cuda.{h,cu} are
# unmodified copies of the validated kernels from
# mini-apps/mvmc-pfaffian-mini in william-dawson/mvmc_deep -- keep them
# in sync with that repo's HANDOFF.md if either changes.
# gpu_calculateMAll.cu is the new glue: an assemble-gather kernel and a
# negate kernel (matrix.c's calculateMAll_child_real, ported), chaining
# those three into one C-callable CalculateMAll_real_gpu(eleIdx, qpStart,
# qpEnd) -- a drop-in GPU replacement for CalculateMAll_real, wired in
# today only at A3's call site (vmcmake_real.c). See HANDOFF.md and the
# mvmc-pfaffian-gpu-port skill for the integration plan.
set(SOURCES_gpu_pfaffian
    gpu_managed_alloc.cu
    dskr2k_cuda.cu
    dsktrf_cuda.cu
    utuinv_cuda.cu
    gpu_calculateMAll.cu
)

add_library(gpu_pfaffian STATIC ${SOURCES_gpu_pfaffian})
set_target_properties(gpu_pfaffian PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(gpu_pfaffian PUBLIC CUDA::cudart CUDA::cublas)
