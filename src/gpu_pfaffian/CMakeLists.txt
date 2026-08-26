# GPU dispatch library for B1/A3 (CalculateMAll -- vmccal.c/vmcmake_real.c)
# and A2 (UpdateMAll -- vmcmake_real.c). Built whenever
# find_package(CUDAToolkit) succeeds (see top-level CMakeLists.txt) -- no
# project-specific ON/OFF flag; USE_GPU_PFAFFIAN there is a plain
# TRUE/FALSE derived from that detection.
#
# dsktrf_cuda.{h,cu} / dskr2k_cuda.{h,cu} / utuinv_cuda.{h,cu} are
# unmodified copies of the validated kernels from
# mini-apps/mvmc-pfaffian-mini in william-dawson/mvmc_deep -- keep them
# in sync with that repo's HANDOFF.md if either changes.
# gpu_calculateMAll.cu chains those three (plus a new gather and negate
# kernel) into CalculateMAll_real_gpu / CalculateMAll_real_gpu_batch --
# drop-in GPU replacements for CalculateMAll_real, wired in at A3's and
# B1's call sites respectively.
# gpu_updateMAll.cu is an adapted copy of mini-apps/mvmc-updatem-mini's
# CUDA kernels (gather/matvec/rank1), wired in as UpdateMAll_real_gpu at
# A2's call site. See HANDOFF.md and the mvmc-pfaffian-gpu-port skill for
# the integration plan.
set(SOURCES_gpu_pfaffian
    gpu_managed_alloc.cu
    dskr2k_cuda.cu
    dsktrf_cuda.cu
    utuinv_cuda.cu
    gpu_calculateMAll.cu
    gpu_updateMAll.cu
)

add_library(gpu_pfaffian STATIC ${SOURCES_gpu_pfaffian})
set_target_properties(gpu_pfaffian PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(gpu_pfaffian PUBLIC CUDA::cudart CUDA::cublas)
