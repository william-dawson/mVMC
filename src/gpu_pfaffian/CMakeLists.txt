# GPU Pfaffian integration library for B1 (CalculateMAll) / A3 (recal
# PfM/InvM). Only built when USE_GPU_PFAFFIAN=ON (see top-level
# CMakeLists.txt). Today this holds only the CUDA Unified Memory
# allocation helper that setmemory.c calls under -DUSE_GPU_PFAFFIAN --
# the ported dsktrf/utu2pfa/utu2inv kernels (validated standalone in
# mini-apps/mvmc-pfaffian-mini) land here incrementally. See HANDOFF.md
# and the mvmc-pfaffian-gpu-port skill for the integration plan.
set(SOURCES_gpu_pfaffian
    gpu_managed_alloc.cu
)

add_library(gpu_pfaffian STATIC ${SOURCES_gpu_pfaffian})
set_target_properties(gpu_pfaffian PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(gpu_pfaffian PUBLIC CUDA::cudart)
