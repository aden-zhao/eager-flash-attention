cmake_minimum_required(VERSION 3.18)
project(tile_comm LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(tile_comm tile_comm.cu)

set_target_properties(tile_comm PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 80
)

target_link_libraries(tile_comm
    PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    CUDA::cublas
)

# Enable this later if nvToolsExt is available and you want NVTX ranges.
# target_link_libraries(tile_comm PRIVATE CUDA::nvToolsExt)
# target_compile_definitions(tile_comm PRIVATE USE_NVTX=1)