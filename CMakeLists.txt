cmake_minimum_required(VERSION 3.10)
project(BreastCancerPrediction)
set(CMAKE_CXX_STANDARD 20)
set(TORCH_CUDA_ARCH_LIST "7.5")
set(CAFFE2_USE_CUDNN 9)
set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-linux/share/opencv4")
add_subdirectory(external/csv-parser)

set(CMAKE_CXX_STANDARD 20)



add_executable(BreastCancerPrediction models/NNfromScratch.cpp
        include/TrainingPipeline.hpp
        src/TrainingPipeline.cpp
        include/SimpleNN.hpp
        external/third_party/doctest.cpp
        external/third_party/doctest.hpp
        include/NNModel.hpp
        models/NNModels.cpp
        include/NNCustumDatasets.hpp
        models/NNCustumDatasets.cpp
)

target_include_directories(BreastCancerPrediction PRIVATE
        "${CMAKE_SOURCE_DIR}/external/libtorch/include"
        "${CMAKE_SOURCE_DIR}/external/libtorch/include/torch/csrc/api/include"

)
target_link_libraries(BreastCancerPrediction PRIVATE

        -Wl,--no-as-needed
        -Wl,--whole-archive
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch_cuda.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch_cpu.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch.so"
        -Wl,--no-whole-archive

        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libc10_cuda.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libc10.so"

        # 1. CUDA Runtime (from your toolkit)
        "/usr/local/cuda/lib64/libcudart.so"

        # 2. CUDA Driver (The one usually missing - found in system libs)
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1"

        # 3. NVRTC (Compiler library)
        "/usr/local/cuda/lib64/libnvrtc.so"
        csv


        pthread
        dl
        m
)


target_compile_options(BreastCancerPrediction PRIVATE -O3 -Wall)
set_target_properties(BreastCancerPrediction PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/external/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
)