cmake_minimum_required(VERSION 3.26)

project(cuda_project LANGUAGES CXX CUDA)

# 设置CUDA架构（根据你的GPU类型进行调整）
set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_BUILD_TYPE Debug)

find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_FOUND)
  message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
else()
  message(STATUS "CUDA not found")
endif()

message(STATUS "CUDAToolkit_VERSION=${CUDAToolkit_VERSION}")
message(STATUS "CUDAToolkit_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}")
message(STATUS "CUDAToolkit_VERSION_MINOR=${CUDAToolkit_VERSION_MINOR}")
message(STATUS "CUDAToolkit_VERSION_PATCH=${CUDAToolkit_VERSION_PATCH}")
message(STATUS "CUDAToolkit_BIN_DIR=${CUDAToolkit_BIN_DIR}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDAToolkit_LIBRARY_ROOT=${CUDAToolkit_LIBRARY_ROOT}")
message(STATUS "CUDAToolkit_TARGET_DIR=${CUDAToolkit_TARGET_DIR}")
message(STATUS "CUDAToolkit_NVCC_EXECUTABLE=${CUDAToolkit_NVCC_EXECUTABLE}")

# 添加CUDA源文件
file(GLOB CUDA_SOURCES *.cu)

message(STATUS "CUDA_SOURCES=${CUDA_SOURCES}")

foreach(file ${CUDA_SOURCES})
  get_filename_component(file_name ${file} NAME_WE)
  set(target_name ${file_name})
  add_executable(${target_name} ${file})

  # 链接CUDA库和头文件
  target_link_libraries(${target_name} CUDA::cuda_driver)

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
  endif()
endforeach()
