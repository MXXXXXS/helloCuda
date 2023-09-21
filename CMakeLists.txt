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

function(add_sources NAME_PATTERN)
  # 添加CUDA源文件
  file(GLOB_RECURSE SOURCE_NAMES CONFIGURE_DEPENDS ${NAME_PATTERN})

  # message(STATUS "SOURCE_NAMES=${SOURCE_NAMES}")
  foreach(file ${SOURCE_NAMES})
    get_filename_component(file_name ${file} NAME_WE)
    get_filename_component(file_ext ${file} LAST_EXT)
    set(target_name ${file_name})
    add_executable(${target_name} ${file})

    if(file_ext MATCHES "\\.cu$")
      # 链接CUDA库和头文件
      target_link_libraries(${target_name} CUDA::cuda_driver)
      target_link_libraries(${target_name} GL GLU glut)

      if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
      endif()
    else()
      target_link_libraries(${target_name} stdc++)
    endif()
  endforeach()
endfunction()

add_sources(src/*.cu)
add_sources(src/*.cpp)

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

message(STATUS "CMAKE_SYSTEM_PREFIX_PATH: ${CMAKE_SYSTEM_PREFIX_PATH}")
