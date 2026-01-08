cmake_minimum_required(VERSION 3.20)
project(CudaRayTracer LANGUAGES CXX C CUDA)

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)

set(SRC_PATH ${PROJECT_SOURCE_DIR}/Src)
set(EIGEN_PATH ${PROJECT_SOURCE_DIR}/ThirdParty/Eigen3)
set(IMGUI_PATH ${PROJECT_SOURCE_DIR}/ThirdParty/Imgui)
set(GLFW_PATH ${PROJECT_SOURCE_DIR}/ThirdParty/Glfw)
set(GLAD_PATH ${PROJECT_SOURCE_DIR}/ThirdParty/Glad)
set(TINY_OBJ_LOADER_PATH ${PROJECT_SOURCE_DIR}/ThirdParty/tinyobj)
set(STB_IMAGE_PATH ${PROJECT_SOURCE_DIR}/ThirdParty/stb_image)
set(JSON_PATH ${PROJECT_SOURCE_DIR}/ThirdParty/Json)
set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(
    ${EIGEN_PATH}/include/eigen3
    ${IMGUI_PATH}
    ${GLFW_PATH}/include
    ${TINY_OBJ_LOADER_PATH}
    ${STB_IMAGE_PATH}
    ${CUDA_PATH}/include
    ${JSON_PATH}
    Src/Utils
    Src/Core
    Src/Sampler
    Src/Camera
    Src/Material
    Src/Light
)

file(GLOB_RECURSE SRC ${SRC_PATH}/*.cpp ${SRC_PATH}/*.cu)
file(GLOB GLAD_SRC ${GLAD_PATH}/glad.c)
file(GLOB IMGUI_SRC ${IMGUI_PATH}/*.cpp)

add_executable(CudaRayTracer ${SRC} ${IMGUI_SRC} ${GLAD_SRC})

set(GLFW_LIB_PATH ${PROJECT_SOURCE_DIR}/ThirdParty/Glfw/lib/lib-vc2022)

if(NOT EXISTS "${GLFW_LIB_PATH}/glfw3.lib")
    message(FATAL_ERROR "glfw3.lib not found at: ${GLFW_LIB_PATH}")
else()
    message(STATUS "Found glfw3.lib at: ${GLFW_LIB_PATH}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -DNDEBUG")  # 启用优化，禁用断言
else()
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0 -G")       # Debug配置
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(${PROJECT_NAME} 
        "${GLFW_LIB_PATH}/glfw3.lib")
else()
    target_link_libraries(${PROJECT_NAME} 
        "${GLFW_LIB_PATH}/glfw3.lib")
endif()