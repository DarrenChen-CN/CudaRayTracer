#pragma once
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <cuda.h>
#include <iostream>
#define eps 1e-6
#define PI 3.1415926
#define FLOATMAX 3.40282e+038
#define FLOATMIN 1.17549e-038
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Vector2f Vec2f;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector4i Vec4i;
typedef Eigen::Vector4f Vec4f;
typedef Eigen::Matrix4f Mat4f;
typedef Eigen::Matrix3f Mat3f;
// 输出文件名和行号
#define CHECK_CUDA_ERROR(call)                                                                  \
    do                                                                                          \
    {                                                                                           \
        cudaError_t err = call;                                                                 \
        if (err != cudaSuccess)                                                                 \
        {                                                                                       \
            std::cout << "CUDA error in " << __FILE__ << ": " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        }                                                                                       \
    } while (0)
