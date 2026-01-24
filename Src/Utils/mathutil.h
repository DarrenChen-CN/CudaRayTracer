#pragma once
#include "global.h"
#include <memory>

__host__ __device__ float AngleToRadian(float angle);
__host__ __device__ float RadianToAngle(float radian);
__host__ __device__ int MaxDimension(const Vec3f &v);
__host__ __device__ Vec3f Permute(const Vec3f &v, int dim0, int dim1, int dim2);
__host__ __device__ Vec3f Abs(const Vec3f &v);
__host__ __device__ Vec3f LocalToWorld(Vec3f v, Vec3f normal);
__host__ __device__ Vec4f Vec3ToVec4(Vec3f v);
__host__ __device__ Vec3f Vec4ToVec3(Vec4f v);
__host__ __device__ float Clamp(float a, float b, float c);
__host__ __device__ float RandomValue();
__host__ __device__ Vec3f Min(Vec3f v1, Vec3f v2);
__host__ __device__ Vec3f Max(Vec3f v1, Vec3f v2);
__host__ __device__ Vec3f Lerp(const Vec3f &v1, const Vec3f &v2, float t);
__host__ __device__ Vec3f Reflect(const Vec3f &I, const Vec3f &N);
__host__ __device__ float Luminance(const Vec3f& color);

// transform
__host__ __device__ Mat4f Translate(const Vec3f &trans);
__host__ __device__ Mat4f Scale(const Vec3f &scale);
__host__ __device__ Mat4f RotateX(float angle);
__host__ __device__ Mat4f RotateY(float angle);
__host__ __device__ Mat4f RotateZ(float angle);

// mvp matrix
__host__ __device__ Mat4f Perspective(float fovy, float aspect, float near, float far);
__host__ __device__ Mat4f LookAt(const Vec3f &eye, const Vec3f &center, const Vec3f &up);


__device__ void CreateONB(const Vec3f& normal, Vec3f& tangent, Vec3f& bitangent);
__host__ __device__ unsigned int PCGHash(unsigned int input);
