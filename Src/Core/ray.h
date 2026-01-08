#pragma once
#include "global.h"
#include <cuda_runtime.h>

class Ray
{
public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(Vec3f origin, Vec3f direction) : origin(origin), direction(direction) {}
    __host__ __device__ Vec3f operator()(float t) { return origin + direction * t; }

    Vec3f origin;
    Vec3f direction;

    // __align__(16) float data[4];
};