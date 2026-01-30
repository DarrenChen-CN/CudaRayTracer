#pragma once
#include "global.h"
#include "ray.h"
#include "mathutil.h"
#include <thrust/execution_policy.h>

class Bounds3D
{
public:
    __host__ __device__ Bounds3D()
    {
        min = Vec3f(FLOATMAX, FLOATMAX, FLOATMAX);
        max = Vec3f(-FLOATMAX, -FLOATMAX, -FLOATMAX);
    }

    __host__ __device__ Bounds3D(const Vec3f& min, const Vec3f& max) : min(min), max(max)
    {
        
    }

    __host__ __device__ Bounds3D(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2)
    {
        min = Min(v0, Min(v1, v2));
        max = Max(v0, Max(v1, v2));
    }

    __host__ __device__ Bounds3D Expand(const Vec3f& point)
    {
        Vec3f newMin = min.cwiseMin(point);
        Vec3f newMax = max.cwiseMax(point);
        return Bounds3D(newMin, newMax);
    }

    __host__ __device__ Bounds3D Expand(const Bounds3D& other)
    {
        auto pmin = Min(min, other.min);
        auto pmax = Max(max, other.max);
        return Bounds3D(pmin, pmax);
    }

    __device__ bool IsIntersect(const Ray& ray, Vec3f invDir, int dirIsNeg[3])
    {
        Vec3f tEnter = (min - ray.origin).cwiseProduct(invDir);
        Vec3f tExit = (max - ray.origin).cwiseProduct(invDir);
        for (int i = 0; i < 3; i++)
        {
            float temp;
            if (dirIsNeg[i])
            {
                float temp = tEnter(i);
                tEnter(i) = tExit(i);
                tExit(i) = temp;
            }
        }
        
        float exit = tExit.minCoeff(), enter = tEnter.maxCoeff();
        return exit - enter + 1e-2 >= 0 && exit + 1e-2 >= 0;
    }

    __device__ bool IsIntersect(const Ray& ray, Vec3f invDir, int dirIsNeg[3], float &t, float tMin, float tMax)
    {
        Vec3f tEnter = (min - ray.origin).cwiseProduct(invDir);
        Vec3f tExit = (max - ray.origin).cwiseProduct(invDir);
        for (int i = 0; i < 3; i++)
        {
            float temp;
            if (dirIsNeg[i])
            {
                float temp = tEnter(i);
                tEnter(i) = tExit(i);
                tExit(i) = temp;
            }
        }
        
        float exit = tExit.minCoeff(), enter = tEnter.maxCoeff();
        t = enter;
        return exit - enter + 1e-2 >= 0 && exit + 1e-2 >= 0 && enter <= tMax && exit >= tMin;
    }

    __host__ __device__ float SurfaceArea()
    {
        if(min(0) > max(0) || min(1) > max(1) || min(2) > max(2))
            return 0.f;
        Vec3f extent = max - min;
        return 2.f * (extent(0) * extent(1) + extent(1) * extent(2) + extent(2) * extent(0));
    }

    __host__ __device__ Vec3f Offset(const Vec3f& point)
    {
        Vec3f o = point - min;
        Vec3f ext = max - min;
        if (ext(0) > 0) o(0) /= ext(0);
        if (ext(1) > 0) o(1) /= ext(1);
        if (ext(2) > 0) o(2) /= ext(2);
        return o;
    }

    __host__ __device__ Vec3f Extent()
    {
        return max - min;
    }

    __host__ __device__ int MaxExtent()
    {
        int res;
        Extent().maxCoeff(&res);
        return res;
    }

    __host__ __device__ Vec3f Center()
    {
        return (min + max) * 0.5f;
    }

    __host__ __device__ float DiagonalLength()
    {
        return (max - min).norm();
    }

public:
    Vec3f min;
    Vec3f max;
};