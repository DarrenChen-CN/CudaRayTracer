#pragma once
#include "global.h"
#include "sampler.h"
#include "triangle.h"
#include <cuda_runtime.h>
#include "texture.h"

class BVH;

enum LightType{
    AREA_LIGHT,
    POINT_LIGHT,
    ENV_LIGHT
};

class Light{
public:
    __host__ __device__ Light(){}
    __host__ __device__ ~Light(){}

    __device__ bool Visible(IntersectionInfo &info, TriangleSampleInfo &lightInfo, BVH *sceneBVH);

    __device__ void SamplePoint(Triangle *triangles, Bounds3D sceneBounds, TriangleSampleInfo &info, float &pdf, Sampler *sampler, int idx) const;
    __device__ void SamplePointPDF(Bounds3D sceneBounds, Vec3f &samplePoint, float &pdf) const;
    __device__ void SampleSolidAnglePDF(Ray &ray, Vec3f &lightPoint, Vec3f &lightNormal, float &pdf) const;

    __device__ void SampleAreaLight(Triangle *triangles, TriangleSampleInfo &info, float &pdf, Sampler *sampler, int idx) const;
    __device__ void SampleAreaLightPDF(Vec3f &samplePoint, float &pdf) const;
    __device__ void SampleAreaLightSolidAnglePDF(Ray &ray, Vec3f &lightPoint, Vec3f &lightNormal, float &pdf) const;

    __device__ void SampleEnvLight(Bounds3D sceneBounds, TriangleSampleInfo &info, Sampler *sampler, int idx) const;
    __device__ void SampleEnvLightPDF(Bounds3D sceneBounds, Vec3f &samplePoint, float &pdf) const;
    __device__ void SampleEnvLightSolidAnglePDF(Ray &ray, Vec3f &lightPoint,  float &pdf) const;

    __device__ Vec3f Emission(Vec3f &wi) const;

    Vec3f emission;
    Mat4f transform;
    Mat4f transformInv;

    LightType type = AREA_LIGHT;

    MeshData *mesh = nullptr; // for area light
    HDRTexture *envMap = nullptr; // for env light
};

void CreateLight(Light *hostLight, Light *deviceLight);