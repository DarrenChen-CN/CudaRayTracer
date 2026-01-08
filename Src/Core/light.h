#pragma once
#include "global.h"
#include "sampler.h"
#include "triangle.h"

class BVH;

enum LightType{
    AREA_LIGHT,
    POINT_LIGHT
};

class Light{
public:
    __host__ __device__ Light(){}
    __host__ __device__ ~Light(){}

    __device__ bool Visible(IntersectionInfo &info, TriangleSampleInfo &lightInfo, BVH *sceneBVH);

    __device__ void SamplePoint(TriangleSampleInfo &info, float &pdf, Sampler *sampler, int idx) const;
    __device__ void SamplePointPDF(Vec3f &samplePoint, float &pdf) const;
    __device__ void SampleDirectionPDF(IntersectionInfo &info, Vec3f &wi, float &pdf) const;

    Vec3f emission;
    Mat4f transform;
    Mat4f transformInv;

    LightType type = AREA_LIGHT;
    Triangle* mesh = nullptr; // Only for area light
    float area = 0.f;
    int numTriangles = 0;
};

void CreateLight(Light *hostLight, Light *deviceLight);