#pragma once 
#include "global.h"
#include "sampler.h"
#include "texture.h"

enum MaterialType {
    LIGHT,
    DIFFUSE,
    PBR,
    SUBSURFACE
};

class BVH;

class Material {

public:
    __host__ __device__ Material();
    __host__ __device__ ~Material();

    __device__ Vec3f Evaluate(Vec3f& wi, Vec3f& normal, Vec3f& wo, Vec2f &uv) const;
    __device__ void Pdf(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Vec2f &uv) const;
    __device__ void Sample(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx, Vec2f &uv) const; // bvh for subsurface scattering

    // diffuse
    __device__ Vec3f EvaluateDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo, Vec2f &uv) const;
    __device__ void PdfDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Vec2f &uv) const;
    __device__ void SampleDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx, Vec2f &uv) const;

    // PBR
    __device__ Vec3f EvaluatePBR(Vec3f& wi, Vec3f& normal, Vec3f& wo, Vec2f &uv) const;
    __device__ void PdfPBR(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Vec2f &uv) const;
    __device__ void SamplePBR(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx, Vec2f &uv) const; 
    __device__ void DistributionGGX(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& D) const;
    __device__ void FresnelSchlick(Vec3f& wi, Vec3f& normal, Vec3f& wo, Vec3f& F, Vec3f& F0) const;
    __device__ void Fresnel(Vec3f& wi, Vec3f& normal, Vec3f& F) const;
    __device__ void GeometrySchlickGGX(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& G) const;

    // Subsurface Scattering
    __device__ Vec3f EvaluateSpatialDiffusionProfile(Vec3f& wi, Vec3f& normal, Vec3f& hitPoint, Vec3f& outgoingPoint, Vec3f &outgoingNormal) const;
    __device__ void SampleSpatialDiffusionProfile(Vec3f& wi, Vec3f& normal, Vec3f& hitPoint, Vec3f& outgoingPoint, Vec3f &outgoingNormal, float& pdf, Sampler* sampler, int idx, BVH *bvh) const; // sample outgoint point based on diffusion profile
    __device__ void PdfSpatialDiffusionProfile(Vec3f& wi, Vec3f& normal, Vec3f& hitPoint, Vec3f& outgoingPoint, float& pdf) const;
    __device__ void SampleBurleyDiffusionProfile(float &distance, float &pdf, Sampler* sampler, int idx) const; // sample r based on R(r)
    __device__ void BurleyDiffusionProfile(float& distance, float& pdf, int channel) const; // R(r)
    __device__ void PdfBurleyDiffusionProfile(float& distance, float& pdf) const;
    __device__ void ProbeOutgongingPoint(Vec3f& hitPoint, Vec3f& normal, Vec3f& outgoingPoint, Vec3f &outgoingNormal, float distance, float angle, BVH *bvh) const;

    __device__ bool IsLight() const;

    std::string name;
    MaterialType type;
    Vec3f ke = Vec3f(0.f, 0.f, 0.f); // emission
    
    Vec3f basecolor = Vec3f(0.f, 0.f, 0.f); // for pbr
    float roughness = 0.5f;
    float metallic = 0.0f;

    Vec3f scatterDistance = Vec3f(0.f, 0.f, 0.f); // for subsurface scattering
    float ior = 1.f;

    bool usingDiffuseTexture = false;
    Texture *diffuseTexture = nullptr;
};

void CreateMaterial(Material *hostMaterial, Material *deviceMaterial);