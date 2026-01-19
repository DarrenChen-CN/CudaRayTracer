#pragma once 
#include "global.h"
#include "sampler.h"

enum MaterialType {
    LIGHT,
    DIFFUSE,
    PBR
};

struct MaterialSampleData {
    Vec3f hitPoint;
    Vec3f wi;
    Vec3f wo;
    Vec3f normal;
    float pdf;

    // bssrdf
    Vec3f outgoingPoint;
};

class Material {

public:
    __host__ __device__ Material();
    __host__ __device__ ~Material();

    __device__ Vec3f Evaluate(Vec3f& wi, Vec3f& normal, Vec3f& wo) const;
    __device__ void Pdf(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf) const;
    __device__ void Sample(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx) const;

    // diffuse
    __device__ Vec3f EvaluateDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo) const;
    __device__ void PdfDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf) const;
    __device__ void SampleDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx) const;

    // PBR
    __device__ Vec3f EvaluatePBR(Vec3f& wi, Vec3f& normal, Vec3f& wo) const;
    __device__ void PdfPBR(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf) const;
    __device__ void SamplePBR(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx) const; 
    __device__ void DistributionGGX(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& D) const;
    __device__ void FresnelSchlick(Vec3f& wi, Vec3f& normal, Vec3f& wo, Vec3f& F) const;
    __device__ void GeometrySchlickGGX(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& G) const;

    __device__ bool IsLight() const;

    std::string name;
    MaterialType type;
    Vec3f ke = Vec3f(0.f, 0.f, 0.f); // emission
    Vec3f kd = Vec3f(0.f, 0.f, 0.f); // diffuse
    Vec3f ks = Vec3f(0.f, 0.f, 0.f); // specular
    
    Vec3f F0 = Vec3f(0.f, 0.f, 0.f); // F0 for PBR
    Vec3f basecolor = Vec3f(0.f, 0.f, 0.f);
    float roughness = 0.5f;
    float metallic = 0.0f;
};

void CreateMaterial(Material *hostMaterial, Material *deviceMaterial);