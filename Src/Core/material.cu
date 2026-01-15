#include "material.h"
#include "mathutil.h"

__host__ __device__ Material::Material(){

}

__host__ __device__ Material::~Material(){

}

__device__ Vec3f Material::Evaluate(Vec3f& wi, Vec3f& normal, Vec3f& wo) const{
    switch(type)
    {
        case DIFFUSE:
            return EvaluateDiffuse(wi, normal, wo);
        case PBR:
            return EvaluatePBR(wi, normal, wo);
        default:
            return EvaluateDiffuse(wi, normal, wo);
    }
}

__device__ void Material::Pdf(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf) const{
    pdf = 0.0f;
    switch(type)
    {
        case DIFFUSE:
            PdfDiffuse(wi, normal, wo, pdf);
            break;
        case PBR:
            PdfPBR(wi, normal, wo, pdf);
            break;
        default:
            PdfDiffuse(wi, normal, wo, pdf);
            break;
    }
}

__device__ void Material::Sample(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx) const{
    switch(type)
    {
        case DIFFUSE:
            SampleDiffuse(wi, normal, wo, pdf, sampler, idx);
            break;
        case PBR:
            SamplePBR(wi, normal, wo, pdf, sampler, idx);
            break;
        default:
            SampleDiffuse(wi, normal, wo, pdf, sampler, idx);
            break;
    }
}

// diffuse
__device__ Vec3f Material::EvaluateDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo) const{
    if (wi.dot(normal) > 0.f && wo.dot(normal) > 0.f)
        return kd / PI;
    return Vec3f(0.f, 0.f, 0.f);
}

__device__ void Material::PdfDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf) const{
    if (wi.dot(normal) > 0.f && wo.dot(normal) > 0) {
        pdf = wo.dot(normal) / PI;
        return;
    }
    pdf = 0.f;
}

__device__ void Material::SampleDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx) const{
    Vec2f sample = sampler->Get2D(idx);
    float t1 = sample(0), t2 = sample(1);
    float theta = 0.5 * acos(1 - 2 * t1), phi = 2 * PI * t2;
    float x = sin(theta) * cos(phi), y = sin(theta) * sin(phi), z = cos(theta);
    wo = LocalToWorld(Vec3f(x, y, z), normal);
    PdfDiffuse(wi, normal, wo, pdf);
    return;
}

// PBR
// 采样过程中wi是 p -> eye方向的入射光，wo是 p -> light方向的出射光
__device__ void Material::DistributionGGX(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& D) const{
    Vec3f halfVector = (wi + wo).normalized();
    float NdotH = fmaxf(normal.dot(halfVector), 0.f);
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (alpha - 1.f) + 1.f);
    denom = PI * denom * denom;
    D = alpha2 / denom;
}

__device__ void Material::FresnelSchlick(Vec3f& wi, Vec3f& normal, Vec3f& wo, Vec3f& F) const{
    Vec3f halfVector = (wi + wo).normalized();
    float HdotV = fmaxf(halfVector.dot(wi), 0.f);
    F = F0 + (Vec3f(1.f, 1.f, 1.f) - F0) * powf((1.f - HdotV), 5.f);
}

__device__ void Material::GeometrySchlickGGX(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& G) const{
    float NdotV = fmaxf(normal.dot(wi), 0.f);
    float NdotL = fmaxf(normal.dot(wo), 0.f);
    float r = (roughness + 1.f);
    float k = (r * r) / 8.f;

    float G1V = NdotV / (NdotV * (1.f - k) + k);
    float G1L = NdotL / (NdotL * (1.f - k) + k);

    G = G1V * G1L;
}

__device__ Vec3f Material::EvaluatePBR(Vec3f& wi, Vec3f& normal, Vec3f& wo) const{
    float NdotV = max(normal.dot(wi), 0.0f);
    float NdotL = max(normal.dot(wo), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return Vec3f(0.f, 0.f, 0.f);

    Vec3f h = (wi + wo).normalized();
    float NdotH = max(normal.dot(h), 0.0f);
    float VdotH = max(wi.dot(h), 0.0f);

    // 1. Specular 
    float D, G;
    Vec3f F;
    DistributionGGX(wi, normal, wo, D);
    GeometrySchlickGGX(wi, normal, wo, G);
    FresnelSchlick(wi, normal, wo, F);

    Vec3f numerator = F * (D * G);
    float denominator = 4.0f * NdotV * NdotL + 0.0001f;
    Vec3f specular = numerator / denominator;

    // 2. Diffuse
    Vec3f ks = F;
    Vec3f kd = (Vec3f(1.0f, 1.0f, 1.0f) - ks) * (1.0f - metallic);
    // printf("kd: %f, %f, %f\n", kd.x(), kd.y(), kd.z());
    Vec3f diffuse;
    if(wi.dot(normal) > 0.f && wo.dot(normal) > 0.f){
        diffuse = kd.cwiseProduct(basecolor) / PI;
    }else{
        diffuse = Vec3f(0.f, 0.f, 0.f);
    }

    return diffuse + specular;
}

__device__ void Material::PdfPBR(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf) const{
    // P_spec
    float NdotV = max(normal.dot(wi), 0.0f);
    Vec3f FAppro = F0 + (Vec3f(1.0f, 1.0f, 1.0f) - F0) * powf((1.0f - NdotV), 5.0f);
    
    float spec_strength = Luminance(FAppro);
    float diff_strength = Luminance(basecolor) * (1.0f - metallic);
    float P_spec = fmaxf(spec_strength / (spec_strength + diff_strength), 1e-6f);
    P_spec = Clamp(0.1f, 0.9f, P_spec);

    // PDF
    float NdotL = max(normal.dot(wo), 0.0f);
    if (NdotL <= 0.0f) { pdf = 0.f; return; }

    Vec3f h = (wi + wo).normalized();
    float VdotH = max(wi.dot(h), 1e-6f);
    float NdotH = max(normal.dot(h), 1e-6f);

    // 1. Specular PDF (GGX)
    float D;
    DistributionGGX(wi, normal, wo, D);
    float pdf_spec = (D * NdotH) / (4.0f * VdotH); // h -> wi

    // 2. Diffuse PDF (Cosine Weighted)
    float pdf_diff = NdotL / PI;

    // printf("P_spec: %f, pdf_spec: %f, pdf_diff: %f\n", P_spec, pdf_spec, pdf_diff);

    // 3. mix PDF
    pdf = P_spec * pdf_spec + (1.0f - P_spec) * pdf_diff;
}

__device__ void Material::SamplePBR(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx) const{
    // P_spec
    float NdotV = max(normal.dot(wi), 0.0f);
    Vec3f FAppro = F0 + (Vec3f(1.0f, 1.0f, 1.0f) - F0) * powf((1.0f - NdotV), 5.0f);
    
    float spec_strength = Luminance(FAppro);
    float diff_strength = Luminance(basecolor) * (1.0f - metallic);
    float P_spec = fmaxf(spec_strength / (spec_strength + diff_strength), 1e-6f);
    
    P_spec = Clamp(0.1f, 0.9f, P_spec);

    float r1 = sampler -> Get1D(idx);
    float r2 = sampler -> Get1D(idx);
    float choose_spec = sampler -> Get1D(idx);

    if (choose_spec < P_spec) {
        // Specular (GGX)
        float a = roughness * roughness;
        float phi = 2.0f * PI * r1;
        float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
        float sinTheta = sqrtf(max(0.0f, 1.0f - cosTheta * cosTheta));

        Vec3f hLocal(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
        Vec3f hWorld = LocalToWorld(hLocal, normal); 
        
        wo = Reflect(-wi, hWorld).normalized();
    } else {
        // Diffuse (Cosine Weighted)
        float phi = 2.0f * PI * r1;
        float cosTheta = sqrtf(r2);
        float sinTheta = sqrtf(1.0f - r2);

        Vec3f lLocal(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
        wo = LocalToWorld(lLocal, normal);
    }

    PdfPBR(wi, normal, wo, pdf);
}
__device__ bool Material::IsLight() const{
    return type == LIGHT;
}

void CreateMaterial(Material *hostMaterial, Material *deviceMaterial){
    // cudaMalloc(&deviceMaterial, sizeof(Material));
    cudaMemcpy(deviceMaterial, hostMaterial, sizeof(Material), cudaMemcpyHostToDevice);
}
