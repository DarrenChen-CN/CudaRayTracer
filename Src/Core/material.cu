#include "material.h"
#include "mathutil.h"
#include "ray.h"
#include "bvh.h"

__host__ __device__ Material::Material(){

}

__host__ __device__ Material::~Material(){

}

__device__ Vec3f Material::Evaluate(Vec3f& wi, Vec3f& normal, Vec3f& wo, Vec2f &uv) const{
    switch(type)
    {
        case DIFFUSE:
            return EvaluateDiffuse(wi, normal, wo, uv);
        case PBR:
        case SUBSURFACE:
            return EvaluatePBR(wi, normal, wo, uv);
        case DIELECTRIC:
            return Vec3f(0.f, 0.f, 0.f);
        default:
            return EvaluateDiffuse(wi, normal, wo, uv);
    }
}

__device__ void Material::Pdf(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Vec2f &uv) const{
    pdf = 0.0f;
    switch(type)
    {
        case DIFFUSE:
            PdfDiffuse(wi, normal, wo, pdf, uv);
            break;
        case PBR:
        case SUBSURFACE:
            PdfPBR(wi, normal, wo, pdf, uv);
            break;
        case DIELECTRIC:
            pdf = 0.0f;
            break;
        default:
            PdfDiffuse(wi, normal, wo, pdf, uv);
            break;
    }
}

__device__ void Material::Sample(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx, Vec2f &uv) const{
    switch(type)
    {
        case DIFFUSE:
            SampleDiffuse(wi, normal, wo, pdf, sampler, idx, uv);
            break;
        case PBR:
        case SUBSURFACE:
            SamplePBR(wi, normal, wo, pdf, sampler, idx, uv); // use pbr sampling for subsurface default
            break;
        case DIELECTRIC: {
            Vec3f weight;
            SampleDielectric(wi, normal, wo, pdf, weight, sampler, idx);
            break;
        }
        default:
            SampleDiffuse(wi, normal, wo, pdf, sampler, idx, uv);
            break;
    }
}

// diffuse
__device__ Vec3f Material::EvaluateDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo, Vec2f &uv) const{
    Vec3f kd;
    if(usingDiffuseTexture && diffuseTexture != nullptr)
        kd = diffuseTexture->Sample(uv(0), uv(1));
    else kd = basecolor;
    if (wo.dot(normal) > 0.f)
        return kd / PI;
    return Vec3f(0.f, 0.f, 0.f);
}

__device__ void Material::PdfDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Vec2f &uv) const{
    if (wo.dot(normal) > 0.f) {
        pdf = wo.dot(normal) / PI;
        return;
    }
    pdf = 0.f;
}

__device__ void Material::SampleDiffuse(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx, Vec2f &uv) const{
    Vec2f sample = sampler->Get2D(idx);
    float t1 = sample(0), t2 = sample(1);
    float theta = 0.5 * acos(1 - 2 * t1), phi = 2 * PI * t2;
    float x = sin(theta) * cos(phi), y = sin(theta) * sin(phi), z = cos(theta);
    wo = LocalToWorld(Vec3f(x, y, z), normal);
    PdfDiffuse(wi, normal, wo, pdf, uv);
    return;
}

// PBR
// 采样过程中wi为p -> eye方向的入射光，wo为p -> light方向的出射光
__device__ void Material::DistributionGGX(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& D) const{
    Vec3f halfVector = (wi + wo).normalized();
    float NdotH = fmaxf(normal.dot(halfVector), 0.f);
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (alpha2 - 1.f) + 1.f);
    denom = PI * denom * denom;
    D = alpha2 / denom;
}

__device__ void Material::FresnelSchlick(Vec3f& wi, Vec3f& normal, Vec3f& wo, Vec3f& F, Vec3f& F0) const{
    Vec3f halfVector = (wi + wo).normalized();
    float HdotV = fmaxf(halfVector.dot(wi), 0.f);
    F = F0 + (Vec3f(1.f, 1.f, 1.f) - F0) * powf((1.f - HdotV), 5.f);
}

__device__ void Material::Fresnel(Vec3f& wi, Vec3f& normal, Vec3f& F) const{
    float reflectance = FresnelDielectric(normal.dot(wi), 1.0f, ior);
    F = Vec3f(reflectance, reflectance, reflectance);
}

__device__ float Material::FresnelDielectric(float cosThetaI, float etaI, float etaT) const{
    cosThetaI = Clamp(-1.0f, 1.0f, cosThetaI);
    bool entering = cosThetaI > 0.0f;
    if (!entering) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
        cosThetaI = fabsf(cosThetaI);
    }

    float sinThetaI = sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1.0f) {
        return 1.0f;
    }

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));
    float rParallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float rPerpendicular = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return 0.5f * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}

__device__ bool Material::RefractDirection(const Vec3f& incident, const Vec3f& normal, float eta, Vec3f& refracted) const{
    float cosThetaI = fmaxf(0.0f, -incident.dot(normal));
    float sin2ThetaT = eta * eta * fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI);
    if (sin2ThetaT >= 1.0f) {
        return false;
    }

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sin2ThetaT));
    refracted = eta * incident + (eta * cosThetaI - cosThetaT) * normal;
    return true;
}

__device__ void Material::SampleDielectric(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Vec3f& weight, Sampler* sampler, int idx) const{
    Vec3f incident = -wi.normalized();
    Vec3f orientedNormal = normal.normalized();
    float etaI = 1.0f;
    float etaT = ior;

    if (incident.dot(orientedNormal) > 0.0f) {
        orientedNormal = -orientedNormal;
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
    }

    float cosThetaI = fmaxf(0.0f, -incident.dot(orientedNormal));
    float fresnel = FresnelDielectric(cosThetaI, etaI, etaT);
    float randomValue = sampler->Get1D(idx);

    if (randomValue < fresnel) {
        wo = Reflect(incident, orientedNormal).normalized();
        pdf = fmaxf(fresnel, 1e-6f);
        weight = Vec3f(1.0f, 1.0f, 1.0f);
        return;
    }

    Vec3f refracted;
    float eta = etaI / etaT;
    if (!RefractDirection(incident, orientedNormal, eta, refracted)) {
        wo = Reflect(incident, orientedNormal).normalized();
        pdf = 1.0f;
        weight = Vec3f(1.0f, 1.0f, 1.0f);
        return;
    }

    wo = refracted.normalized();
    pdf = fmaxf(1.0f - fresnel, 1e-6f);
    weight = Lerp(Vec3f(1.0f, 1.0f, 1.0f), basecolor, transmission);
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

__device__ Vec3f Material::EvaluatePBR(Vec3f& wi, Vec3f& normal, Vec3f& wo, Vec2f &uv) const{
    float NdotV = max(normal.dot(wi), 0.0f);
    float NdotL = max(normal.dot(wo), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return Vec3f(0.f, 0.f, 0.f);

    Vec3f albedo;
    if(usingDiffuseTexture && diffuseTexture != nullptr)
        albedo = diffuseTexture->Sample(uv(0), uv(1));
    else albedo = basecolor;

    Vec3f h = (wi + wo).normalized();
    float NdotH = max(normal.dot(h), 0.0f);
    float VdotH = max(wi.dot(h), 0.0f);

    // 1. Specular 
    float D, G;
    Vec3f F, F0;
    F0 = Lerp(Vec3f(0.04f, 0.04f, 0.04f), albedo, metallic);
    DistributionGGX(wi, normal, wo, D);
    GeometrySchlickGGX(wi, normal, wo, G);
    FresnelSchlick(wi, normal, wo, F, F0);
    Vec3f numerator = F * (D * G);
    float denominator = 4.0f * NdotV * NdotL + 0.0001f;
    Vec3f specular = numerator / denominator;

    // 2. Diffuse
    Vec3f ks = F;
    Vec3f kd = (Vec3f(1.0f, 1.0f, 1.0f) - ks) * (1.0f - metallic);
    Vec3f diffuse;
    if(wi.dot(normal) > 0.f && wo.dot(normal) > 0.f){
        diffuse = kd.cwiseProduct(albedo) / PI;
    }else{
        diffuse = Vec3f(0.f, 0.f, 0.f);
    }

    return diffuse + specular;
}

__device__ void Material::PdfPBR(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Vec2f &uv) const{
    // P_spec
    Vec3f albedo;
    if(usingDiffuseTexture && diffuseTexture != nullptr)
        albedo = diffuseTexture->Sample(uv(0), uv(1));
    else albedo = basecolor;

    Vec3f F0;
    F0 = Lerp(Vec3f(0.04f, 0.04f, 0.04f), albedo, metallic);
    float NdotV = max(normal.dot(wi), 0.0f);
    Vec3f FAppro = F0 + (Vec3f(1.0f, 1.0f, 1.0f) - F0) * powf((1.0f - NdotV), 5.0f);
    
    float spec_strength = Luminance(FAppro);
    float diff_strength = Luminance(albedo) * (1.0f - metallic);
    float P_spec = fmaxf(spec_strength / (spec_strength + diff_strength), 1e-6f);
    P_spec = Clamp(0.01f, 0.99f, P_spec);

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

    // 3. mix PDF
    pdf = P_spec * pdf_spec + (1.0f - P_spec) * pdf_diff;
}

__device__ void Material::SamplePBR(Vec3f& wi, Vec3f& normal, Vec3f& wo, float& pdf, Sampler* sampler, int idx, Vec2f &uv) const{
    // P_spec
    Vec3f albedo;
    if(usingDiffuseTexture && diffuseTexture != nullptr)
        albedo = diffuseTexture->Sample(uv(0), uv(1));
    else albedo = basecolor;

    Vec3f F0;
    F0 = Lerp(Vec3f(0.04f, 0.04f, 0.04f), albedo, metallic);
    float NdotV = max(normal.dot(wi), 0.0f);
    Vec3f FAppro = F0 + (Vec3f(1.0f, 1.0f, 1.0f) - F0) * powf((1.0f - NdotV), 5.0f);
    
    float spec_strength = Luminance(FAppro);
    float diff_strength = Luminance(albedo) * (1.0f - metallic);
    float P_spec = fmaxf(spec_strength / (spec_strength + diff_strength), 1e-6f);
    
    P_spec = Clamp(0.01f, 0.99f, P_spec);

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

    PdfPBR(wi, normal, wo, pdf, uv);
}

__device__ Vec3f Material::EvaluateSpatialDiffusionProfile(Vec3f& wi, Vec3f& normal, Vec3f& hitPoint, Vec3f& outgoingPoint, Vec3f &outgoingNormal, Vec2f &uv) const{
    // profile
    float distance = (outgoingPoint - hitPoint).norm();
    if(distance < 1e-6f)
        return Vec3f(0.f, 0.f, 0.f);
    Vec3f profile;
    BurleyDiffusionProfile(distance, profile(0), 0); 
    BurleyDiffusionProfile(distance, profile(1), 1); 
    BurleyDiffusionProfile(distance, profile(2), 2); 
    Vec3f albedo = basecolor;
    if(usingDiffuseTexture && diffuseTexture != nullptr)
        albedo = diffuseTexture->Sample(uv(0), uv(1));
    return albedo.cwiseProduct(profile) / PI;
}

__device__ void Material::SampleSpatialDiffusionProfile(Vec3f& wi, Vec3f& normal, Vec3f& hitPoint, Vec3f& outgoingPoint, Vec3f &outgoingNormal, float& pdf, Sampler* sampler, int idx, BVH *bvh) const{
    // sample r based on diffusion profile
    float distance, profilePdf;
    SampleBurleyDiffusionProfile(distance, profilePdf, sampler, idx);
    // sample angle
    float angle = sampler->Get1D(idx) * 2.f * PI;
    // compute outgoing point
    ProbeOutgongingPoint(hitPoint, normal, outgoingPoint, outgoingNormal, distance, angle, bvh);
    PdfSpatialDiffusionProfile(wi, normal, hitPoint, outgoingPoint, pdf);
}

__device__ void Material::PdfSpatialDiffusionProfile(Vec3f& wi, Vec3f& normal, Vec3f& hitPoint, Vec3f& outgoingPoint, float& pdf) const{
    // profile pdf
    float distance = (outgoingPoint - hitPoint).norm();
    if(distance < 1e-6f){
        pdf = 0.f;
        return;
    }
    PdfBurleyDiffusionProfile(distance, pdf);
}

__device__ void Material::SampleBurleyDiffusionProfile(float& distance, float &pdf, Sampler* sampler, int idx) const{
    // select channel
    int channel = sampler->Get1D(idx) * 3;;
    channel = Clamp(0, 2, channel);
    float d = scatterDistance(channel);
    if(d < 1e-6f){
        distance = 0.f;
        pdf = 0.f;
        return;
    }
    float r1 = sampler->Get1D(idx);
    if(r1 < 0.5f){
        r1 = r1 * 2.f; // [0, 0.5) -> [0, 1)
        distance = -d * logf(1.f - r1);
    }else{
        r1 = (r1 - 0.5f) * 2.f; // [0.5, 1) -> [0, 1)
        distance = -3.f * d * logf(1.f - r1);
    }

    PdfBurleyDiffusionProfile(distance, pdf);
}

__device__ void Material::BurleyDiffusionProfile(float& distance, float& pdf, int channel) const{
    float d = scatterDistance(channel);
    float r = distance;
    if(r < 1e-6f || d < 1e-6f){
        pdf = 0.f;
        return;
    }

    float exp1 = expf(-r / d);
    float exp2 = expf(- r / (3 *d));
    pdf = (exp1 + exp2) / (8.f * PI * d * r);
}

__device__ void Material::PdfBurleyDiffusionProfile(float& distance, float& pdf) const{
    float pdf0, pdf1, pdf2;
    BurleyDiffusionProfile(distance, pdf0, 0);
    BurleyDiffusionProfile(distance, pdf1, 1);
    BurleyDiffusionProfile(distance, pdf2, 2);
    pdf = (pdf0 + pdf1 + pdf2) / 3.f;
}

__device__ void Material::ProbeOutgongingPoint(Vec3f& hitPoint, Vec3f& normal, Vec3f& outgoingPoint, Vec3f &outgoingNormal, float distance, float angle, BVH *bvh) const{
    Vec3f tangent, bitangent;
    Vec3f probeNormal = normal.normalized();
    CreateONB(probeNormal, tangent, bitangent);
    Vec3f offset = distance * (cosf(angle) * tangent + sinf(angle) * bitangent);
    outgoingPoint = hitPoint + offset;

    float rayOffset = fmaxf(distance, 1e-3f);
    float tMin = 1e-4f;
    float tMax = fmaxf(distance * 2.f, 1e-2f);

    Ray forwardProbe(outgoingPoint + probeNormal * rayOffset, -probeNormal);
    Ray backwardProbe(outgoingPoint - probeNormal * rayOffset, probeNormal);
    IntersectionInfo forwardHit, backwardHit;
    bool hitForward = bvh->IsIntersect(forwardProbe, forwardHit, tMin, tMax);
    bool hitBackward = bvh->IsIntersect(backwardProbe, backwardHit, tMin, tMax);

    if(hitForward && hitBackward){
        if(forwardHit.hitTime <= backwardHit.hitTime){
            outgoingPoint = forwardHit.hitPoint;
            outgoingNormal = forwardHit.normal;
        }else{
            outgoingPoint = backwardHit.hitPoint;
            outgoingNormal = backwardHit.normal;
        }
    }else if(hitForward){
        outgoingPoint = forwardHit.hitPoint;
        outgoingNormal = forwardHit.normal;
    }else if(hitBackward){
        outgoingPoint = backwardHit.hitPoint;
        outgoingNormal = backwardHit.normal;
    }else{
        // using the original hit point
        outgoingPoint = hitPoint;
        outgoingNormal = probeNormal;
    }
}

__device__ bool Material::IsLight() const{
    return type == LIGHT;
}

__device__ bool Material::IsDelta() const{
    return type == DIELECTRIC;
}

void CreateMaterial(Material *hostMaterial, Material *deviceMaterial){
    cudaMemcpy(deviceMaterial, hostMaterial, sizeof(Material), cudaMemcpyHostToDevice);
}


