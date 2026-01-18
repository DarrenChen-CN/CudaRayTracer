#include "light.h"
#include "scene.h"
#include "bvh.h"

 __device__ bool Light::Visible(IntersectionInfo &info, TriangleSampleInfo &lightInfo, BVH *sceneBVH){
    Vec3f hitPoint = info.hitPoint;
    Vec3f hitNormal = info.normal;
    Vec3f lightPoint = lightInfo.position;
    Vec3f lightNormal = lightInfo.normal;
    hitPoint = hitPoint + 1e-4 * hitNormal;
    lightPoint = lightPoint + 1e-4* lightNormal;

    float sampleDistance = (hitPoint - lightPoint).norm();
    Vec3f rayDirection = (hitPoint - lightPoint).normalized();
    Ray ray(lightPoint + 1e-2 * rayDirection, rayDirection);

    IntersectionInfo testInfo;
    bool hit = sceneBVH -> IsIntersect(ray, testInfo);
    // return true;
    if(!hit)return true;

    Vec3f testHitPoint = testInfo.hitPoint;
    float testDistance = (testHitPoint - lightPoint).norm();
    if((hitPoint - testHitPoint).norm() > 1e-1 && abs(testDistance - sampleDistance) > 1e-1)return false;

    return true;
 }

 __device__ void Light::SamplePoint(Triangle *triangles, Bounds3D sceneBounds, TriangleSampleInfo &info, float &pdf, Sampler *sampler, int idx) const{
    if(type == AREA_LIGHT){
        SampleAreaLight(triangles, info, pdf, sampler, idx);
    }else if(type == ENV_LIGHT){
        SampleEnvLight(sceneBounds, info, sampler, idx);
    }
    else {
        printf("Light type not supported for SamplePoint\n");
    }
 }

 __device__ void Light::SamplePointPDF(Bounds3D sceneBound, Vec3f &samplePoint, float &pdf) const{
    if(type == AREA_LIGHT){
        SampleAreaLightPDF(samplePoint, pdf);
    }else if(type == ENV_LIGHT){
        SampleEnvLightPDF(sceneBound, samplePoint, pdf);
    }else{
        printf("Light type not supported for SamplePointPDF\n");
    }
 }

 __device__ void Light::SampleSolidAnglePDF(Ray &ray, Vec3f &lightPoint, Vec3f &lightNormal,float &pdf) const{
    if(type == AREA_LIGHT){
        SampleAreaLightSolidAnglePDF(ray, lightPoint, lightNormal, pdf);
    }else if(type == ENV_LIGHT){
        SampleEnvLightSolidAnglePDF(ray, lightPoint, pdf);
    }else{
        printf("Light type not supported for SampleSolidAnglePDF\n");
    }
 }

 __device__ void Light::SampleAreaLight(Triangle *triangles, TriangleSampleInfo &info, float &pdf, Sampler *sampler, int idx) const{
    if(type == AREA_LIGHT && mesh != nullptr){
        int triangleId = static_cast<int>(sampler->Get1D(idx) * mesh->numTriangles);
        if(triangleId == mesh->numTriangles) triangleId = mesh->numTriangles - 1; //
        triangles[triangleId + mesh -> startTriangleID].Sample(info, sampler, idx);
        SampleAreaLightPDF(info.position, pdf);
    }else {
        printf("Area Light doesn't contain mesh\n");
    }
 }

__device__ void Light::SampleAreaLightPDF(Vec3f &samplePoint, float &pdf) const{
    if(type == AREA_LIGHT && mesh != nullptr){
        pdf = 1.f / mesh -> area;
    }else{
        printf("Area Light doesn't contain mesh\n");
    }
}

__device__ void Light::SampleAreaLightSolidAnglePDF(Ray &ray, Vec3f &lightPoint, Vec3f &lightNormal,float &pdf) const{
    if(type == AREA_LIGHT && mesh != nullptr){
        Vec3f hitPoint = ray.origin;
        Vec3f wi = (hitPoint - lightPoint).normalized();
        float distanceSquared = (hitPoint - lightPoint).squaredNorm();
        float cosTheta = fmaxf(lightNormal.dot(wi), 1e-6f);
        pdf = distanceSquared / (mesh -> area * cosTheta);
    }else{
        printf("Area Light doesn't contain mesh\n");
    }
}

__device__ void Light::SampleEnvLight(Bounds3D sceneBounds, TriangleSampleInfo &info, Sampler *sampler, int idx) const{
    if(type == ENV_LIGHT && envMap != nullptr){
        // Sample spherical direction
        float u, v, pdf;
        envMap -> Sample(u, v, sampler, idx);
        float theta = u * 2.f * PI;
        float phi = acosf(1.f - 2.f * v);
        float x = sinf(phi) * sinf(theta);
        float z = sinf(phi) * cosf(theta);
        float y = cosf(phi);
        Vec3f dir = Vec3f(x, y, z);
        info.position = sceneBounds.Center() + dir * 2 * sceneBounds.DiagonalLength();  // place the light point far away, no normal for env light
        info.normal = -dir; // point towards the scene center
    }else{
        printf("Env Light doesn't contain envMap\n");
    }
}

__device__ void Light::SampleEnvLightPDF(Bounds3D sceneBounds, Vec3f &samplePoint, float &pdf) const{
    if(type == ENV_LIGHT && envMap != nullptr){
        Vec3f dir = (samplePoint - sceneBounds.Center()).normalized();
        // Convert dir to uv
        float theta = acosf(dir(1)); // y is up
        float phi = atan2f(dir(0), dir(2));
        if(phi < 0) phi += 2.f * PI;
        float u = phi / (2.f * PI);
        float v = 1.f - (theta / PI); // flip v
        envMap -> SamplePDF(u, v, pdf);
    }else{
        printf("Env Light doesn't contain envMap\n");
    }
}

__device__ void Light::SampleEnvLightSolidAnglePDF(Ray &ray, Vec3f &lightPoint, float &pdf) const{
    if(type == ENV_LIGHT && envMap != nullptr){
        Vec3f dir = (lightPoint - ray.origin).normalized();
        // Convert dir to uv
        float theta = acosf(dir(1)); // y is up
        float phi = atan2f(dir(0), dir(2));
        if(phi < 0) phi += 2.f * PI;
        float u = phi / (2.f * PI);
        float v = 1.f - (theta / PI); // flip v
        envMap -> SampleSolidAnglePDF(u, v, pdf);
    }else{
        printf("Env Light doesn't contain envMap\n");
    }
}

__device__ Vec3f Light::Emission(Vec3f &wi) const{
    if(type == ENV_LIGHT && envMap != nullptr){
        // Convert wo to spherical coordinates
        float theta = acosf(wi(1)); // y is up
        float phi = atan2f(wi(0), wi(2));
        if(phi < 0) phi += 2.f * PI;
        // Map to [0,1]
        float u = phi / (2.f * PI);
        float v = 1.f - (theta / PI); // flip v
        // printf("Env Light Emission UV: (%f, %f)\n", u, v);
        // printf("cudaTextureObj: %d\n", envMap -> cudaTextureObj);
        float4 color = tex2D<float4>(envMap -> cudaTextureObj, u, v);
        // printf("Env Light Emission Color: (%f, %f, %f)\n", color.x, color.y, color.z);
        return Vec3f(color.x, color.y, color.z);
    }else{
        return emission;
    }
}

void CreateLight(Light *hostLight, Light *deviceLight){
    cudaMemcpy(deviceLight, hostLight, sizeof(Light), cudaMemcpyHostToDevice);
 }