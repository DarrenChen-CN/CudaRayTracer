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
    if(!hit)return false;

    Vec3f testHitPoint = testInfo.hitPoint;
    float testDistance = (testHitPoint - lightPoint).norm();
    if((hitPoint - testHitPoint).norm() > 1e-1 && abs(testDistance - sampleDistance) > 1e-1)return false;

    return true;

    // Vec3f hitPoint = info.hitPoint + 1e-1 * info.normal;
    // Vec3f lightPoint = lightInfo.position + 1e-1 * lightInfo.normal;
    // Vec3f dir = (lightPoint - hitPoint).normalized();
    // Ray ray(hitPoint, dir);
    // IntersectionInfo testInfo;
    // bool hit = sceneBVH -> IsIntersect(ray, testInfo);
    // if(!hit) return true;
    // float EPS = 5e-1;
    // if((testInfo.hitPoint - lightPoint).norm() < EPS) return true;
    // // printf("ray origin: %f, %f, %f; hit point: %f, %f, %f\n", ray.origin(0), ray.origin(1), ray.origin(2), testInfo.hitPoint(0), testInfo.hitPoint(1), testInfo.hitPoint(2));
    // return false;
 }

 __device__ void Light::SamplePoint(TriangleSampleInfo &info, float &pdf, Sampler *sampler, int idx) const{
    if(type == AREA_LIGHT && mesh != nullptr){
        mesh -> Sample(info, sampler, idx);
        SamplePointPDF(info.position, pdf);
    }else {
        printf("Area Light doesn't contain mesh\n");
    }
 }

 __device__ void Light::SamplePointPDF(Vec3f &samplePoint, float &pdf) const{
    if(type == AREA_LIGHT && mesh != nullptr){
        pdf = 1.f / area;
    }else{
        printf("Area Light doesn't contain mesh\n");
    }
 }

void CreateLight(Light *hostLight, Light *deviceLight){
    // cudaMalloc(&deviceLight, sizeof(Light));
    if(hostLight -> mesh != nullptr){
        Triangle *deviceMesh;
        cudaMalloc(&deviceMesh, sizeof(Triangle) * hostLight -> numTriangles);
        cudaMemcpy(deviceMesh, hostLight -> mesh, sizeof(Triangle) * hostLight -> numTriangles, cudaMemcpyHostToDevice);
        hostLight -> mesh = deviceMesh;
    }
    cudaMemcpy(deviceLight, hostLight, sizeof(Light), cudaMemcpyHostToDevice);
 }