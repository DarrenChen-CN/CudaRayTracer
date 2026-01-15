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
 }

 __device__ void Light::SamplePoint(TriangleSampleInfo &info, float &pdf, Sampler *sampler, int idx) const{
    if(type == AREA_LIGHT && mesh != nullptr){
        int triangleId = static_cast<int>(sampler->Get1D(idx) * numTriangles);
        if(triangleId == numTriangles) triangleId = numTriangles - 1; //
        mesh[triangleId].Sample(info, sampler, idx);
        // mesh -> Sample(info, sampler, idx);
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

 __device__ void Light::SampleDirectionPDF(Ray &ray, Vec3f &lightPoint, Vec3f &lightNormal,float &pdf) const{
    if(type == AREA_LIGHT && mesh != nullptr){
        Vec3f hitPoint = ray.origin;
        Vec3f wi = (hitPoint - lightPoint).normalized();
        float distanceSquared = (hitPoint - lightPoint).squaredNorm();
        float cosTheta = fmaxf(lightNormal.dot(wi), 1e-6f);
        pdf = distanceSquared / (area * cosTheta);
        // if(area > 0.04)
        //     printf("r2: %f, cosTheta: %f, area: %f, pdf: %f, triangle num: %d\n", distanceSquared, cosTheta, area, pdf, numTriangles);
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