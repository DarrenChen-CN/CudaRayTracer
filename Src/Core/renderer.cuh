#include "scene.h"
#include "ui.h"
#include "light.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "camera.h"
#include "lightmanager.h"
#include <filesystem>

// Render kernel

RenderSegment *renderSegments;
RenderSegment *renderSegmentsBuffer;
IntersectionInfo *intersections;
IntersectionInfo *intersectionsBuffer;
int *segmentValidFlags;
int *segmentPos;
int *materialIDs;
int *materialSortIndices;

dim3 generateRayBlockSize(32, 32);
dim3 renderBlockSize(128);
dim3 denoiseBlockSize(16, 16);
dim3 gatherBlockSize(32, 32);
dim3 generateRayGridSize;
dim3 renderGridSize;
dim3 gatherGridSize;
dim3 denoiseGridSize;

// reference from https://github.com/jacquespillet/SVGF
struct GBuffer{
    GLuint fbo;
    GLuint depthBuffer; // linear z
    // texture
    GLuint texPositionTriID; // rgb: position, a: triangleID 32F
    GLuint texNormalMatID; // rgb: normal, a: materialID 16UI
    GLuint texBaryMeshID; // rgb: barycentric coord, a: instanceID 16UI
    GLuint texMotionDepth; // rg: motion vector, b: linear depth, a: dZ 32F

    // cuda resource
    cudaGraphicsResource *cudaPositionTriIDResource;
    cudaGraphicsResource *cudaNormalMatIDResource;
    cudaGraphicsResource *cudaBaryMeshIDResource;
    cudaGraphicsResource *cudaMotionDepthResource;

    // texture resource object
    cudaTextureObject_t cudaTexObjPositionTriID;
    cudaTextureObject_t cudaTexObjNormalMatID;
    cudaTextureObject_t cudaTexObjBaryMeshID;
    cudaTextureObject_t cudaTexObjMotionDepth;

    int width, height;
};

struct RenderBuffer{
    float *directLightingBuffer; // rgb variance
    float *indirectLightingBuffer;
};

struct DenoiseParam{
    // float2 *moments;
    // float *variance;
    float2 *directMoments;
    float2 *indirectMoments;
    int *historyLength;
    int maxHistoryLength = 24;
    float sigmaLight = 30.f;
    float sigmaNormal = 128.f;
    float sigmaDepth = 1.f;
};

cudaTextureObject_t CreateTextureObject(cudaArray_t cuArray, cudaTextureAddressMode addressMode, cudaTextureFilterMode filterMode, cudaTextureReadMode readMode)
{
    cudaTextureObject_t texObj;
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = addressMode;
    texDesc.addressMode[1] = addressMode;
    texDesc.filterMode = filterMode;
    texDesc.readMode = readMode;
    texDesc.normalizedCoords = 0;

    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    return texObj;
}

__global__ void GenerateRayKernel(RenderSegment* segments, RenderParam renderParam, CameraParam cameraParam)
{
    int width = renderParam.width, height = renderParam.height;
    Camera *camera = renderParam.camera;
    Sampler *sampler = renderParam.sampler;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return; // Ensure we don't access out of bounds
    int idx = j * width + i;
    camera -> GeneratingRay(i, j, sampler, &segments[idx].ray, cameraParam);
    segments[idx].index = idx;
    segments[idx].remainingBounces = cameraParam.maxBounces; // Set the maximum number of bounces
    // segments[idx].color = Vec3f(0.0f, 0.0f, 0.0f);
    segments[idx].directColor = Vec3f(0.0f, 0.0f, 0.0f);
    segments[idx].indirectColor = Vec3f(0.0f, 0.0f, 0.0f);
    segments[idx].weight = Vec3f(1.0f, 1.0f, 1.0f);
    segments[idx].firstBounce = true; // Initialize first bounce flag
}

__device__ void AccumulateColor(int index, Vec3f color, float *accumulator)
{
    accumulator[index * 4 + 0] += color(0);
    accumulator[index * 4 + 1] += color(1);
    accumulator[index * 4 + 2] += color(2);
}

__device__ bool TraceRay(Ray &ray, BVH *bvh, IntersectionInfo &info)
{
    info.hitTime = FLOATMAX;
    info.hit = false;
    bool hit = bvh->IsIntersect(ray, info);
    return hit;
}

__global__ void IntersectionKernel(RenderSegment *segments, RenderParam renderParam, IntersectionInfo *intersectionInfo, int numSegments, int *segmentValidFlags, RenderBuffer renderBuffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSegments)
        return;
    BVH *bvh = renderParam.bvh;
    Ray ray = segments[idx].ray; // Get the ray for this pixel
    float *directLightingBuffer = renderBuffer.directLightingBuffer;
    float *indirectLightingBuffer = renderBuffer.indirectLightingBuffer;
    bool hit = TraceRay(ray, bvh, intersectionInfo[idx]);
    if(!hit){
        // If no hit, accumulate background color (black here)
        int index = segments[idx].index;
        LightManager *lightManager = renderParam.lightManager;
        if(lightManager -> envMapLightIdx != -1){
            Light *envLight = lightManager -> GetLight(lightManager -> envMapLightIdx);
            Vec3f bgColor = envLight -> Emission(ray.direction);
            // segments[idx].color += segments[idx].weight.cwiseProduct(bgColor);
            if(segments[idx].firstBounce){
                segments[idx].directColor += segments[idx].weight.cwiseProduct(bgColor);
            }else{
                segments[idx].indirectColor += segments[idx].weight.cwiseProduct(bgColor);
            }
        }
        AccumulateColor(index, segments[idx].directColor, directLightingBuffer);
        AccumulateColor(index, segments[idx].indirectColor, indirectLightingBuffer);
        segments[idx].remainingBounces = 0; // Terminate the path
        segmentValidFlags[idx] = 0;
    }else {
        segmentValidFlags[idx] = 1;
    }
}

__device__ float MISWeight(float a, float b, int c){
    float t1 = powf(a, c), t2 = powf(b, c);
    return t1 / (t1 + t2);
}

__device__ void PathTracing(RenderSegment &segment, IntersectionInfo &info, RenderParam renderParam, RenderBuffer renderBuffer, int *segmentValidFlags, int idx){
    Triangle *triangles = renderParam.triangles;
    MeshData *meshData = renderParam.meshData;
    Material *materials = renderParam.materials;
    LightManager *lightManager = renderParam.lightManager;
    BVH *bvh = renderParam.bvh;
    Sampler *sampler = renderParam.sampler;
    float rr = renderParam.rr;
    Bounds3D sceneBounds = renderParam.sceneBounds;

    Material &material = materials[meshData[info.meshID].materialID];
    Vec3f hitPoint = info.hitPoint;
    Vec3f hitNormal = info.normal;
    Vec2f uv = info.texCoord;
    Vec3f wo = -segment.ray.direction; // wo为p -> eye方向, wi为p -> light方向

    Vec3f brdf;

    if(material.IsLight()){
        if(segment.firstBounce){
            // segment.color += material.ke;
            segment.directColor+= material.ke;
            segment.firstBounce = false;
            segment.remainingBounces = 0;
            segmentValidFlags[idx] = 0;
            return ;
        }else{
            // MIS
            int lightID = meshData[info.meshID].lightID;
            // printf("Hit light ID: %d\n", lightID);
            Light *light = lightManager -> GetLight(lightID);
            float pdfLight;
            light -> SampleSolidAnglePDF(segment.ray, info.hitPoint, info.normal, pdfLight);

            // MIS weight
            float brdfMisWeight = MISWeight(segment.pdfBrdf, pdfLight, 2);
            segment.indirectColor += segment.weight.cwiseProduct(material.ke) * brdfMisWeight;
            // Vec3f temp = segment.weight.cwiseProduct(material.ke) * brdfMisWeight;
            // printf("temp light: %f, %f, %f\n", temp(0), temp(1), temp(2));
        }
    }

    // Sample light(NEE)
    float selectLightPDF;
    Light *light = lightManager -> SampleLight(sampler, idx, selectLightPDF);
    if(light != nullptr){
        float sampleLightPointPDF;
        TriangleSampleInfo sampleLightPointInfo;
        light -> SamplePoint(triangles, sceneBounds, sampleLightPointInfo, sampleLightPointPDF, sampler, idx);
        
        bool visible = light -> Visible(info, sampleLightPointInfo, bvh);
        // bool visible = true;
        if(visible){
            Vec3f dirWi = (sampleLightPointInfo.position - hitPoint).normalized();
            float pdfLight;
            Ray sampleLightRay;
            sampleLightRay.origin = hitPoint;
            sampleLightRay.direction = dirWi;
            light -> SampleSolidAnglePDF(sampleLightRay, sampleLightPointInfo.position, sampleLightPointInfo.normal, pdfLight);
            pdfLight *= selectLightPDF;
            float cosTheta = hitNormal.dot(dirWi);
            Vec3f dirBrdf = material.Evaluate(wo, hitNormal, dirWi);
            Vec3f dirL = light -> Emission(dirWi).cwiseProduct(dirBrdf) * cosTheta / pdfLight;

            // MIS weight
            float pdfBrdf;
            material.Pdf(wo, hitNormal, dirWi, pdfBrdf);
            float lightMisWeight = MISWeight(pdfLight, pdfBrdf, 2);
            // segment.color += lightMisWeight * segment.weight.cwiseProduct(dirL);
            if(segment.firstBounce){
                segment.directColor += lightMisWeight * segment.weight.cwiseProduct(dirL);
            }else{
                segment.indirectColor += lightMisWeight * segment.weight.cwiseProduct(dirL);
            }
            Vec3f temp = lightMisWeight * segment.weight.cwiseProduct(dirL);
        }
    }
    
    // return;

    // indir
    Vec3f indirWi;
    float indirWiPdf;
    material.Sample(wo, hitNormal, indirWi, indirWiPdf, sampler, idx);
    brdf = material.Evaluate(wo, hitNormal, indirWi);
    segment.pdfBrdf = indirWiPdf;
    segment.ray.origin = hitPoint; // Offset to avoid self-intersection
    segment.ray.direction = indirWi;

    // compute F in
    Vec3f FIn;
    material.Fresnel(wo, hitNormal, FIn);
    float Ftransmit = 1 - FIn(0);
    float rTransmission = sampler -> Get1D(idx);
    // printf("Ftransmit: %f, rTransmission: %f\n", Ftransmit, rTransmission);
    if(material.type != SUBSURFACE || rTransmission >= Ftransmit){
        float indirCosTheta = hitNormal.dot(indirWi);
        if(!(indirWiPdf < eps || indirCosTheta < 0)){
            segment.weight = segment.weight.cwiseProduct(brdf) * indirCosTheta / indirWiPdf;
        }else{
            segment.remainingBounces = 0;
            segmentValidFlags[idx] = 0;
            return ;
        }
    }else{
        // Subsurface scattering
        // Sample a point inside the surface
        Vec3f outgoingPoint, outgoingNormal, outgoingDir;
        float spatialPdf, outgoingDirPdf;
        material.SampleSpatialDiffusionProfile(wo, hitNormal, hitPoint, outgoingPoint, outgoingNormal, spatialPdf, sampler, idx, bvh);
        // update weight for subsurface
        segment.weight = segment.weight.cwiseProduct(material.basecolor);
        
        // NEE
        float selectLightPDF;
        Light *light = lightManager -> SampleLight(sampler, idx, selectLightPDF);
        if(light != nullptr){
            float sampleLightPointPDF;
            TriangleSampleInfo sampleLightPointInfo;
            light -> SamplePoint(triangles, sceneBounds, sampleLightPointInfo, sampleLightPointPDF, sampler, idx);
            IntersectionInfo tempInfo;
            tempInfo.hitPoint = outgoingPoint;
            tempInfo.normal = outgoingNormal;
            bool visible = light -> Visible(tempInfo, sampleLightPointInfo, bvh);
            if(visible){
                Vec3f dirWi = (sampleLightPointInfo.position - outgoingPoint).normalized();
                Vec3f FNee;
                material.Fresnel(dirWi, outgoingNormal, FNee);
                float FtransmitOut = 1 - FNee(0);
                float pdfLight;
                Ray sampleLightRay;
                sampleLightRay.origin = outgoingPoint;
                sampleLightRay.direction = dirWi;
                light -> SampleSolidAnglePDF(sampleLightRay, sampleLightPointInfo.position, sampleLightPointInfo.normal, pdfLight);
                pdfLight *= selectLightPDF;
                float cosTheta = outgoingNormal.dot(dirWi);
                Vec3f dirBrdf = material.Evaluate(dirWi, outgoingNormal, dirWi);
                Vec3f dirL = light -> Emission(dirWi).cwiseProduct(dirBrdf) * cosTheta / pdfLight;

                // MIS weight
                float pdfBrdf;
                material.Pdf(wo, outgoingNormal, dirWi, pdfBrdf);
                float lightMisWeight = MISWeight(pdfLight, pdfBrdf, 2);
                // segment.color += lightMisWeight * segment.weight.cwiseProduct(dirL) * FtransmitOut;
                if(segment.firstBounce){
                    segment.directColor += lightMisWeight * segment.weight.cwiseProduct(dirL) * FtransmitOut;
                }else{
                    segment.indirectColor += lightMisWeight * segment.weight.cwiseProduct(dirL) * FtransmitOut;
                }
            }
        }

        material.Sample(wo, outgoingNormal, outgoingDir, outgoingDirPdf, sampler, idx);
        // compute F out
        Vec3f FOut;
        material.Fresnel(outgoingDir, outgoingNormal, FOut);
        float FtransmitOut = 1 - FOut(0);

        // update weight and ray
        segment.ray.origin = outgoingPoint;
        segment.ray.direction = outgoingDir;
        segment.weight = segment.weight * FtransmitOut;
        segment.pdfBrdf = outgoingDirPdf;
    }
    

    // Russian roulette
    float p = sampler -> Get1D(idx);
    if(p > rr){
        segment.remainingBounces = 0; // Terminate the path
        segmentValidFlags[idx] = 0;
        return ;
    }

    segment.weight /= rr;
    

    segment.firstBounce = false; // After the first bounce
    segmentValidFlags[idx] = 1;

    // printf("here\n");
}

__global__ void ShadingKernel(RenderSegment *segments, RenderParam renderParam, IntersectionInfo *intersectionInfo, int numSegments, RenderBuffer renderBuffer, int *segmentValidFlags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSegments)
        return;

    IntersectionInfo info = intersectionInfo[idx];
    if(segments[idx].remainingBounces > 0){
        PathTracing(segments[idx], info, renderParam, renderBuffer, segmentValidFlags, idx);
    }
    if(segmentValidFlags[idx] == 0){
        AccumulateColor(segments[idx].index, segments[idx].directColor, renderBuffer.directLightingBuffer);
        AccumulateColor(segments[idx].index, segments[idx].indirectColor, renderBuffer.indirectLightingBuffer);
    }
        
    segments[idx].remainingBounces--;
}

__device__ bool GetLastFramePos(int index, CameraParam cameraParam, RenderParam renderParam, RenderBuffer currentBuffer, RenderBuffer lastBuffer, GBuffer currentGBuffer, GBuffer lastGBuffer, Vec2f &pixelPos){
    // Get world position from current frame's gbuffer
    int width = renderParam.width, height = renderParam.height;
    float4 positionTriID = tex2D<float4>(currentGBuffer.cudaTexObjPositionTriID, (index % width), index / width);
    ushort4 baryMeshID = tex2D<ushort4>(currentGBuffer.cudaTexObjBaryMeshID, (index % width), index / width);
    ushort4 normalMatID = tex2D<ushort4>(currentGBuffer.cudaTexObjNormalMatID, (index % width), index / width);
    float4 motionDepth = tex2D<float4>(currentGBuffer.cudaTexObjMotionDepth, (index % width), index / width);
    Vec3f position = Vec3f(positionTriID.x, positionTriID.y, positionTriID.z);
    int meshID = baryMeshID.w;
    Vec3f normal = Vec3f(normalMatID.x / 65535.0f * 2.f - 1.f, normalMatID.y / 65535.0f * 2.f - 1.f, normalMatID.z / 65535.0f * 2.f - 1.f);
    float depth = motionDepth.z;
    float dz = motionDepth.w;

    // if(position.norm() < eps && meshID == 0){
    //     pixelPos = Vec2f(-1, -1);
    //     return false;
    // }

    Vec2i motionVec = Vec2i((motionDepth.x) * renderParam.width * 0.5f, (motionDepth.y) * renderParam.height * 0.5f);
    Vec2f currentPixelPos = Vec2f(index % width, index / width);
    pixelPos = Vec2f(currentPixelPos(0) - motionVec(0), currentPixelPos(1) - motionVec(1));

    // WorldToLastFramePos(position, cameraParam, renderParam, pixelPos);

    if(pixelPos(0) < 0 || pixelPos(0) >= renderParam.width || pixelPos(1) < 0 || pixelPos(1) >= renderParam.height){
        // printf("index: %d, last frame uv: %f, %f\n", index, pixelPos(0), pixelPos(1));
        return false;
    }
    // printf("index: %d, current pixel: %f, %f, motion: %f, %f, last frame uv: %f, %f\n", index, currentPixelPos(0), currentPixelPos(1), motionVec(0), motionVec(1), pixelPos(0), pixelPos(1));
    // printf("index: %d, last frame uv: %f, %f, last frame index: %d\n", index, pixelPos(0), pixelPos(1), pixelPos(1) * renderParam.width + pixelPos(0));

    // get last frame gbuffer info
    float4 lastPositionTriID = tex2D<float4>(lastGBuffer.cudaTexObjPositionTriID, pixelPos(0), pixelPos(1));
    ushort4 lastBaryMeshID = tex2D<ushort4>(lastGBuffer.cudaTexObjBaryMeshID, pixelPos(0), pixelPos(1));
    ushort4 lastNormalMatID = tex2D<ushort4>(lastGBuffer.cudaTexObjNormalMatID, pixelPos(0), pixelPos(1));
    float4 lastMotionDepth = tex2D<float4>(lastGBuffer.cudaTexObjMotionDepth, pixelPos(0), pixelPos(1));
    int lastMeshID = lastBaryMeshID.w;
    Vec3f lastNormal = Vec3f(lastNormalMatID.x / 65535.0f * 2.f - 1.f, lastNormalMatID.y / 65535.0f * 2.f - 1.f, lastNormalMatID.z / 65535.0f * 2.f - 1.f);
    float lastDepth = lastMotionDepth.z;
    Vec3f lastPosition = Vec3f(lastPositionTriID.x, lastPositionTriID.y, lastPositionTriID.z);

    float dotNormal = normal.dot(lastNormal);
    // float depthDiff = fabs(depth - lastDepth);
    float depthDiff = fabs(position(2) - lastPosition(2));
    float normalThreshold = 0.9; // cos(18 degrees)
    float depthThreshold = dz * 6.f + (0.01f * depth); // 1% of depth

    // if(meshID != lastMeshID){
    //     // sample 3 * 3 neighborhood
    //     int dx[] = {-1, 0, 1};
    //     int dy[] = {-1, 0, 1};
    //     int sameCount = 0;
    //     for(int i = 0; i < 3; i++){
    //         for(int j = 0; j < 3; j++){
    //             Vec2f neighborPos = Vec2f(pixelPos(0) + dx[i], pixelPos(1) + dy[j]);
    //             if(neighborPos(0) < 0 || neighborPos(0) >= renderParam.width || neighborPos(1) < 0 || neighborPos(1) >= renderParam.height){
    //                 continue;
    //             }
    //             ushort4 neighborBaryMeshID = tex2D<ushort4>(lastGBuffer.cudaTexObjBaryMeshID, neighborPos(0), neighborPos(1));
    //             int neighborMeshID = neighborBaryMeshID.w;
    //             if(neighborMeshID == meshID){
    //                 sameCount++;
    //             }   
    //         }
    //     }
    //     if(sameCount < 4)
    //         return false;
    // }

    if(meshID != lastMeshID || dotNormal < normalThreshold || depthDiff > depthThreshold){
        // printf("current pixel: %f, %f, last pixel: %f, %f, motion vector: %f, %f\n", currentPixelPos(0), currentPixelPos(1), pixelPos(0), pixelPos(1), motionVec(0), motionVec(1));
        // printf("index: %d, meshID: %d, lastMeshID: %d, dotNormal: %f, depthDiff: %f\n", index, meshID, lastMeshID, dotNormal, depthDiff);
        // printf("index: %d, current normal: %f, %f, %f, last normal: %f, %f, %f\n", index, normal(0), normal(1), normal(2), lastNormal(0), lastNormal(1), lastNormal(2));
        return false;
    }

    return true;
}

__device__ void TemporalDenoising(int index, RenderBuffer currentBuffer, RenderBuffer lastBuffer, DenoiseParam denoiseParam, CameraParam cameraParam, RenderParam renderParam, GBuffer currentGBuffer, GBuffer lastGBuffer, bool firstFrame)
{
    if(firstFrame){
        // Initialize
        float directLuminance = Luminance(Vec3f(currentBuffer.directLightingBuffer[index * 4 + 0], currentBuffer.directLightingBuffer[index * 4 + 1], currentBuffer.directLightingBuffer[index * 4 + 2]));
        float indirectLuminance = Luminance(Vec3f(currentBuffer.indirectLightingBuffer[index * 4 + 0], currentBuffer.indirectLightingBuffer[index * 4 + 1], currentBuffer.indirectLightingBuffer[index * 4 + 2]));
        denoiseParam.directMoments[index] = make_float2(directLuminance, directLuminance * directLuminance);
        denoiseParam.indirectMoments[index] = make_float2(indirectLuminance, indirectLuminance * indirectLuminance);
        currentBuffer.directLightingBuffer[index * 4 + 3] = 1.f;
        currentBuffer.indirectLightingBuffer[index * 4 + 3] = 1.f;
        denoiseParam.historyLength[index] = 1;
        return;
    }

    // get last frame pixel position
    Vec2f pixelPos;
    bool valid = GetLastFramePos(index, cameraParam, renderParam, currentBuffer, lastBuffer, currentGBuffer, lastGBuffer, pixelPos);

    // if(valid){
    //     currentBuffer.directLightingBuffer[index * 4 + 0] = 1;
    //     currentBuffer.directLightingBuffer[index * 4 + 1] = 0;
    //     currentBuffer.directLightingBuffer[index * 4 + 2] = 0;
    //     currentBuffer.indirectLightingBuffer[index * 4 + 0] = 0;
    //     currentBuffer.indirectLightingBuffer[index * 4 + 1] = 0;
    //     currentBuffer.indirectLightingBuffer[index * 4 + 2] = 0;
    // }else{
    //     currentBuffer.directLightingBuffer[index * 4 + 0] = 0;
    //     currentBuffer.directLightingBuffer[index * 4 + 1] = 1;
    //     currentBuffer.directLightingBuffer[index * 4 + 2] = 0;
    //     currentBuffer.indirectLightingBuffer[index * 4 + 0] = 0;
    //     currentBuffer.indirectLightingBuffer[index * 4 + 1] = 0;
    //     currentBuffer.indirectLightingBuffer[index * 4 + 2] = 0;
    // }
    // return;

    denoiseParam.historyLength[index] = valid ? min(denoiseParam.historyLength[index], denoiseParam.maxHistoryLength) : 1;
    float alpha = valid ? 1.f / denoiseParam.historyLength[index] : 1.0f;
    int lastFrameIndex = valid ? (int)pixelPos(1) * renderParam.width + (int)pixelPos(0) : index;

    currentBuffer.directLightingBuffer[index * 4 + 0] = alpha * currentBuffer.directLightingBuffer[index * 4 + 0] + (1 - alpha) * lastBuffer.directLightingBuffer[lastFrameIndex * 4 + 0];
    currentBuffer.directLightingBuffer[index * 4 + 1] = alpha * currentBuffer.directLightingBuffer[index * 4 + 1] + (1 - alpha) * lastBuffer.directLightingBuffer[lastFrameIndex * 4 + 1];
    currentBuffer.directLightingBuffer[index * 4 + 2] = alpha * currentBuffer.directLightingBuffer[index * 4 + 2] + (1 - alpha) * lastBuffer.directLightingBuffer[lastFrameIndex * 4 + 2];
    currentBuffer.indirectLightingBuffer[index * 4 + 0] = alpha * currentBuffer.indirectLightingBuffer[index * 4 + 0] + (1 - alpha) * lastBuffer.indirectLightingBuffer[lastFrameIndex * 4 + 0];
    currentBuffer.indirectLightingBuffer[index * 4 + 1] = alpha * currentBuffer.indirectLightingBuffer[index * 4 + 1] + (1 - alpha) * lastBuffer.indirectLightingBuffer[lastFrameIndex * 4 + 1];
    currentBuffer.indirectLightingBuffer[index * 4 + 2] = alpha * currentBuffer.indirectLightingBuffer[index * 4 + 2] + (1 - alpha) * lastBuffer.indirectLightingBuffer[lastFrameIndex * 4 + 2];

    // update moments
    // float2 &moment = denoiseParam.moments[index];
    // float2 lastMoment = valid ? denoiseParam.moments[lastFrameIndex] : make_float2(0.f, 0.f);
    // float luminance = Luminance(Vec3f(currentBuffer.directLightingBuffer[index * 3 + 0] + currentBuffer.indirectLightingBuffer[index * 3 + 0], currentBuffer.directLightingBuffer[index * 3 + 1] + currentBuffer.indirectLightingBuffer[index * 3 + 1], currentBuffer.directLightingBuffer[index * 3 + 2] + currentBuffer.indirectLightingBuffer[index * 3 + 2]));
    // moment.x = alpha * luminance + (1 - alpha) * lastMoment.x;
    // moment.y = alpha * luminance * luminance + (1 - alpha) * lastMoment.y;
    float2 &directMoment = denoiseParam.directMoments[index];
    float2 &indirectMoment = denoiseParam.indirectMoments[index];
    float2 lastDirectMoment = valid ? denoiseParam.directMoments[lastFrameIndex] : make_float2(0.f, 0.f);
    float2 lastIndirectMoment = valid ? denoiseParam.indirectMoments[lastFrameIndex] : make_float2(0.f, 0.f);
    float directLuminance = Luminance(Vec3f(currentBuffer.directLightingBuffer[index * 4 + 0], currentBuffer.directLightingBuffer[index * 4 + 1], currentBuffer.directLightingBuffer[index * 4 + 2]));
    float indirectLuminance = Luminance(Vec3f(currentBuffer.indirectLightingBuffer[index * 4 + 0], currentBuffer.indirectLightingBuffer[index * 4 + 1], currentBuffer.indirectLightingBuffer[index * 4 + 2]));
    directMoment.x = alpha * directLuminance + (1 - alpha) * lastDirectMoment.x;
    directMoment.y = alpha * directLuminance * directLuminance + (1 - alpha) * lastDirectMoment.y;
    indirectMoment.x = alpha * indirectLuminance + (1 - alpha) * lastIndirectMoment.x;
    indirectMoment.y = alpha * indirectLuminance * indirectLuminance + (1 - alpha) * lastIndirectMoment.y;
    if(valid){
        // denoiseParam.variance[index] = fmaxf(1e-6f, moment.y - moment.x * moment.x);
        currentBuffer.directLightingBuffer[index * 4 + 3] = directMoment.y - directMoment.x * directMoment.x;
        currentBuffer.indirectLightingBuffer[index * 4 + 3] = indirectMoment.y - indirectMoment.x * indirectMoment.x;
        denoiseParam.historyLength[index] = min(denoiseParam.historyLength[index] + 1, denoiseParam.maxHistoryLength);
    }else{
        // denoiseParam.variance[index] = 1.f;
        // denoiseParam.historyLength[index] = 1;
        currentBuffer.directLightingBuffer[index * 4 + 3] = 1.f;
        currentBuffer.indirectLightingBuffer[index * 4 + 3] = 1.f;
        denoiseParam.historyLength[index] = 1;
    }
    
}

__global__ void TemporalDenoiseKernel(DenoiseParam denoiseParam, CameraParam cameraParam, RenderParam renderParam, RenderBuffer currentBuffer, RenderBuffer lastBuffer, GBuffer currentGBuffer, GBuffer lastGBuffer, bool firstFrame)
{
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    j = height - j - 1; // flip y
    if (i >= width || j >= height)
        return;
    int idx = j * width + i;
    TemporalDenoising(idx, currentBuffer, lastBuffer, denoiseParam, cameraParam, renderParam, currentGBuffer, lastGBuffer, firstFrame);
}

__global__ void ComputeVarianceKernel(DenoiseParam denoiseParam, RenderParam renderParam, RenderBuffer currentBuffer, RenderBuffer lastRenderBuffer, GBuffer gBuffer, int kernelSize){
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    int idx = j * width + i;
    int historyLength = denoiseParam.historyLength[idx];
    if(historyLength < 4){
        // 获取中心像素信息
        Vec3f centerDirectColor = Vec3f(currentBuffer.directLightingBuffer[4 * idx + 0], currentBuffer.directLightingBuffer[4 * idx + 1], currentBuffer.directLightingBuffer[4 * idx + 2]);
        Vec3f centerIndirectColor = Vec3f(currentBuffer.indirectLightingBuffer[4 * idx + 0], currentBuffer.indirectLightingBuffer[4 * idx + 1], currentBuffer.indirectLightingBuffer[4 * idx + 2]);
        
        float centerDirectLuminance = Luminance(centerDirectColor);
        float centerIndirectLuminance = Luminance(centerIndirectColor);

        float4 motionDepth = tex2D<float4>(gBuffer.cudaTexObjMotionDepth, i, j);
        float centerDepth = motionDepth.z;
        float dz = fmaxf(motionDepth.w, 1e-4f);

        ushort4 normalMatID = tex2D<ushort4>(gBuffer.cudaTexObjNormalMatID, i, j);
        Vec3f centerNormal = Vec3f(normalMatID.x / 65535.0f * 2.f - 1.f, normalMatID.y / 65535.0f * 2.f - 1.f, normalMatID.z / 65535.0f * 2.f - 1.f).normalized();

        float centerDirectVariance = fmaxf(0.0f, currentBuffer.directLightingBuffer[idx * 4 + 3]);
        float centerIndirectVariance = fmaxf(0.0f, currentBuffer.indirectLightingBuffer[idx * 4 + 3]);
        float stdDevDirect = sqrtf(centerDirectVariance) + 1e-3f; 
        float stdDevIndirect = sqrtf(centerIndirectVariance) + 1e-3f; 

        // 累加器
        Vec3f finalDirectColor(0.f, 0.f, 0.f);
        Vec3f finalIndirectColor(0.f, 0.f, 0.f);
        Vec2f finalDirectMoments(0.f, 0.f);
        Vec2f finalIndirectMoments(0.f, 0.f);
        float totalDirectWeight = 0.f;
        float totalIndirectWeight = 0.f;

        int halfKernel = kernelSize / 2;

        float sigmaDepth = 3 * fmaxf(dz, 1e-4f);
        for(int kj = -halfKernel; kj <= halfKernel; kj++){
            for(int ki = -halfKernel; ki <= halfKernel; ki++){
                int ni = i + ki;
                int nj = j + kj;
                if(ni >= 0 && ni < width && nj >= 0 && nj < height){
                    int nidx = nj * width + ni;
                    // 获取邻居像素信息
                    Vec3f nDirectColor = Vec3f(currentBuffer.directLightingBuffer[4 * nidx + 0], currentBuffer.directLightingBuffer[4 * nidx + 1], currentBuffer.directLightingBuffer[4 * nidx + 2]);
                    Vec3f nIndirectColor = Vec3f(currentBuffer.indirectLightingBuffer[4 * nidx + 0], currentBuffer.indirectLightingBuffer[4 * nidx + 1], currentBuffer.indirectLightingBuffer[4 * nidx + 2]);
                    float4 nMotionDepth = tex2D<float4>(gBuffer.cudaTexObjMotionDepth, ni, nj);
                    float nDepth = nMotionDepth.z;
                    ushort4 nNormalMatID = tex2D<ushort4>(gBuffer.cudaTexObjNormalMatID, ni, nj);
                    Vec3f nNormal = Vec3f(nNormalMatID.x / 65535.0f * 2.f - 1.f, nNormalMatID.y / 65535.0f * 2.f - 1.f, nNormalMatID.z / 65535.0f * 2.f - 1.f).normalized();
                    float nDirectLuminance = Luminance(nDirectColor);
                    float nIndirectLuminance = Luminance(nIndirectColor);
                    Vec2f nDirectMoment = Vec2f(denoiseParam.directMoments[nidx].x, denoiseParam.directMoments[nidx].y);
                    Vec2f nIndirectMoment = Vec2f(denoiseParam.indirectMoments[nidx].x, denoiseParam.indirectMoments[nidx].y);

                    // 计算权重
                    float dist = sqrtf(float(ki * ki + kj * kj));
                    float wz = -fabsf(centerDepth - nDepth) / (sigmaDepth * dist + 1e-4f);
                    // printf("idx: %d, nidx: %d, centerDepth: %f, nDepth: %f, dz: %f, wz: %f\n", idx, nidx, centerDepth, nDepth, dz, wz);
                    float wn = powf(fmaxf(0.f, centerNormal.dot(nNormal)), denoiseParam.sigmaNormal);
                    float wDirect = -fabsf(centerDirectLuminance - nDirectLuminance) / (denoiseParam.sigmaLight);
                    float wIndirect = -fabsf(centerIndirectLuminance - nIndirectLuminance) / (denoiseParam.sigmaLight);

                    float dWeight = expf(wz + wDirect) * wn;
                    float iWeight = expf(wz + wIndirect) * wn;

                    finalDirectColor += nDirectColor * dWeight;
                    finalIndirectColor += nIndirectColor * iWeight;
                    finalDirectMoments += nDirectMoment * dWeight;
                    finalIndirectMoments += nIndirectMoment * iWeight;
                    totalDirectWeight += dWeight;
                    totalIndirectWeight += iWeight;
                }
            }
        }
        // 归一化
        totalDirectWeight = fmaxf(totalDirectWeight, 1e-6f);
        totalIndirectWeight = fmaxf(totalIndirectWeight, 1e-6f);
        finalDirectMoments /= totalDirectWeight;
        finalIndirectMoments /= totalIndirectWeight;
        finalDirectColor /= totalDirectWeight;
        finalIndirectColor /= totalIndirectWeight;

        float directVariance = fmaxf(0.f, finalDirectMoments(1) - finalDirectMoments(0) * finalDirectMoments(0));
        float indirectVariance = fmaxf(0.f, finalIndirectMoments(1) - finalIndirectMoments(0) * finalIndirectMoments(0));
        directVariance *= 4.f / historyLength;
        indirectVariance *= 4.f / historyLength;

        lastRenderBuffer.directLightingBuffer[idx * 4 + 0] = finalDirectColor(0);
        lastRenderBuffer.directLightingBuffer[idx * 4 + 1] = finalDirectColor(1);
        lastRenderBuffer.directLightingBuffer[idx * 4 + 2] = finalDirectColor(2);
        lastRenderBuffer.directLightingBuffer[idx * 4 + 3] = directVariance;
        lastRenderBuffer.indirectLightingBuffer[idx * 4 + 0] = finalIndirectColor(0);
        lastRenderBuffer.indirectLightingBuffer[idx * 4 + 1] = finalIndirectColor(1);
        lastRenderBuffer.indirectLightingBuffer[idx * 4 + 2] = finalIndirectColor(2);
        lastRenderBuffer.indirectLightingBuffer[idx * 4 + 3] = indirectVariance;
        
        // printf("idx: %d, history length: %d, direct variance: %f, indirect variance: %f, finalDirectMoments: %f, %f, finalIndirectMoments: %f, %f\n", idx, historyLength, directVariance, indirectVariance, finalDirectMoments(0), finalDirectMoments(1), finalIndirectMoments(0), finalIndirectMoments(1));
    }else{
        // 直接复制
        lastRenderBuffer.directLightingBuffer[idx * 4 + 0] = currentBuffer.directLightingBuffer[idx * 4 + 0];
        lastRenderBuffer.directLightingBuffer[idx * 4 + 1] = currentBuffer.directLightingBuffer[idx * 4 + 1];
        lastRenderBuffer.directLightingBuffer[idx * 4 + 2] = currentBuffer.directLightingBuffer[idx * 4 + 2];
        lastRenderBuffer.directLightingBuffer[idx * 4 + 3] = currentBuffer.directLightingBuffer[idx * 4 + 3];
        lastRenderBuffer.indirectLightingBuffer[idx * 4 + 0] = currentBuffer.indirectLightingBuffer[idx * 4 + 0];
        lastRenderBuffer.indirectLightingBuffer[idx * 4 + 1] = currentBuffer.indirectLightingBuffer[idx * 4 + 1];
        lastRenderBuffer.indirectLightingBuffer[idx * 4 + 2] = currentBuffer.indirectLightingBuffer[idx * 4 + 2];
        lastRenderBuffer.indirectLightingBuffer[idx * 4 + 3] = currentBuffer.indirectLightingBuffer[idx * 4 + 3];
    }
    
}

__global__ void AtrousKernel(DenoiseParam denoiseParam, CameraParam cameraParam, RenderParam renderParam, RenderBuffer renderBuffer, RenderBuffer lastRenderBuffer, GBuffer gBuffer, int stepSize) {
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int idx = j * width + i;

    Vec3f centerDirectColor = Vec3f(renderBuffer.directLightingBuffer[4 * idx + 0], renderBuffer.directLightingBuffer[4 * idx + 1], renderBuffer.directLightingBuffer[4 * idx + 2]);
    Vec3f centerIndirectColor = Vec3f(renderBuffer.indirectLightingBuffer[4 * idx + 0], renderBuffer.indirectLightingBuffer[4 * idx + 1], renderBuffer.indirectLightingBuffer[4 * idx + 2]);
    
    float centerDirectLuminance = Luminance(centerDirectColor);
    float centerIndirectLuminance = Luminance(centerIndirectColor);

    float4 motionDepth = tex2D<float4>(gBuffer.cudaTexObjMotionDepth, i, j);
    float centerDepth = motionDepth.z;
    float dz = fmaxf(motionDepth.w, 1e-4f); 

    ushort4 normalMatID = tex2D<ushort4>(gBuffer.cudaTexObjNormalMatID, i, j);
    Vec3f centerNormal = Vec3f(normalMatID.x / 65535.0f * 2.f - 1.f, normalMatID.y / 65535.0f * 2.f - 1.f, normalMatID.z / 65535.0f * 2.f - 1.f).normalized();

    // 核心：使用预滤波后的方差，并增加保护项
    float centerDirectVariance = fmaxf(0.0f, renderBuffer.directLightingBuffer[idx * 4 + 3]);
    float centerIndirectVariance = fmaxf(0.0f, renderBuffer.indirectLightingBuffer[idx * 4 + 3]);
    // printf("idx: %d, direct variance: %f, indirect variance: %f\n", idx, centerDirectVariance, centerIndirectVariance);
    float stdDevDirect = sqrtf(centerDirectVariance) + 1e-3f; 
    float stdDevIndirect = sqrtf(centerIndirectVariance) + 1e-3f; 

    // 累加器
    Vec3f finalDirectColor(0.f, 0.f, 0.f);
    Vec3f finalIndirectColor(0.f, 0.f, 0.f);
    float totalDirectWeight = 0.f;
    float totalIndirectWeight = 0.f;
    float directVariance = 0.f;
    float indirectVariance = 0.f;

    float sigmaDepth = 1000 * fmaxf(dz, 1e-4f);
    
    const float kernelWeights[3] = {3.f / 8.f, 1.f / 4.f, 1.f / 16.f}; // 5x5 kernel weights

    for(int kj = -2; kj <= 2; kj++){
        for(int ki = -2; ki <= 2; ki++){
            int ni = i + ki * stepSize;
            int nj = j + kj * stepSize;

            if(ni >= 0 && ni < width && nj >= 0 && nj < height){
                int nidx = nj * width + ni;

                // 采样邻域信息
                float4 nMotionDepth = tex2D<float4>(gBuffer.cudaTexObjMotionDepth, ni, nj);
                float nDepth = nMotionDepth.z;

                ushort4 nNormalMatID = tex2D<ushort4>(gBuffer.cudaTexObjNormalMatID, ni, nj);
                Vec3f nNormal = Vec3f(nNormalMatID.x / 65535.0f * 2.f - 1.f, nNormalMatID.y / 65535.0f * 2.f - 1.f, nNormalMatID.z / 65535.0f * 2.f - 1.f).normalized();

                Vec3f nDirectColor = Vec3f(renderBuffer.directLightingBuffer[4 * nidx + 0], renderBuffer.directLightingBuffer[4 * nidx + 1], renderBuffer.directLightingBuffer[4 * nidx + 2]);
                float nDirectVariance = fmaxf(0.0f, renderBuffer.directLightingBuffer[nidx * 4 + 3]);
                Vec3f nIndirectColor = Vec3f(renderBuffer.indirectLightingBuffer[4 * nidx + 0], renderBuffer.indirectLightingBuffer[4 * nidx + 1], renderBuffer.indirectLightingBuffer[4 * nidx + 2]);
                float nIndirectVariance = fmaxf(0.0f, renderBuffer.indirectLightingBuffer[nidx * 4 + 3]);

                float dist = sqrtf(float(ki * ki + kj * kj)) * stepSize;
                float wz = -fabsf(centerDepth - nDepth) / (sigmaDepth * dist + 1e-6f);
                float wn = powf(fmaxf(0.f, centerNormal.dot(nNormal)), denoiseParam.sigmaNormal);
                float directLuminance = Luminance(nDirectColor);
                float indirectLuminance = Luminance(nIndirectColor);
                float wDirect = -fabsf(centerDirectLuminance - directLuminance) / (denoiseParam.sigmaLight * stdDevDirect);
                float wIndirect = -fabsf(centerIndirectLuminance - indirectLuminance) / (denoiseParam.sigmaLight * stdDevIndirect);

                float kWeight = kernelWeights[abs(ki)] * kernelWeights[abs(kj)];
                float dWeight = expf(wz + wDirect) * wn * kWeight;
                float iWeight = expf(wz + wIndirect) * wn * kWeight;

                finalDirectColor += dWeight * nDirectColor;
                finalIndirectColor += iWeight * nIndirectColor;
                totalDirectWeight += dWeight;
                totalIndirectWeight += iWeight;
                directVariance += dWeight * dWeight * nDirectVariance;
                indirectVariance += iWeight * iWeight * nIndirectVariance;
            }
        }
    }

    // 5. 归一化输出
    if(totalDirectWeight > 1e-6f) {
        lastRenderBuffer.directLightingBuffer[4 * idx + 0] = finalDirectColor(0) / totalDirectWeight;
        lastRenderBuffer.directLightingBuffer[4 * idx + 1] = finalDirectColor(1) / totalDirectWeight;
        lastRenderBuffer.directLightingBuffer[4 * idx + 2] = finalDirectColor(2) / totalDirectWeight;
        float newVariance = directVariance / (totalDirectWeight * totalDirectWeight);
        // printf("idx: %d, direct variance before: %f, after: %f\n", idx, centerDirectVariance, newVariance);
        lastRenderBuffer.directLightingBuffer[4 * idx + 3] = directVariance / (totalDirectWeight * totalDirectWeight);
    } else {
        lastRenderBuffer.directLightingBuffer[4 * idx + 0] = centerDirectColor(0);
        lastRenderBuffer.directLightingBuffer[4 * idx + 1] = centerDirectColor(1);
        lastRenderBuffer.directLightingBuffer[4 * idx + 2] = centerDirectColor(2);
        lastRenderBuffer.directLightingBuffer[4 * idx + 3] = 1;
    }

    if(totalIndirectWeight > 1e-6f) {
        lastRenderBuffer.indirectLightingBuffer[4 * idx + 0] = finalIndirectColor(0) / totalIndirectWeight;
        lastRenderBuffer.indirectLightingBuffer[4 * idx + 1] = finalIndirectColor(1) / totalIndirectWeight;
        lastRenderBuffer.indirectLightingBuffer[4 * idx + 2] = finalIndirectColor(2) / totalIndirectWeight;
        lastRenderBuffer.indirectLightingBuffer[4 * idx + 3] = indirectVariance / (totalIndirectWeight * totalIndirectWeight);
    } else {
        lastRenderBuffer.indirectLightingBuffer[4 * idx + 0] = centerIndirectColor(0);
        lastRenderBuffer.indirectLightingBuffer[4 * idx + 1] = centerIndirectColor(1);
        lastRenderBuffer.indirectLightingBuffer[4 * idx + 2] = centerIndirectColor(2);
        lastRenderBuffer.indirectLightingBuffer[4 * idx + 3] = 1;
    }
}

__global__ void GatherKernel(uchar4 *pixels, RenderBuffer renderBuffer, GBuffer gBuffer, int spp, RenderParam renderParam, CameraParam cameraParam, DenoiseParam denoiseParam)
{
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return; 
    int idx = j * width + i;
    j = height - j - 1; // Flip vertically for OpenGL
    int segIndex = j * width + i;
    int renderTargetMode = renderParam.renderTargetMode;
    float *directLightingBuffer = renderBuffer.directLightingBuffer;
    float *indirectLightingBuffer = renderBuffer.indirectLightingBuffer;
    if(renderTargetMode == 0){
        // color
        if(renderParam.denoise){
            Vec3f color = Vec3f(directLightingBuffer[4 * idx] + indirectLightingBuffer[4 * idx],
                                directLightingBuffer[4 * idx + 1] + indirectLightingBuffer[4 * idx + 1],
                                directLightingBuffer[4 * idx + 2] + indirectLightingBuffer[4 * idx + 2]);
            pixels[segIndex] = make_uchar4(
                static_cast<unsigned char>(fminf(fmaxf(powf(color(0), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                static_cast<unsigned char>(fminf(fmaxf(powf(color(1), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                static_cast<unsigned char>(fminf(fmaxf(powf(color(2), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                255);
        }else{
            // using accumulated framebuffer
            Vec3f color = Vec3f((directLightingBuffer[4 * idx] + indirectLightingBuffer[4 * idx]) / spp, (directLightingBuffer[4 * idx + 1] + indirectLightingBuffer[4 * idx + 1]) / spp, (directLightingBuffer[4 * idx + 2] + indirectLightingBuffer[4 * idx + 2]) / spp);
            pixels[segIndex] = make_uchar4(
                static_cast<unsigned char>(fminf(fmaxf(powf(color(0), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                static_cast<unsigned char>(fminf(fmaxf(powf(color(1), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                static_cast<unsigned char>(fminf(fmaxf(powf(color(2), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                255);
        }
       
    }else if(renderTargetMode == 1){
        // depth
        float4 motionDepth = tex2D<float4>(gBuffer.cudaTexObjMotionDepth, i + 0.5f, j + 0.5f);
        float nearPlane = 0.1f;
        float farPlane = renderParam.sceneBounds.DiagonalLength() + (cameraParam.position - renderParam.sceneBounds.Center()).norm();
        float depth = (motionDepth.z - nearPlane) / (farPlane - nearPlane);
        unsigned char depthUC = static_cast<unsigned char>(fminf(fmaxf(depth * 255.0f, 0.0f), 255.0f));
        pixels[idx] = make_uchar4(depthUC, depthUC, depthUC, 255);
    }else if(renderTargetMode == 2){
        // normal
        ushort4 normalMatID = tex2D<ushort4>(gBuffer.cudaTexObjNormalMatID, i + 0.5f, j + 0.5f);
        float nx = normalMatID.x / 65535.0f;
        float ny = normalMatID.y / 65535.0f;
        float nz = normalMatID.z / 65535.0f;
        pixels[idx] = make_uchar4(static_cast<unsigned char>(fminf(fmaxf(nx * 255.0f, 0.0f), 255.0f)),
                                       static_cast<unsigned char>(fminf(fmaxf(ny * 255.0f, 0.0f), 255.0f)),
                                       static_cast<unsigned char>(fminf(fmaxf(nz * 255.0f, 0.0f), 255.0f)),
                                       255);
    }else if(renderTargetMode == 3){
        // id
        ushort4 baryMeshID = tex2D<ushort4>(gBuffer.cudaTexObjBaryMeshID, i + 0.5f, j + 0.5f);
        unsigned short id = baryMeshID.w;
        unsigned char r = (id * 37) % 256;
        unsigned char g = (id * 57) % 256;
        unsigned char b = (id * 97) % 256;
        pixels[idx] = make_uchar4(r, g, b, 255);
    }else if(renderTargetMode == 4){
        // position
        float4 positionTriID = tex2D<float4>(gBuffer.cudaTexObjPositionTriID, i + 0.5f, j + 0.5f);
        float px = positionTriID.x;
        float py = positionTriID.y;
        float pz = positionTriID.z;
        Vec3f minPos = renderParam.sceneBounds.min;
        Vec3f maxPos = renderParam.sceneBounds.max;
        pixels[idx] = make_uchar4(
            static_cast<unsigned char>(fminf(fmaxf((px - minPos(0)) / (maxPos(0) - minPos(0)) * 255.0f, 0.0f), 255.0f)),
            static_cast<unsigned char>(fminf(fmaxf((py - minPos(1)) / (maxPos(1) - minPos(1)) * 255.0f, 0.0f), 255.0f)),
            static_cast<unsigned char>(fminf(fmaxf((pz - minPos(2)) / (maxPos(2) - minPos(2)) * 255.0f, 0.0f), 255.0f)),
            255);
    }
}


int StreamCompaction(int* rayValid, int* rayIndex, int numRays)
{
	if (numRays == 0) return 0;
	thrust::device_ptr<RenderSegment> segmentPtr(renderSegments), segmentBufferPtr(renderSegmentsBuffer);
	thrust::device_ptr<IntersectionInfo> intersectionInfoPtr(intersections), intersectionInfoBufferPtr(intersectionsBuffer);
	thrust::device_ptr<int> rayValidPtr(rayValid), rayIndexPtr(rayIndex);
	thrust::exclusive_scan(rayValidPtr, rayValidPtr + numRays, rayIndexPtr);
	int nextNumRays, tmp;
	cudaMemcpy(&tmp, rayIndex + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays = tmp;
	cudaMemcpy(&tmp, rayValid + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays += tmp;
	thrust::scatter_if(segmentPtr, segmentPtr + numRays, rayIndexPtr, rayValidPtr, segmentBufferPtr);
	thrust::scatter_if(intersectionInfoPtr, intersectionInfoPtr + numRays, rayIndexPtr, rayValidPtr, intersectionInfoBufferPtr);
	std::swap(renderSegments, renderSegmentsBuffer);
	std::swap(intersections, intersectionsBuffer);
	return nextNumRays;
}

__global__ void MaterialIDGetterKernel(MeshData *meshes, IntersectionInfo* intersectionInfo, int* materialIDs, int numSegments){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSegments)
        return;
    int meshID = intersectionInfo[idx].meshID;
    MeshData *mesh = meshes + meshID;
    materialIDs[idx] = mesh->materialID;
}

__global__ void GatherMaterialSortResult(RenderSegment *segments, RenderSegment * sortedSegments, int numSegments, IntersectionInfo *intersections, IntersectionInfo * sortedIntersections, int *materialIDs, int *materialSortIndices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSegments)
        return;
    int sortedIdx = materialSortIndices[idx];
    sortedSegments[sortedIdx] = segments[idx];
    sortedIntersections[sortedIdx] = intersections[idx];
}

void SortByMaterialID(RenderSegment *segments, RenderSegment * sortedSegments, int numSegments, IntersectionInfo *intersections, IntersectionInfo * sortedIntersections, int *materialIDs, int *materialSortIndices)
{
    thrust::sequence(thrust::device, materialSortIndices, materialSortIndices + numSegments, 0);
    thrust::device_ptr<RenderSegment> segmentPtr(segments);
    thrust::device_ptr<IntersectionInfo> intersectionInfoPtr(intersections);
    // get material IDs
    int blockSize = 128;
    dim3 gridSize((numSegments + blockSize - 1) / blockSize);
    MaterialIDGetterKernel<<<gridSize, blockSize>>>(renderParam.meshData, intersections, materialIDs, numSegments);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
    thrust::device_ptr<int> materialIDPtr(materialIDs);
    // sort by material IDs
    thrust::sort_by_key(materialIDPtr, materialIDPtr + numSegments, materialSortIndices);
    // gather the segments and intersection info
    GatherMaterialSortResult<<<gridSize, blockSize>>>(segments, sortedSegments, numSegments, intersections, sortedIntersections, materialIDs, materialSortIndices);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
    std::swap(segments, sortedSegments);
    std::swap(intersections, sortedIntersections);
}

class Renderer{
public:
    __host__ Renderer(UI *ui){
        this -> ui = ui;
        sppCounter = 0;

        CudaInit();
        GBufferInit();
    }
    __host__ ~Renderer(){

    }

    __host__ void RenderLoop(){
        bool firstFrame = true;
        while (!glfwWindowShouldClose(ui -> window))
        {
            // LOG_DEBUG("here");
            RenderBuffer renderBuffer = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[0] : renderBuffers[1];
            RenderBuffer lastRenderBuffer = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[1] : renderBuffers[0];
            GBuffer gBuffer = renderParam.currentGBufferIndex == 0 ? gBuffers[0] : gBuffers[1];
            GBuffer lastGBuffer = renderParam.currentGBufferIndex == 0 ? gBuffers[1] : gBuffers[0];
            // Start the Dear ImGui frame
            ui -> GuiBegin(sppCounter, framebufferReset);
            if(framebufferReset){
                sppCounter = 0;
                cudaMemset(renderBuffer.directLightingBuffer, 0, sizeof(float) * 4 * renderParam.width * renderParam.height);
                cudaMemset(renderBuffer.indirectLightingBuffer, 0, sizeof(float) * 4 * renderParam.width * renderParam.height);
                framebufferReset = false;
            }

            if(sppCounter < renderParam.spp || renderParam.denoise){
                RenderGBuffer(renderParam, cameraParam);
                sppCounter++;

                // Rendering pipeline start
                int width = renderParam.width, height = renderParam.height;
                int maxBounces = renderParam.maxBounces;
                int numSegments = width * height;
                // 1. Generate rays
                generateRayGridSize = dim3((width + generateRayBlockSize.x - 1) / generateRayBlockSize.x, (height + generateRayBlockSize.y - 1) / generateRayBlockSize.y);
                GenerateRayKernel<<<generateRayGridSize, generateRayBlockSize>>>(renderSegments, renderParam, cameraParam);
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());

                renderGridSize = dim3((numSegments + renderBlockSize.x - 1) / renderBlockSize.x);

                for(int i = 0; i < maxBounces; i ++){
                    if(numSegments <= 0) break;
                    cudaMemset(intersections, 0, sizeof(IntersectionInfo) * numSegments); // Clear intersection info for next bounce
                    cudaMemset(intersectionsBuffer, 0, sizeof(IntersectionInfo) * numSegments); // Clear intersection info for next bounce
                    cudaMemset(materialIDs, 0, sizeof(int) * numSegments); // Clear material IDs buffer
                    // printf("Bounce %d, Active segments: %d\n", i, numSegments);

                    // 2. Intersection
                    renderGridSize = dim3((numSegments + renderBlockSize.x - 1) / renderBlockSize.x);
                    IntersectionKernel<<<renderGridSize, renderBlockSize>>>(renderSegments, renderParam, intersections, numSegments, segmentValidFlags, renderBuffer);
                    cudaDeviceSynchronize();
                    CHECK_CUDA_ERROR(cudaGetLastError());

                    // stream compaction
                    numSegments = StreamCompaction(segmentValidFlags, segmentPos, numSegments);
                    // printf("After intersection stream compaction, Active segments: %d\n", numSegments);
                    if(numSegments <= 0)break;

                    // todo: sort intersections by materialID to improve memory coherence
                    // SortByMaterialID(renderSegments, renderSegmentsBuffer, numSegments, intersections, intersectionsBuffer, materialIDs, materialSortIndices);

                    // 3. Shading
                    renderGridSize = dim3((numSegments + renderBlockSize.x - 1) / renderBlockSize.x);
                    ShadingKernel<<<renderGridSize, renderBlockSize>>>(renderSegments, renderParam, intersections, numSegments, renderBuffer, segmentValidFlags);
                    cudaDeviceSynchronize();
                    CHECK_CUDA_ERROR(cudaGetLastError());

                    // stream compaction
                    numSegments = StreamCompaction(segmentValidFlags, segmentPos, numSegments);
                    // printf("After shading stream compaction, Active segments: %d\n", numSegments);
                }
                
                // map cuda resource
                // GBuffer &gBuffer = gBuffers[renderParam.currentBufferIndex];
                cudaGraphicsMapResources(1, &gBuffer.cudaPositionTriIDResource, 0);
                cudaGraphicsMapResources(1, &gBuffer.cudaNormalMatIDResource, 0);
                cudaGraphicsMapResources(1, &gBuffer.cudaBaryMeshIDResource, 0);
                cudaGraphicsMapResources(1, &gBuffer.cudaMotionDepthResource, 0);

                cudaGraphicsMapResources(1, &lastGBuffer.cudaPositionTriIDResource, 0);
                cudaGraphicsMapResources(1, &lastGBuffer.cudaNormalMatIDResource, 0);
                cudaGraphicsMapResources(1, &lastGBuffer.cudaBaryMeshIDResource, 0);
                cudaGraphicsMapResources(1, &lastGBuffer.cudaMotionDepthResource, 0);

                // 4. Denoising
                if(renderParam.denoise){
                    denoiseGridSize = dim3((width + denoiseBlockSize.x - 1) / denoiseBlockSize.x, (height + denoiseBlockSize.y - 1) / denoiseBlockSize.y);
                    // Temporal denoising
                    TemporalDenoiseKernel<<<denoiseGridSize, denoiseBlockSize>>>(denoiseParam, cameraParam, renderParam, renderBuffer, lastRenderBuffer, gBuffer, lastGBuffer, firstFrame);

                    // Computing Variance
                    int varianceKernelSize = 7; // 7x7 kernel
                    RenderBuffer current = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[0] : renderBuffers[1];
                    RenderBuffer last = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[1] : renderBuffers[0];
                    ComputeVarianceKernel<<<denoiseGridSize, denoiseBlockSize>>>(denoiseParam, renderParam, current, last, gBuffer, varianceKernelSize);
                    renderParam.currentRenderBufferIndex = 1 - renderParam.currentRenderBufferIndex;

                    // // Remove albedo
                    // RemoveAlbedoKernel<<<denoiseGridSize, denoiseBlockSize>>>(renderParam, renderBuffer);

                    // Atrous denoising
                    if(!firstFrame){
                        for(int i = 0; i < 5; i ++){
                            int stepSize = 1 << i;
                            current = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[0] : renderBuffers[1];
                            last = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[1] : renderBuffers[0];
                            AtrousKernel<<<denoiseGridSize, denoiseBlockSize>>>(denoiseParam, cameraParam, renderParam, current, last, gBuffer, stepSize);
                            cudaDeviceSynchronize();
                            CHECK_CUDA_ERROR(cudaGetLastError());
                            renderParam.currentRenderBufferIndex = 1 - renderParam.currentRenderBufferIndex;
                        }
                    }

                    // // std::swap(renderBuffer.framebuffer, renderBuffer.lastFrame);

                    // // Restore albedo
                    // RestoreAlbedoKernel<<<denoiseGridSize, denoiseBlockSize>>>(renderParam, renderBuffer);
                }

                // 5. Gather
                RenderBuffer currentRenderBuffer = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[0] : renderBuffers[1];
                gatherGridSize = dim3((width + gatherBlockSize.x - 1) / gatherBlockSize.x, (height + gatherBlockSize.y - 1) / gatherBlockSize.y);
                // GatherKernel<<<gatherGridSize, gatherBlockSize>>>(pixels, renderBuffer, gBuffer, sppCounter, renderParam, cameraParam, denoiseParam);
                GatherKernel<<<gatherGridSize, gatherBlockSize>>>(pixels, currentRenderBuffer, gBuffer, sppCounter, renderParam, cameraParam, denoiseParam);
                cudaDeviceSynchronize();

                cudaGraphicsUnmapResources(1, &gBuffer.cudaPositionTriIDResource, 0);
                cudaGraphicsUnmapResources(1, &gBuffer.cudaNormalMatIDResource, 0);
                cudaGraphicsUnmapResources(1, &gBuffer.cudaBaryMeshIDResource, 0);
                cudaGraphicsUnmapResources(1, &gBuffer.cudaMotionDepthResource, 0);

                cudaGraphicsUnmapResources(1, &lastGBuffer.cudaPositionTriIDResource, 0);
                cudaGraphicsUnmapResources(1, &lastGBuffer.cudaNormalMatIDResource, 0);
                cudaGraphicsUnmapResources(1, &lastGBuffer.cudaBaryMeshIDResource, 0);
                cudaGraphicsUnmapResources(1, &lastGBuffer.cudaMotionDepthResource, 0);
                // if(renderParam.denoise){
                //     // save last id buffer
                //     CHECK_CUDA_ERROR(cudaMemcpy(renderBuffer.lastIdBuffer, renderBuffer.idBuffer, sizeof(int) * width * height, cudaMemcpyDeviceToDevice));
                //     cudaDeviceSynchronize();
                // }
                ui -> shader -> Unuse();
                firstFrame = false;
                // Rendering pipeline end
            }
            
            
            cudaDeviceSynchronize(); // Render the frame buffer
            ui -> UpdateTexture();
            ui -> RenderFrameBuffer();
            ui -> GuiEnd();

            // Swap buffers and poll events
            glfwSwapBuffers(ui -> window);
            glfwSwapInterval(0); // Disable VSync
            glfwPollEvents();

            if(renderParam.denoise){
                renderParam.currentRenderBufferIndex = 1 - renderParam.currentRenderBufferIndex;
                renderParam.currentGBufferIndex = 1 - renderParam.currentGBufferIndex;
                framebufferReset = true;
            }

        }
    }

    void CudaInit()
    {
        // Initialize
        // render parameters
        int width = renderParam.width, height = renderParam.height;
        cudaMalloc(&renderSegments, sizeof(RenderSegment) * width * height);
        cudaMalloc(&renderSegmentsBuffer, sizeof(RenderSegment) * width * height);
        cudaMalloc(&intersections, sizeof(IntersectionInfo) * width * height);
        cudaMalloc(&intersectionsBuffer, sizeof(IntersectionInfo) * width * height);
        cudaMalloc(&segmentValidFlags, sizeof(int) * width * height);
        cudaMalloc(&segmentPos, sizeof(int) * width * height);
        cudaMemset(segmentValidFlags, 0, sizeof(int) * height);
        cudaMemset(segmentPos, 0, sizeof(int) * width * height);
        cudaMemset(renderSegmentsBuffer, 0, sizeof(RenderSegment) * width * height);
        cudaMalloc(&materialIDs, sizeof(int) * width * height);
        cudaMemset(materialIDs, 0, sizeof(int) * width * height);
        cudaMalloc(&materialSortIndices, sizeof(int) * width * height);
        cudaMemset(materialSortIndices, 0, sizeof(int) * width * height);

        CHECK_CUDA_ERROR(cudaSetDevice(0)); // Set the device to use
        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&ui -> cudaPBOResource, ui -> PBO, cudaGraphicsMapFlagsWriteDiscard));
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &ui -> cudaPBOResource, 0));
        CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&pixels, &numBytes, ui -> cudaPBOResource));
        if (numBytes != width * height * sizeof(uchar4))
        {
            std::cout << "Mapped PBO size does not match expected size: " << width * height * sizeof(uchar4) << " bytes";
            return;
        }

        // Initialize render buffer
        cudaMalloc(&renderBuffers[0].directLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc(&renderBuffers[0].indirectLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc(&renderBuffers[1].directLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc(&renderBuffers[1].indirectLightingBuffer, sizeof(float) * 4 * width * height);

        // Initialize denoise buffer
        // cudaMalloc(&denoiseParam.moments, sizeof(float2) * width * height);
        // cudaMalloc(&denoiseParam.variance, sizeof(float) * width * height);
        cudaMalloc(&denoiseParam.directMoments, sizeof(float2) * width * height);
        cudaMalloc(&denoiseParam.indirectMoments, sizeof(float2) * width * height);
        cudaMalloc(&denoiseParam.historyLength, sizeof(int) * width * height);

        std::cout << "Render buffer initialized." << std::endl;
    }
    void GBufferInit(){
        int width = renderParam.width, height = renderParam.height;
        // Initialize G-Buffer
        for(int i = 0; i < 2; i++){
            glGenFramebuffers(1, &gBuffers[i].fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, gBuffers[i].fbo);

            // texture0 position + triangleID
            glGenTextures(1, &gBuffers[i].texPositionTriID);
            glBindTexture(GL_TEXTURE_2D, gBuffers[i].texPositionTriID);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gBuffers[i].texPositionTriID, 0);

            // texture1 normal + mat
            glGenTextures(1, &gBuffers[i].texNormalMatID);
            glBindTexture(GL_TEXTURE_2D, gBuffers[i].texNormalMatID);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16UI, width, height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gBuffers[i].texNormalMatID, 0);

            // texture2 barycentric + meshID
            glGenTextures(1, &gBuffers[i].texBaryMeshID);
            glBindTexture(GL_TEXTURE_2D, gBuffers[i].texBaryMeshID);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16UI, width, height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gBuffers[i].texBaryMeshID, 0);

            // texture3 motion vector + depth + dz
            glGenTextures(1, &gBuffers[i].texMotionDepth);
            glBindTexture(GL_TEXTURE_2D, gBuffers[i].texMotionDepth);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, gBuffers[i].texMotionDepth, 0);

            // depth buffer
            glGenRenderbuffers(1, &gBuffers[i].depthBuffer);
            glBindRenderbuffer(GL_RENDERBUFFER, gBuffers[i].depthBuffer);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, width, height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gBuffers[i].depthBuffer);

            // Set the list of draw buffers
            GLuint drawBuffers[4] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3}; 
            glDrawBuffers(4, drawBuffers);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                std::cerr << "GBuffer FBO not complete!" << std::endl;

            glBindFramebuffer(GL_FRAMEBUFFER, 0); // Unbind framebuffer

            // init cuda resource 
            CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&gBuffers[i].cudaPositionTriIDResource, gBuffers[i].texPositionTriID, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
            CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&gBuffers[i].cudaNormalMatIDResource, gBuffers[i].texNormalMatID, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
            CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&gBuffers[i].cudaBaryMeshIDResource, gBuffers[i].texBaryMeshID, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
            CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&gBuffers[i].cudaMotionDepthResource, gBuffers[i].texMotionDepth, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

            // generate texture object
            GBuffer& gBuffer = gBuffers[i];
            cudaArray *positionTriIDArray, *normalMatIDArray, *baryMeshIDArray, *motionDepthArray;
            cudaTextureAddressMode addressMode = cudaAddressModeClamp;
            cudaTextureFilterMode filterMode = cudaFilterModePoint;
            cudaTextureReadMode readMode = cudaReadModeElementType;

            cudaGraphicsMapResources(1, &gBuffer.cudaPositionTriIDResource, 0);
            cudaGraphicsSubResourceGetMappedArray(&positionTriIDArray, gBuffer.cudaPositionTriIDResource, 0, 0);
            gBuffer.cudaTexObjPositionTriID = CreateTextureObject(positionTriIDArray, addressMode, cudaFilterModeLinear, readMode);
            cudaGraphicsUnmapResources(1, &gBuffer.cudaPositionTriIDResource, 0);

            cudaGraphicsMapResources(1, &gBuffer.cudaNormalMatIDResource, 0);
            cudaGraphicsSubResourceGetMappedArray(&normalMatIDArray, gBuffer.cudaNormalMatIDResource, 0, 0);
            gBuffer.cudaTexObjNormalMatID = CreateTextureObject(normalMatIDArray, addressMode, filterMode, readMode);
            cudaGraphicsUnmapResources(1, &gBuffer.cudaNormalMatIDResource, 0);

            cudaGraphicsMapResources(1, &gBuffer.cudaBaryMeshIDResource, 0);
            cudaGraphicsSubResourceGetMappedArray(&baryMeshIDArray, gBuffer.cudaBaryMeshIDResource, 0, 0);
            gBuffer.cudaTexObjBaryMeshID = CreateTextureObject(baryMeshIDArray, addressMode, filterMode, readMode);
            cudaGraphicsUnmapResources(1, &gBuffer.cudaBaryMeshIDResource, 0);

            cudaGraphicsMapResources(1, &gBuffer.cudaMotionDepthResource, 0);
            cudaGraphicsSubResourceGetMappedArray(&motionDepthArray, gBuffer.cudaMotionDepthResource, 0, 0);
            gBuffer.cudaTexObjMotionDepth = CreateTextureObject(motionDepthArray, addressMode, cudaFilterModeLinear, readMode);
            cudaGraphicsUnmapResources(1, &gBuffer.cudaMotionDepthResource, 0);

            // printf("G-Buffer %d CUDA texture objects created. positionTriID: %lu, normalMatID: %lu, baryMeshID: %lu, motionDepth: %lu\n", i, (unsigned long)gBuffer.cudaTexObjPositionTriID, (unsigned long)gBuffer.cudaTexObjNormalMatID, (unsigned long)gBuffer.cudaTexObjBaryMeshID, (unsigned long)gBuffer.cudaTexObjMotionDepth);
        }
        gBufferShader = new Shader("../../Shader/gbuffer.vs", "../../Shader/gbuffer.fs");
        std::cout << "G-Buffer initialized." << std::endl;
    }
    void RenderGBuffer(RenderParam renderParam, CameraParam cameraParam){
        int currentGBufferIndex = renderParam.currentGBufferIndex;
        glBindFramebuffer(GL_FRAMEBUFFER, gBuffers[currentGBufferIndex].fbo);
        glViewport(0, 0, renderParam.width, renderParam.height);

        // Clear the framebuffer
        float clearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float clearDepth = 1.0f;
        uint32_t clearUInt[4] = {0, 0, 0, 0};

        glClearBufferfv(GL_COLOR, 0, clearColor);
        glClearBufferuiv(GL_COLOR, 1, clearUInt);
        glClearBufferuiv(GL_COLOR, 2, clearUInt);
        glClearBufferfv(GL_COLOR, 3, clearColor);
        glClearBufferfv(GL_DEPTH, 0, &clearDepth);

        glEnable(GL_DEPTH_TEST);

        gBufferShader -> Use();

        // Set camera uniforms
        Mat4f viewMatrix = LookAt(cameraParam.position, cameraParam.lookat, cameraParam.up);
        float nearPlane = 0.1f;
        float farPlane = renderParam.sceneBounds.DiagonalLength() + (cameraParam.position - renderParam.sceneBounds.Center()).norm();
        Mat4f projectionMatrix = Perspective(cameraParam.fovy, cameraParam.ratio, nearPlane, farPlane);
        Mat4f modelMatrix = Mat4f::Identity();
        Mat4f lastViewMatrix = LookAt(cameraParam.lastPosition, cameraParam.lastLookat, cameraParam.lastUp);
        Mat4f lastProjectionMatrix = Perspective(cameraParam.fovy, cameraParam.ratio, nearPlane, farPlane);
        Mat4f viewProjectionMatrix = projectionMatrix * viewMatrix;
        Mat4f lastViewProjectionMatrix = lastProjectionMatrix * lastViewMatrix;
        gBufferShader -> SetUniformMat4("model", modelMatrix.data());
        // gBufferShader -> SetUniformMat4("viewProjection", viewProjectionMatrix.data());
        // gBufferShader -> SetUniformMat4("prevViewProjection", lastViewProjectionMatrix.data());
        gBufferShader -> SetUniformMat4("view", viewMatrix.data());
        gBufferShader -> SetUniformMat4("projection", projectionMatrix.data());
        gBufferShader -> SetUniformMat4("prevView", lastViewMatrix.data());
        gBufferShader -> SetUniformMat4("prevProjection", lastProjectionMatrix.data());
        gBufferShader -> SetUniformFloat("uNearPlane", nearPlane);
        gBufferShader -> SetUniformFloat("uFarPlane", farPlane);
        
        // Render all meshes
        for (size_t i = 0; i < renderParam.numMeshes; ++i)
        {
            MeshData* mesh = &renderParam.hostMeshes[i];
            // Set material ID uniform
            gBufferShader -> SetUniformInt("uMaterialID", mesh -> materialID);
            gBufferShader -> SetUniformInt("uMeshID", mesh -> meshID + 1); // meshID + 1 to distinguish from background(0)
            
            // Bind VAO and draw
            glBindVertexArray(mesh->vao);
            glDrawArrays(GL_TRIANGLES, 0, mesh->numTriangles * 3);
            glBindVertexArray(0);

            // std::cout << "Rendered G-Buffer for mesh " << i + 1 << "/" << renderParam.numMeshes << ", triangles: " << mesh->numTriangles << " VAO: "<< mesh->vao << std::endl;
        }
        gBufferShader -> Unuse();
        glDisable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    UI *ui;
    uchar4* pixels;
    RenderBuffer renderBuffers[2];
    GBuffer gBuffers[2];
    DenoiseParam denoiseParam;
    size_t numBytes;
    int sppCounter = 0;
    bool framebufferReset = false;
    Shader *gBufferShader;
};