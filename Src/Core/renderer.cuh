#include "scene.h"
#include "ui.h"
#include "light.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "camera.h"
#include "lightmanager.h"
#include <filesystem>
#include <chrono>

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

__device__ __forceinline__ int BufferYToGBufferY(int bufferY, int height)
{
    return bufferY;
}

__device__ __forceinline__ Vec2f BufferIndexToGBufferSamplePos(int index, int width, int height)
{
    int x = index % width;
    int y = index / width;
    return Vec2f((float)x + 0.5f, (float)BufferYToGBufferY(y, height) + 0.5f);
}

__device__ __forceinline__ int GBufferPixelToBufferIndex(int x, int y, int width, int height)
{
    return y * width + x;
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
    segments[idx].specularBounce = false;
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
    Vec3f geomNormal = info.normal.normalized();
    Vec3f hitNormal = info.shadingNormal.squaredNorm() > eps ? info.shadingNormal.normalized() : geomNormal;
    if (hitNormal.dot(geomNormal) < 0.f) {
        hitNormal = -hitNormal;
    }
    Vec2f uv = info.texCoord;
    Vec3f wo = -segment.ray.direction; // wo为p -> eye方向, wi为p -> light方向

    Vec3f brdf;

    if(material.IsLight()){
        if(segment.firstBounce){
            segment.directColor+= material.ke;
            segment.firstBounce = false;
            segment.remainingBounces = 0;
            segmentValidFlags[idx] = 0;
            return ;
        }else if(segment.specularBounce){
            segment.indirectColor += segment.weight.cwiseProduct(material.ke);
        }else{
            int lightID = meshData[info.meshID].lightID;
            Light *light = lightManager -> GetLight(lightID);
            float pdfLight;
            light -> SampleSolidAnglePDF(segment.ray, info.hitPoint, info.normal, pdfLight);
            // if (lightManager->numLights > 0) {
            //     pdfLight *= 1.0f / lightManager->numLights;
            // }

            float brdfMisWeight = MISWeight(segment.pdfBrdf, pdfLight, 2);
            segment.indirectColor += segment.weight.cwiseProduct(material.ke) * brdfMisWeight;
        }
        segment.remainingBounces = 0;
        segmentValidFlags[idx] = 0;
        return;
    }

    if(material.IsDelta()){
        Vec3f deltaWi;
        Vec3f deltaWeight;
        float deltaPdf = 1.0f;
        material.SampleDielectric(wo, hitNormal, deltaWi, deltaPdf, deltaWeight, sampler, idx);
        segment.pdfBrdf = deltaPdf;
        segment.ray.origin = hitPoint + deltaWi * 1e-4f;
        segment.ray.direction = deltaWi;
        segment.weight = segment.weight.cwiseProduct(deltaWeight);

        float p = sampler -> Get1D(idx);
        if(p > rr){
            segment.remainingBounces = 0;
            segmentValidFlags[idx] = 0;
            return ;
        }

        segment.weight /= rr;
        segment.firstBounce = false;
        segment.specularBounce = true;
        segmentValidFlags[idx] = 1;
        return;
    }

    if(material.type != SUBSURFACE){
        float selectLightPDF;
        Light *light = lightManager -> SampleLight(sampler, idx, selectLightPDF);
        if(light != nullptr){
            float sampleLightPointPDF;
            TriangleSampleInfo sampleLightPointInfo;
            light -> SamplePoint(triangles, sceneBounds, sampleLightPointInfo, sampleLightPointPDF, sampler, idx);
            
            bool visible = light -> Visible(info, sampleLightPointInfo, bvh);
            if(visible){
                Vec3f dirWi = (sampleLightPointInfo.position - hitPoint).normalized();
                float pdfLight;
                Ray sampleLightRay;
                sampleLightRay.origin = hitPoint;
                sampleLightRay.direction = dirWi;
                light -> SampleSolidAnglePDF(sampleLightRay, sampleLightPointInfo.position, sampleLightPointInfo.normal, pdfLight);
                pdfLight *= selectLightPDF;
                float cosTheta = hitNormal.dot(dirWi);
                Vec3f dirBrdf = material.Evaluate(wo, hitNormal, dirWi, uv);
                Vec3f dirL = light -> Emission(dirWi).cwiseProduct(dirBrdf) * cosTheta / pdfLight;

                float pdfBrdf;
                material.Pdf(wo, hitNormal, dirWi, pdfBrdf, uv);
                float lightMisWeight = MISWeight(pdfLight, pdfBrdf, 2);
                if(segment.firstBounce){
                    segment.directColor += lightMisWeight * segment.weight.cwiseProduct(dirL);
                }else{
                    segment.indirectColor += lightMisWeight * segment.weight.cwiseProduct(dirL);
                }
            }
        }

        Vec3f indirWi;
        float indirWiPdf;
        material.Sample(wo, hitNormal, indirWi, indirWiPdf, sampler, idx, uv);
        brdf = material.Evaluate(wo, hitNormal, indirWi, uv);
        segment.pdfBrdf = indirWiPdf;
        segment.ray.origin = hitPoint;
        segment.ray.direction = indirWi;

        float indirCosTheta = hitNormal.dot(indirWi);
        if(!(indirWiPdf < eps || indirCosTheta < 0)){
            segment.weight = segment.weight.cwiseProduct(brdf) * indirCosTheta / indirWiPdf;
        }else{
            segment.remainingBounces = 0;
            segmentValidFlags[idx] = 0;
            return ;
        }
    }else{
        Vec3f FIn;
        material.Fresnel(wo, hitNormal, FIn);
        float reflectProb = Clamp(0.02f, 0.98f, FIn(0));
        float transmitProb = fmaxf(1.f - reflectProb, eps);
        float eventSample = sampler -> Get1D(idx);

        if(eventSample < reflectProb){
            Vec3f reflectedDir;
            if (material.roughness > 1e-3f) {
                float r1 = sampler->Get1D(idx);
                float r2 = sampler->Get1D(idx);
                float a = fmaxf(material.roughness * material.roughness, 0.02f);
                float phi = 2.0f * PI * r1;
                float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
                float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
                Vec3f hLocal(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
                Vec3f hWorld = LocalToWorld(hLocal, hitNormal);
                reflectedDir = Reflect(-wo, hWorld).normalized();
                if (reflectedDir.dot(hitNormal) <= 0.0f) {
                    reflectedDir = Reflect(-wo, hitNormal).normalized();
                }
            } else {
                reflectedDir = Reflect(-wo, hitNormal).normalized();
            }
            segment.ray.origin = hitPoint + reflectedDir * 1e-4f;
            segment.ray.direction = reflectedDir;
            segment.weight = segment.weight.cwiseProduct(FIn) / reflectProb;
            segment.pdfBrdf = 1.0f;
            float p = sampler -> Get1D(idx);
            if(p > rr){
                segment.remainingBounces = 0;
                segmentValidFlags[idx] = 0;
                return ;
            }

            segment.weight /= rr;
            segment.firstBounce = false;
            segment.specularBounce = true;
            segmentValidFlags[idx] = 1;
            return;
        }else{
            Vec3f outgoingPoint, outgoingNormal, outgoingDir;
            float spatialPdf, outgoingDirPdf;
            material.SampleSpatialDiffusionProfile(wo, geomNormal, hitPoint, outgoingPoint, outgoingNormal, spatialPdf, sampler, idx, bvh);
            if(spatialPdf < eps){
                segment.remainingBounces = 0;
                segmentValidFlags[idx] = 0;
                return ;
            }

            Vec3f spatialProfile = material.EvaluateSpatialDiffusionProfile(wo, geomNormal, hitPoint, outgoingPoint, outgoingNormal, uv);
            segment.weight = segment.weight.cwiseProduct(Vec3f(1.f, 1.f, 1.f) - FIn).cwiseProduct(spatialProfile) / (transmitProb * spatialPdf);

            Vec3f woSubsurface = hitPoint - outgoingPoint;
            if(woSubsurface.norm() < eps){
                woSubsurface = wo;
            }else{
                woSubsurface = woSubsurface.normalized();
            }
            
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
                    if(cosTheta > 0.f && pdfLight > eps){
                        Vec3f dirBrdf = Vec3f(1.f, 1.f, 1.f) / PI;
                        Vec3f dirL = light -> Emission(dirWi).cwiseProduct(dirBrdf) * cosTheta / pdfLight;
                        float pdfBrdf = cosTheta / PI;
                        float lightMisWeight = MISWeight(pdfLight, pdfBrdf, 2);
                        if(segment.firstBounce){
                            segment.directColor += lightMisWeight * segment.weight.cwiseProduct(dirL) * FtransmitOut;
                        }else{
                            segment.indirectColor += lightMisWeight * segment.weight.cwiseProduct(dirL) * FtransmitOut;
                        }
                    }
                }
            }

            material.SampleDiffuse(woSubsurface, outgoingNormal, outgoingDir, outgoingDirPdf, sampler, idx, uv);
            float cosOut = outgoingNormal.dot(outgoingDir);
            Vec3f FOut;
            material.Fresnel(outgoingDir, outgoingNormal, FOut);
            float FtransmitOut = 1 - FOut(0);

            if(outgoingDirPdf < eps || cosOut < 0.f){
                segment.remainingBounces = 0;
                segmentValidFlags[idx] = 0;
                return ;
            }

            segment.ray.origin = outgoingPoint + outgoingNormal * 1e-4f;
            segment.ray.direction = outgoingDir;
            segment.weight *= FtransmitOut;
            segment.pdfBrdf = outgoingDirPdf;
        }
    }
    

    // Russian roulette
    float p = sampler -> Get1D(idx);
    if(p > rr){
        segment.remainingBounces = 0;
        segmentValidFlags[idx] = 0;
        return ;
    }

    segment.weight /= rr;
    

    segment.firstBounce = false;
    segment.specularBounce = false;
    segmentValidFlags[idx] = 1;

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

__device__ bool GetLastFramePos(int index, RenderParam renderParam, GBuffer currentGBuffer, GBuffer lastGBuffer, Vec2f &pixelPos, float reprojectionDepthFactor){
    int width = renderParam.width, height = renderParam.height;
    Vec2f currentPixelPos = BufferIndexToGBufferSamplePos(index, width, height);
    float4 positionTriID = tex2D<float4>(currentGBuffer.cudaTexObjPositionTriID, currentPixelPos(0), currentPixelPos(1));
    ushort4 baryMeshID = tex2D<ushort4>(currentGBuffer.cudaTexObjBaryMeshID, currentPixelPos(0), currentPixelPos(1));
    ushort4 normalMatID = tex2D<ushort4>(currentGBuffer.cudaTexObjNormalMatID, currentPixelPos(0), currentPixelPos(1));
    float4 motionDepth = tex2D<float4>(currentGBuffer.cudaTexObjMotionDepth, currentPixelPos(0), currentPixelPos(1));
    Vec3f position = Vec3f(positionTriID.x, positionTriID.y, positionTriID.z);
    int meshID = baryMeshID.w;
    Vec3f normal = Vec3f(normalMatID.x / 65535.0f * 2.f - 1.f, normalMatID.y / 65535.0f * 2.f - 1.f, normalMatID.z / 65535.0f * 2.f - 1.f);
    float depth = motionDepth.z;
    float dz = motionDepth.w;

    Vec2f motionVec = Vec2f(motionDepth.x * width * 0.5f, motionDepth.y * height * 0.5f);
    pixelPos = Vec2f(currentPixelPos(0) - motionVec(0), currentPixelPos(1) - motionVec(1));
    if(pixelPos(0) < 0.5f || pixelPos(0) >= width - 0.5f || pixelPos(1) < 0.5f || pixelPos(1) >= height - 0.5f){
        return false;
    }

    int lastPixelX = max(0, min(width - 1, (int)floorf(pixelPos(0))));
    int lastPixelY = max(0, min(height - 1, (int)floorf(pixelPos(1))));
    Vec2f lastPixelPos = Vec2f((float)lastPixelX + 0.5f, (float)lastPixelY + 0.5f);

    // get last frame gbuffer info
    float4 lastPositionTriID = tex2D<float4>(lastGBuffer.cudaTexObjPositionTriID, lastPixelPos(0), lastPixelPos(1));
    ushort4 lastBaryMeshID = tex2D<ushort4>(lastGBuffer.cudaTexObjBaryMeshID, lastPixelPos(0), lastPixelPos(1));
    ushort4 lastNormalMatID = tex2D<ushort4>(lastGBuffer.cudaTexObjNormalMatID, lastPixelPos(0), lastPixelPos(1));
    float4 lastMotionDepth = tex2D<float4>(lastGBuffer.cudaTexObjMotionDepth, lastPixelPos(0), lastPixelPos(1));
    int lastMeshID = lastBaryMeshID.w;
    int materialID = normalMatID.w;
    int lastMaterialID = lastNormalMatID.w;
    Vec3f lastNormal = Vec3f(lastNormalMatID.x / 65535.0f * 2.f - 1.f, lastNormalMatID.y / 65535.0f * 2.f - 1.f, lastNormalMatID.z / 65535.0f * 2.f - 1.f);
    float lastDepth = lastMotionDepth.z;
    Vec3f lastPosition = Vec3f(lastPositionTriID.x, lastPositionTriID.y, lastPositionTriID.z);

    float dotNormal = normal.dot(lastNormal);
    float depthDiff = fabsf(depth - lastDepth);
    float normalThreshold = 0.9;
    float pixelDistance = sqrtf(motionVec(0) * motionVec(0) + motionVec(1) * motionVec(1));
    float depthThreshold = fmaxf(dz, lastMotionDepth.w) * (pixelDistance + 1.f) + reprojectionDepthFactor * fmaxf(depth, lastDepth);

    if(meshID != lastMeshID || materialID != lastMaterialID || dotNormal < normalThreshold || depthDiff > depthThreshold){
        return false;
    }

    pixelPos = lastPixelPos;
    return true;
}

__device__ void TemporalDenoising(int index, RenderBuffer currentBuffer, RenderBuffer lastBuffer, DenoiseParam denoiseParam, CameraParam cameraParam, RenderParam renderParam, GBuffer currentGBuffer, GBuffer lastGBuffer, bool firstFrame)
{
    float2 *currentDirectMoments = denoiseParam.directMoments[denoiseParam.currentTemporalBufferIndex];
    float2 *currentIndirectMoments = denoiseParam.indirectMoments[denoiseParam.currentTemporalBufferIndex];
    int *currentHistoryLength = denoiseParam.historyLength[denoiseParam.currentTemporalBufferIndex];
    float2 *lastDirectMomentsBuffer = denoiseParam.directMoments[1 - denoiseParam.currentTemporalBufferIndex];
    float2 *lastIndirectMomentsBuffer = denoiseParam.indirectMoments[1 - denoiseParam.currentTemporalBufferIndex];
    int *lastHistoryLengthBuffer = denoiseParam.historyLength[1 - denoiseParam.currentTemporalBufferIndex];

    if(firstFrame){
        float directLuminance = Luminance(Vec3f(currentBuffer.directLightingBuffer[index * 4 + 0], currentBuffer.directLightingBuffer[index * 4 + 1], currentBuffer.directLightingBuffer[index * 4 + 2]));
        float indirectLuminance = Luminance(Vec3f(currentBuffer.indirectLightingBuffer[index * 4 + 0], currentBuffer.indirectLightingBuffer[index * 4 + 1], currentBuffer.indirectLightingBuffer[index * 4 + 2]));
        currentDirectMoments[index] = make_float2(directLuminance, directLuminance * directLuminance);
        currentIndirectMoments[index] = make_float2(indirectLuminance, indirectLuminance * indirectLuminance);
        currentBuffer.directLightingBuffer[index * 4 + 3] = 1.f;
        currentBuffer.indirectLightingBuffer[index * 4 + 3] = 1.f;
        currentHistoryLength[index] = 1;
        return;
    }

    Vec2f pixelPos;
    bool valid = GetLastFramePos(index, renderParam, currentGBuffer, lastGBuffer, pixelPos, denoiseParam.reprojectionDepthFactor);
    int lastFrameIndex = index;
    if(valid){
        int lastFrameX = max(0, min(renderParam.width - 1, (int)floorf(pixelPos(0))));
        int lastFrameY = max(0, min(renderParam.height - 1, (int)floorf(pixelPos(1))));
        lastFrameIndex = GBufferPixelToBufferIndex(lastFrameX, lastFrameY, renderParam.width, renderParam.height);
    }

    int historyLength = valid ? min(lastHistoryLengthBuffer[lastFrameIndex] + 1, denoiseParam.maxHistoryLength) : 1;
    float alpha = valid ? 1.f / historyLength : 1.0f;

    Vec3f currentDirectColor = Vec3f(currentBuffer.directLightingBuffer[index * 4 + 0], currentBuffer.directLightingBuffer[index * 4 + 1], currentBuffer.directLightingBuffer[index * 4 + 2]);
    Vec3f currentIndirectColor = Vec3f(currentBuffer.indirectLightingBuffer[index * 4 + 0], currentBuffer.indirectLightingBuffer[index * 4 + 1], currentBuffer.indirectLightingBuffer[index * 4 + 2]);
    float currentDirectLuminance = Luminance(currentDirectColor);
    float currentIndirectLuminance = Luminance(currentIndirectColor);
    currentBuffer.directLightingBuffer[index * 4 + 0] = alpha * currentBuffer.directLightingBuffer[index * 4 + 0] + (1 - alpha) * lastBuffer.directLightingBuffer[lastFrameIndex * 4 + 0];
    currentBuffer.directLightingBuffer[index * 4 + 1] = alpha * currentBuffer.directLightingBuffer[index * 4 + 1] + (1 - alpha) * lastBuffer.directLightingBuffer[lastFrameIndex * 4 + 1];
    currentBuffer.directLightingBuffer[index * 4 + 2] = alpha * currentBuffer.directLightingBuffer[index * 4 + 2] + (1 - alpha) * lastBuffer.directLightingBuffer[lastFrameIndex * 4 + 2];
    currentBuffer.indirectLightingBuffer[index * 4 + 0] = alpha * currentBuffer.indirectLightingBuffer[index * 4 + 0] + (1 - alpha) * lastBuffer.indirectLightingBuffer[lastFrameIndex * 4 + 0];
    currentBuffer.indirectLightingBuffer[index * 4 + 1] = alpha * currentBuffer.indirectLightingBuffer[index * 4 + 1] + (1 - alpha) * lastBuffer.indirectLightingBuffer[lastFrameIndex * 4 + 1];
    currentBuffer.indirectLightingBuffer[index * 4 + 2] = alpha * currentBuffer.indirectLightingBuffer[index * 4 + 2] + (1 - alpha) * lastBuffer.indirectLightingBuffer[lastFrameIndex * 4 + 2];

    float2 &directMoment = currentDirectMoments[index];
    float2 &indirectMoment = currentIndirectMoments[index];
    float2 lastDirectMoment = valid ? lastDirectMomentsBuffer[lastFrameIndex] : make_float2(0.f, 0.f);
    float2 lastIndirectMoment = valid ? lastIndirectMomentsBuffer[lastFrameIndex] : make_float2(0.f, 0.f);
    directMoment.x = alpha * currentDirectLuminance + (1 - alpha) * lastDirectMoment.x;
    directMoment.y = alpha * currentDirectLuminance * currentDirectLuminance + (1 - alpha) * lastDirectMoment.y;
    indirectMoment.x = alpha * currentIndirectLuminance + (1 - alpha) * lastIndirectMoment.x;
    indirectMoment.y = alpha * currentIndirectLuminance * currentIndirectLuminance + (1 - alpha) * lastIndirectMoment.y;
    if(valid){
        currentBuffer.directLightingBuffer[index * 4 + 3] = directMoment.y - directMoment.x * directMoment.x;
        currentBuffer.indirectLightingBuffer[index * 4 + 3] = indirectMoment.y - indirectMoment.x * indirectMoment.x;
        currentHistoryLength[index] = historyLength;
    }else{
        currentBuffer.directLightingBuffer[index * 4 + 3] = 1.f;
        currentBuffer.indirectLightingBuffer[index * 4 + 3] = 1.f;
        currentHistoryLength[index] = 1;
    }
    
}

__global__ void TemporalDenoiseKernel(DenoiseParam denoiseParam, CameraParam cameraParam, RenderParam renderParam, RenderBuffer currentBuffer, RenderBuffer lastBuffer, GBuffer currentGBuffer, GBuffer lastGBuffer, bool firstFrame)
{
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
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
    float2 *currentDirectMoments = denoiseParam.directMoments[denoiseParam.currentTemporalBufferIndex];
    float2 *currentIndirectMoments = denoiseParam.indirectMoments[denoiseParam.currentTemporalBufferIndex];
    int *currentHistoryLength = denoiseParam.historyLength[denoiseParam.currentTemporalBufferIndex];
    int sampleY = BufferYToGBufferY(j, height);
    float sampleXf = (float)i + 0.5f;
    float sampleYf = (float)sampleY + 0.5f;
    int historyLength = currentHistoryLength[idx];
    if(historyLength < 4){
        Vec3f centerDirectColor = Vec3f(currentBuffer.directLightingBuffer[4 * idx + 0], currentBuffer.directLightingBuffer[4 * idx + 1], currentBuffer.directLightingBuffer[4 * idx + 2]);
        Vec3f centerIndirectColor = Vec3f(currentBuffer.indirectLightingBuffer[4 * idx + 0], currentBuffer.indirectLightingBuffer[4 * idx + 1], currentBuffer.indirectLightingBuffer[4 * idx + 2]);
        float centerDirectLuminance = Luminance(centerDirectColor);
        float centerIndirectLuminance = Luminance(centerIndirectColor);
        float4 motionDepth = tex2D<float4>(gBuffer.cudaTexObjMotionDepth, sampleXf, sampleYf);
        float centerDepth = motionDepth.z;
        float dz = fmaxf(motionDepth.w, 1e-4f);
        ushort4 normalMatID = tex2D<ushort4>(gBuffer.cudaTexObjNormalMatID, sampleXf, sampleYf);
        ushort4 baryMeshID = tex2D<ushort4>(gBuffer.cudaTexObjBaryMeshID, sampleXf, sampleYf);
        int centerMaterialID = normalMatID.w;
        int centerMeshID = baryMeshID.w;
        Vec3f centerNormal = Vec3f(normalMatID.x / 65535.0f * 2.f - 1.f, normalMatID.y / 65535.0f * 2.f - 1.f, normalMatID.z / 65535.0f * 2.f - 1.f).normalized();
        float centerDirectVariance = fmaxf(0.0f, currentBuffer.directLightingBuffer[idx * 4 + 3]);
        float centerIndirectVariance = fmaxf(0.0f, currentBuffer.indirectLightingBuffer[idx * 4 + 3]);

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
                    Vec3f nDirectColor = Vec3f(currentBuffer.directLightingBuffer[4 * nidx + 0], currentBuffer.directLightingBuffer[4 * nidx + 1], currentBuffer.directLightingBuffer[4 * nidx + 2]);
                    Vec3f nIndirectColor = Vec3f(currentBuffer.indirectLightingBuffer[4 * nidx + 0], currentBuffer.indirectLightingBuffer[4 * nidx + 1], currentBuffer.indirectLightingBuffer[4 * nidx + 2]);
                    int neighborSampleY = BufferYToGBufferY(nj, height);
                    float neighborSampleXf = (float)ni + 0.5f;
                    float neighborSampleYf = (float)neighborSampleY + 0.5f;
                    float4 nMotionDepth = tex2D<float4>(gBuffer.cudaTexObjMotionDepth, neighborSampleXf, neighborSampleYf);
                    float nDepth = nMotionDepth.z;
                    ushort4 nNormalMatID = tex2D<ushort4>(gBuffer.cudaTexObjNormalMatID, neighborSampleXf, neighborSampleYf);
                    ushort4 nBaryMeshID = tex2D<ushort4>(gBuffer.cudaTexObjBaryMeshID, neighborSampleXf, neighborSampleYf);
                    int neighborMaterialID = nNormalMatID.w;
                    int neighborMeshID = nBaryMeshID.w;
                    if(centerMaterialID != neighborMaterialID || centerMeshID != neighborMeshID){
                        continue;
                    }
                    Vec3f nNormal = Vec3f(nNormalMatID.x / 65535.0f * 2.f - 1.f, nNormalMatID.y / 65535.0f * 2.f - 1.f, nNormalMatID.z / 65535.0f * 2.f - 1.f).normalized();
                    float nDirectLuminance = Luminance(nDirectColor);
                    float nIndirectLuminance = Luminance(nIndirectColor);
                    Vec2f nDirectMoment = Vec2f(currentDirectMoments[nidx].x, currentDirectMoments[nidx].y);
                    Vec2f nIndirectMoment = Vec2f(currentIndirectMoments[nidx].x, currentIndirectMoments[nidx].y);

                    float dist = sqrtf(float(ki * ki + kj * kj));
                    float wz = -fabsf(centerDepth - nDepth) / (sigmaDepth * dist + 1e-4f);
                    float wn = powf(fmaxf(0.f, centerNormal.dot(nNormal)), denoiseParam.sigmaNormal);
                    float directLumaRef = fmaxf(fmaxf(centerDirectLuminance, nDirectLuminance), 1e-4f);
                    float wDirect = -fabsf(centerDirectLuminance - nDirectLuminance) / (denoiseParam.sigmaLight / sqrtf(directLumaRef));
                    float wIndirect = -fabsf(centerIndirectLuminance - nIndirectLuminance) / (denoiseParam.sigmaLight / sqrtf(centerIndirectLuminance));

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
    }else{
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
    int *currentHistoryLength = denoiseParam.historyLength[denoiseParam.currentTemporalBufferIndex];
    int centerHistoryLength = max(currentHistoryLength[idx], 1);
    int sampleY = BufferYToGBufferY(j, height);
    float sampleXf = (float)i + 0.5f;
    float sampleYf = (float)sampleY + 0.5f;

    Vec3f centerDirectColor = Vec3f(renderBuffer.directLightingBuffer[4 * idx + 0], renderBuffer.directLightingBuffer[4 * idx + 1], renderBuffer.directLightingBuffer[4 * idx + 2]);
    Vec3f centerIndirectColor = Vec3f(renderBuffer.indirectLightingBuffer[4 * idx + 0], renderBuffer.indirectLightingBuffer[4 * idx + 1], renderBuffer.indirectLightingBuffer[4 * idx + 2]);
    
    float centerDirectLuminance = Luminance(centerDirectColor);
    float centerIndirectLuminance = Luminance(centerIndirectColor);

    float4 motionDepth = tex2D<float4>(gBuffer.cudaTexObjMotionDepth, sampleXf, sampleYf);
    float centerDepth = motionDepth.z;
    float dz = fmaxf(motionDepth.w, 1e-4f); 

        ushort4 normalMatID = tex2D<ushort4>(gBuffer.cudaTexObjNormalMatID, sampleXf, sampleYf);
        ushort4 baryMeshID = tex2D<ushort4>(gBuffer.cudaTexObjBaryMeshID, sampleXf, sampleYf);
        int centerMaterialID = normalMatID.w;
        int centerMeshID = baryMeshID.w;
    Vec3f centerNormal = Vec3f(normalMatID.x / 65535.0f * 2.f - 1.f, normalMatID.y / 65535.0f * 2.f - 1.f, normalMatID.z / 65535.0f * 2.f - 1.f).normalized();

    float centerDirectVariance = fmaxf(0.0f, renderBuffer.directLightingBuffer[idx * 4 + 3]);
    float centerIndirectVariance = fmaxf(0.0f, renderBuffer.indirectLightingBuffer[idx * 4 + 3]);
    float stdDevDirect = sqrtf(centerDirectVariance) + 1e-3f; 
    float stdDevIndirect = sqrtf(centerIndirectVariance) + 1e-3f; 

    Vec3f finalDirectColor(0.f, 0.f, 0.f);
    Vec3f finalIndirectColor(0.f, 0.f, 0.f);
    float totalDirectWeight = 0.f;
    float totalIndirectWeight = 0.f;
    float directVariance = 0.f;
    float indirectVariance = 0.f;

    float bootstrap = fminf(1.f, fmaxf(0.f, (4.f - (float)centerHistoryLength) / 3.f));
    float sigmaDepth = fmaxf(denoiseParam.sigmaDepth, 1e-3f) * fmaxf(dz, 1e-4f) * (1.f + 2.5f * bootstrap);
    float sigmaLightDirect = denoiseParam.sigmaLight * (1.f + 2.0f * bootstrap);
    float sigmaLightIndirect = denoiseParam.sigmaLight * (1.f + 1.5f * bootstrap);
    float sigmaNormal = fmaxf(8.f, denoiseParam.sigmaNormal * (1.f - 0.85f * bootstrap));
    
    const float kernelWeights[3] = {3.f / 8.f, 1.f / 4.f, 1.f / 16.f};

    for(int kj = -2; kj <= 2; kj++){
        for(int ki = -2; ki <= 2; ki++){
            int ni = i + ki * stepSize;
            int nj = j + kj * stepSize;

            if(ni >= 0 && ni < width && nj >= 0 && nj < height){
                int nidx = nj * width + ni;
                int neighborHistoryLength = max(currentHistoryLength[nidx], 1);
                int neighborSampleY = BufferYToGBufferY(nj, height);
                float neighborSampleXf = (float)ni + 0.5f;
                float neighborSampleYf = (float)neighborSampleY + 0.5f;
                float4 nMotionDepth = tex2D<float4>(gBuffer.cudaTexObjMotionDepth, neighborSampleXf, neighborSampleYf);
                float nDepth = nMotionDepth.z;

                ushort4 nNormalMatID = tex2D<ushort4>(gBuffer.cudaTexObjNormalMatID, neighborSampleXf, neighborSampleYf);
                ushort4 nBaryMeshID = tex2D<ushort4>(gBuffer.cudaTexObjBaryMeshID, neighborSampleXf, neighborSampleYf);
                int neighborMaterialID = nNormalMatID.w;
                int neighborMeshID = nBaryMeshID.w;
                if(centerMaterialID != neighborMaterialID || centerMeshID != neighborMeshID){
                    continue;
                }
                Vec3f nNormal = Vec3f(nNormalMatID.x / 65535.0f * 2.f - 1.f, nNormalMatID.y / 65535.0f * 2.f - 1.f, nNormalMatID.z / 65535.0f * 2.f - 1.f).normalized();

                Vec3f nDirectColor = Vec3f(renderBuffer.directLightingBuffer[4 * nidx + 0], renderBuffer.directLightingBuffer[4 * nidx + 1], renderBuffer.directLightingBuffer[4 * nidx + 2]);
                float nDirectVariance = fmaxf(0.0f, renderBuffer.directLightingBuffer[nidx * 4 + 3]);
                Vec3f nIndirectColor = Vec3f(renderBuffer.indirectLightingBuffer[4 * nidx + 0], renderBuffer.indirectLightingBuffer[4 * nidx + 1], renderBuffer.indirectLightingBuffer[4 * nidx + 2]);
                float nIndirectVariance = fmaxf(0.0f, renderBuffer.indirectLightingBuffer[nidx * 4 + 3]);

                float dist = sqrtf(float(ki * ki + kj * kj)) * stepSize;
                float wz = -fabsf(centerDepth - nDepth) / (sigmaDepth * dist + 1e-6f);
                float wn = powf(fmaxf(0.f, centerNormal.dot(nNormal)), sigmaNormal);
                float directLuminance = Luminance(nDirectColor);
                float indirectLuminance = Luminance(nIndirectColor);
                float directLumaRef = fmaxf(fmaxf(centerDirectLuminance, directLuminance), 1e-4f);
                float indirectLumaRef = fmaxf(centerIndirectLuminance, 1e-4f);
                float wDirect = -fabsf(centerDirectLuminance - directLuminance) / (sigmaLightDirect * stdDevDirect / sqrtf(directLumaRef));
                float wIndirect = -fabsf(centerIndirectLuminance - indirectLuminance) / (sigmaLightIndirect * stdDevIndirect / sqrtf(indirectLumaRef));
                float historyConfidence = bootstrap > 0.f ? (0.5f + 0.5f * fminf((float)neighborHistoryLength / 4.f, 1.f)) : 1.f;

                float kWeight = kernelWeights[abs(ki)] * kernelWeights[abs(kj)];
                float dWeight = expf(wz + wDirect) * wn * kWeight * historyConfidence;
                float iWeight = expf(wz + wIndirect) * wn * kWeight * historyConfidence;

                finalDirectColor += dWeight * nDirectColor;
                finalIndirectColor += iWeight * nIndirectColor;
                totalDirectWeight += dWeight;
                totalIndirectWeight += iWeight;
                directVariance += dWeight * dWeight * nDirectVariance;
                indirectVariance += iWeight * iWeight * nIndirectVariance;
            }
        }
    }

    if(totalDirectWeight > 1e-6f) {
        lastRenderBuffer.directLightingBuffer[4 * idx + 0] = finalDirectColor(0) / totalDirectWeight;
        lastRenderBuffer.directLightingBuffer[4 * idx + 1] = finalDirectColor(1) / totalDirectWeight;
        lastRenderBuffer.directLightingBuffer[4 * idx + 2] = finalDirectColor(2) / totalDirectWeight;
        float newVariance = directVariance / (totalDirectWeight * totalDirectWeight);
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
        CudaFree();
        GBufferFree();
    }

    __host__ void RenderLoop(){
        bool firstFrame = true;
        using Clock = std::chrono::high_resolution_clock;
        while (!glfwWindowShouldClose(ui -> window))
        {
            RenderBuffer renderBuffer = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[0] : renderBuffers[1];
            RenderBuffer lastRenderBuffer = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[1] : renderBuffers[0];
            int temporalHistoryWriteIndex = denoiseParam.currentTemporalBufferIndex;
            int temporalHistoryReadIndex = 1 - temporalHistoryWriteIndex;
            RenderBuffer temporalHistoryCurrent = temporalHistoryBuffers[temporalHistoryWriteIndex];
            RenderBuffer temporalHistoryLast = temporalHistoryBuffers[temporalHistoryReadIndex];
            GBuffer gBuffer = renderParam.currentGBufferIndex == 0 ? gBuffers[0] : gBuffers[1];
            GBuffer lastGBuffer = renderParam.currentGBufferIndex == 0 ? gBuffers[1] : gBuffers[0];
            bool hardResetDenoiseHistory = false;
            ui -> GuiBegin(sppCounter, framebufferReset, renderParam, denoiseParam, denoiseTimingStats, hardResetDenoiseHistory);
            if(framebufferReset){
                sppCounter = 0;
                cudaMemset(renderBuffer.directLightingBuffer, 0, sizeof(float) * 4 * renderParam.width * renderParam.height);
                cudaMemset(renderBuffer.indirectLightingBuffer, 0, sizeof(float) * 4 * renderParam.width * renderParam.height);
                if(hardResetDenoiseHistory){
                    size_t historyBytes = sizeof(float) * 4 * renderParam.width * renderParam.height;
                    size_t momentBytes = sizeof(float2) * renderParam.width * renderParam.height;
                    size_t historyLenBytes = sizeof(int) * renderParam.width * renderParam.height;
                    cudaMemset(temporalHistoryBuffers[0].directLightingBuffer, 0, historyBytes);
                    cudaMemset(temporalHistoryBuffers[0].indirectLightingBuffer, 0, historyBytes);
                    cudaMemset(temporalHistoryBuffers[1].directLightingBuffer, 0, historyBytes);
                    cudaMemset(temporalHistoryBuffers[1].indirectLightingBuffer, 0, historyBytes);
                    for(int i = 0; i < 2; i++){
                        cudaMemset(denoiseParam.directMoments[i], 0, momentBytes);
                        cudaMemset(denoiseParam.indirectMoments[i], 0, momentBytes);
                        cudaMemset(denoiseParam.historyLength[i], 0, historyLenBytes);
                    }
                    denoiseParam.currentTemporalBufferIndex = 0;
                    firstFrame = true;
                }
                framebufferReset = false;
            }

            if(sppCounter < renderParam.spp || renderParam.denoise){
                auto frameStart = Clock::now();
                DenoiseTimingStats frameTimings = {};

                auto gbufferStart = Clock::now();
                RenderGBuffer(renderParam, cameraParam);
                auto gbufferEnd = Clock::now();
                sppCounter++;

                // Rendering pipeline start
                int width = renderParam.width, height = renderParam.height;
                int maxBounces = renderParam.maxBounces;
                int numSegments = width * height;
                auto pathTraceStart = Clock::now();
                // 1. Generate rays
                generateRayGridSize = dim3((width + generateRayBlockSize.x - 1) / generateRayBlockSize.x, (height + generateRayBlockSize.y - 1) / generateRayBlockSize.y);
                GenerateRayKernel<<<generateRayGridSize, generateRayBlockSize>>>(renderSegments, renderParam, cameraParam);
                cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(cudaGetLastError());

                renderGridSize = dim3((numSegments + renderBlockSize.x - 1) / renderBlockSize.x);

                for(int i = 0; i < maxBounces; i ++){
                    if(numSegments <= 0) break;
                    cudaMemset(intersections, 0, sizeof(IntersectionInfo) * numSegments);
                    cudaMemset(intersectionsBuffer, 0, sizeof(IntersectionInfo) * numSegments);
                    cudaMemset(materialIDs, 0, sizeof(int) * numSegments);

                    // 2. Intersection
                    renderGridSize = dim3((numSegments + renderBlockSize.x - 1) / renderBlockSize.x);
                    IntersectionKernel<<<renderGridSize, renderBlockSize>>>(renderSegments, renderParam, intersections, numSegments, segmentValidFlags, renderBuffer);
                    cudaDeviceSynchronize();
                    CHECK_CUDA_ERROR(cudaGetLastError());

                    // stream compaction
                    numSegments = StreamCompaction(segmentValidFlags, segmentPos, numSegments);
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
                }
                auto pathTraceEnd = Clock::now();
                
                // map cuda resource
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
                    auto temporalStart = Clock::now();
                    TemporalDenoiseKernel<<<denoiseGridSize, denoiseBlockSize>>>(denoiseParam, cameraParam, renderParam, renderBuffer, temporalHistoryLast, gBuffer, lastGBuffer, firstFrame);
                    cudaDeviceSynchronize();
                    CHECK_CUDA_ERROR(cudaGetLastError());
                    size_t temporalHistoryBytes = sizeof(float) * 4 * width * height;
                    cudaMemcpy(temporalHistoryCurrent.directLightingBuffer, renderBuffer.directLightingBuffer, temporalHistoryBytes, cudaMemcpyDeviceToDevice);
                    cudaMemcpy(temporalHistoryCurrent.indirectLightingBuffer, renderBuffer.indirectLightingBuffer, temporalHistoryBytes, cudaMemcpyDeviceToDevice);
                    auto temporalEnd = Clock::now();

                    // Computing Variance
                    auto varianceStart = Clock::now();
                    int varianceKernelSize = 7; // 7x7 kernel
                    RenderBuffer current = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[0] : renderBuffers[1];
                    RenderBuffer last = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[1] : renderBuffers[0];
                    ComputeVarianceKernel<<<denoiseGridSize, denoiseBlockSize>>>(denoiseParam, renderParam, current, last, gBuffer, varianceKernelSize);
                    cudaDeviceSynchronize();
                    CHECK_CUDA_ERROR(cudaGetLastError());
                    auto varianceEnd = Clock::now();
                    renderParam.currentRenderBufferIndex = 1 - renderParam.currentRenderBufferIndex;

                    // Remove albedo
                    // RemoveAlbedoKernel<<<denoiseGridSize, denoiseBlockSize>>>(renderParam, renderBuffer);

                    // Atrous denoising
                    frameTimings.atrousMs = 0.f;
                    if(!firstFrame){
                        int atrousPasses = max(1, min(8, denoiseParam.atrousIterations));
                        for(int i = 0; i < atrousPasses; i ++){
                            auto atrousStart = Clock::now();
                            int stepSize = 1 << i;
                            current = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[0] : renderBuffers[1];
                            last = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[1] : renderBuffers[0];
                            AtrousKernel<<<denoiseGridSize, denoiseBlockSize>>>(denoiseParam, cameraParam, renderParam, current, last, gBuffer, stepSize);
                            cudaDeviceSynchronize();
                            CHECK_CUDA_ERROR(cudaGetLastError());
                            auto atrousEnd = Clock::now();
                            frameTimings.atrousMs += std::chrono::duration<float, std::milli>(atrousEnd - atrousStart).count();
                            renderParam.currentRenderBufferIndex = 1 - renderParam.currentRenderBufferIndex;
                        }
                    }

                    // // Restore albedo
                    // RestoreAlbedoKernel<<<denoiseGridSize, denoiseBlockSize>>>(renderParam, renderBuffer);

                    frameTimings.temporalMs = std::chrono::duration<float, std::milli>(temporalEnd - temporalStart).count();
                    frameTimings.varianceMs = std::chrono::duration<float, std::milli>(varianceEnd - varianceStart).count();
                    frameTimings.svgfTotalMs = frameTimings.temporalMs + frameTimings.varianceMs + frameTimings.atrousMs;
                }

                // 5. Gather
                auto gatherStart = Clock::now();
                RenderBuffer currentRenderBuffer = renderParam.currentRenderBufferIndex == 0 ? renderBuffers[0] : renderBuffers[1];
                gatherGridSize = dim3((width + gatherBlockSize.x - 1) / gatherBlockSize.x, (height + gatherBlockSize.y - 1) / gatherBlockSize.y);
                GatherKernel<<<gatherGridSize, gatherBlockSize>>>(pixels, currentRenderBuffer, gBuffer, sppCounter, renderParam, cameraParam, denoiseParam);
                cudaDeviceSynchronize();
                auto gatherEnd = Clock::now();

                cudaGraphicsUnmapResources(1, &gBuffer.cudaPositionTriIDResource, 0);
                cudaGraphicsUnmapResources(1, &gBuffer.cudaNormalMatIDResource, 0);
                cudaGraphicsUnmapResources(1, &gBuffer.cudaBaryMeshIDResource, 0);
                cudaGraphicsUnmapResources(1, &gBuffer.cudaMotionDepthResource, 0);

                cudaGraphicsUnmapResources(1, &lastGBuffer.cudaPositionTriIDResource, 0);
                cudaGraphicsUnmapResources(1, &lastGBuffer.cudaNormalMatIDResource, 0);
                cudaGraphicsUnmapResources(1, &lastGBuffer.cudaBaryMeshIDResource, 0);
                cudaGraphicsUnmapResources(1, &lastGBuffer.cudaMotionDepthResource, 0);
                ui -> shader -> Unuse();
                firstFrame = false;
                frameTimings.gbufferMs = std::chrono::duration<float, std::milli>(gbufferEnd - gbufferStart).count();
                frameTimings.pathTraceMs = std::chrono::duration<float, std::milli>(pathTraceEnd - pathTraceStart).count();
                frameTimings.gatherMs = std::chrono::duration<float, std::milli>(gatherEnd - gatherStart).count();
                frameTimings.totalMs = std::chrono::duration<float, std::milli>(gatherEnd - frameStart).count();
                frameTimings.valid = renderParam.denoise;
                if(renderParam.denoise){
                    denoiseTimingStats = frameTimings;
                }
                // Rendering pipeline end
            }
            
            
            cudaDeviceSynchronize();
            ui -> UpdateTexture();
            ui -> RenderFrameBuffer();
            ui -> GuiEnd();
            glfwSwapBuffers(ui -> window);
            glfwSwapInterval(0);
            glfwPollEvents();

            if(renderParam.denoise){
                renderParam.currentRenderBufferIndex = 1 - renderParam.currentRenderBufferIndex;
                renderParam.currentGBufferIndex = 1 - renderParam.currentGBufferIndex;
                denoiseParam.currentTemporalBufferIndex = 1 - denoiseParam.currentTemporalBufferIndex;
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

        // Initialize render buffer
        cudaMalloc(&renderBuffers[0].directLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc(&renderBuffers[0].indirectLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc(&renderBuffers[1].directLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc(&renderBuffers[1].indirectLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc(&temporalHistoryBuffers[0].directLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc(&temporalHistoryBuffers[0].indirectLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc(&temporalHistoryBuffers[1].directLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc(&temporalHistoryBuffers[1].indirectLightingBuffer, sizeof(float) * 4 * width * height);
        cudaMemset(temporalHistoryBuffers[0].directLightingBuffer, 0, sizeof(float) * 4 * width * height);
        cudaMemset(temporalHistoryBuffers[0].indirectLightingBuffer, 0, sizeof(float) * 4 * width * height);
        cudaMemset(temporalHistoryBuffers[1].directLightingBuffer, 0, sizeof(float) * 4 * width * height);
        cudaMemset(temporalHistoryBuffers[1].indirectLightingBuffer, 0, sizeof(float) * 4 * width * height);

        // Initialize denoise buffer
        for(int i = 0; i < 2; i++){
            cudaMalloc(&denoiseParam.directMoments[i], sizeof(float2) * width * height);
            cudaMalloc(&denoiseParam.indirectMoments[i], sizeof(float2) * width * height);
            cudaMalloc(&denoiseParam.historyLength[i], sizeof(int) * width * height);
            cudaMemset(denoiseParam.directMoments[i], 0, sizeof(float2) * width * height);
            cudaMemset(denoiseParam.indirectMoments[i], 0, sizeof(float2) * width * height);
            cudaMemset(denoiseParam.historyLength[i], 0, sizeof(int) * width * height);
        }

        this -> pixels = ui -> pixels;
        this -> numBytes = ui -> numBytes;
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
                std::cout << "GBuffer FBO not complete!" << std::endl;

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
            gBuffer.cudaTexObjPositionTriID = CreateTextureObject(positionTriIDArray, addressMode, filterMode, readMode);
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
            gBuffer.cudaTexObjMotionDepth = CreateTextureObject(motionDepthArray, addressMode, filterMode, readMode);
            cudaGraphicsUnmapResources(1, &gBuffer.cudaMotionDepthResource, 0);
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
            gBufferShader -> SetUniformInt("uMeshID", mesh -> meshID + 1); 
            
            // Bind VAO and draw
            glBindVertexArray(mesh->vao);
            glDrawArrays(GL_TRIANGLES, 0, mesh->numTriangles * 3);
            glBindVertexArray(0);
        }
        gBufferShader -> Unuse();
        glDisable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    void CudaFree(){
        cudaFree(renderSegments);
        cudaFree(renderSegmentsBuffer);
        cudaFree(intersections);
        cudaFree(intersectionsBuffer);
        cudaFree(segmentValidFlags);
        cudaFree(segmentPos);
        cudaFree(materialIDs);
        cudaFree(materialSortIndices);
        cudaFree(renderBuffers[0].directLightingBuffer);
        cudaFree(renderBuffers[0].indirectLightingBuffer);
        cudaFree(renderBuffers[1].directLightingBuffer);
        cudaFree(renderBuffers[1].indirectLightingBuffer);
        cudaFree(temporalHistoryBuffers[0].directLightingBuffer);
        cudaFree(temporalHistoryBuffers[0].indirectLightingBuffer);
        cudaFree(temporalHistoryBuffers[1].directLightingBuffer);
        cudaFree(temporalHistoryBuffers[1].indirectLightingBuffer);
        for(int i = 0; i < 2; i++){
            cudaFree(denoiseParam.directMoments[i]);
            cudaFree(denoiseParam.indirectMoments[i]);
            cudaFree(denoiseParam.historyLength[i]);
        }
    }
    void GBufferFree(){
        for(int i = 0; i < 2; i++){
            glDeleteTextures(1, &gBuffers[i].texPositionTriID);
            glDeleteTextures(1, &gBuffers[i].texNormalMatID);
            glDeleteTextures(1, &gBuffers[i].texBaryMeshID);
            glDeleteTextures(1, &gBuffers[i].texMotionDepth);
            glDeleteRenderbuffers(1, &gBuffers[i].depthBuffer);
            glDeleteFramebuffers(1, &gBuffers[i].fbo);
        }
        delete gBufferShader;
    }
    UI *ui;
    uchar4* pixels;
    RenderBuffer renderBuffers[2];
    RenderBuffer temporalHistoryBuffers[2];
    GBuffer gBuffers[2];
    DenoiseParam denoiseParam;
    DenoiseTimingStats denoiseTimingStats;
    size_t numBytes;
    int sppCounter = 0;
    bool framebufferReset = false;
    Shader *gBufferShader;
};
