#include "scene.h"
#include "ui.h"
#include "light.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "camera.h"
#include "lightmanager.h"

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

struct RenderBuffer{
    float *framebuffer;
    float *lastFrame;
    float4 *normalDepthBuffer; // xyzw: normal.xyz, depth.w
    float *positionBuffer;
    float *albedoBuffer;
    int *idBuffer;
    int *lastIdBuffer;
    cudaTextureObject_t normalDepthTexture;
};

struct DenoiseParam{
    float2 *moments;
    float *variance;
    int *historyLength;
    int maxHistoryLength = 32;
    float sigmaLight = 4.f;
    float sigmaNormal = 128.f;
    float sigmaDepth = 1.f;
};

inline DenoiseParam denoiseParam;

cudaTextureObject_t CreateTextureObject(float4 *data, int width, int height)
{
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
    resDesc.res.linear.sizeInBytes = width * height * sizeof(float4);

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t textureObject = 0;
    cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr);
    return textureObject;
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
    segments[idx].color = Vec3f(0.0f, 0.0f, 0.0f);
    segments[idx].weight = Vec3f(1.0f, 1.0f, 1.0f);
    segments[idx].firstBounce = true; // Initialize first bounce flag
}

__device__ void AccumulateColor(int index, Vec3f color, float *accumulator)
{
    accumulator[index * 3 + 0] += color(0);
    accumulator[index * 3 + 1] += color(1);
    accumulator[index * 3 + 2] += color(2);
}

__device__ bool TraceRay(Ray &ray, BVH *bvh, IntersectionInfo &info)
{
    info.hitTime = FLOATMAX;
    info.hit = false;
    bool hit = bvh->IsIntersect(ray, info);
    return hit;
}

__global__ void IntersectionKernel(RenderSegment *segments, RenderParam renderParam, IntersectionInfo *intersectionInfo, int numSegments, int *segmentValidFlags, RenderBuffer renderBuffer, bool renderGBuffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSegments)
        return;
    BVH *bvh = renderParam.bvh;
    Ray ray = segments[idx].ray; // Get the ray for this pixel
    float *framebuffer = renderBuffer.framebuffer;
    bool hit = TraceRay(ray, bvh, intersectionInfo[idx]);
    if(!hit){
        // If no hit, accumulate background color (black here)
        int index = segments[idx].index;
        LightManager *lightManager = renderParam.lightManager;
        // printf("lightManager env idx: %d\n", lightManager -> envMapLightIdx);
        if(lightManager -> envMapLightIdx != -1){
            Light *envLight = lightManager -> GetLight(lightManager -> envMapLightIdx);
            Vec3f bgColor = envLight -> Emission(ray.direction);
            segments[idx].color += segments[idx].weight.cwiseProduct(bgColor);
        }
        AccumulateColor(index, segments[idx].color, framebuffer);
        segments[idx].remainingBounces = 0; // Terminate the path
        segmentValidFlags[idx] = 0;
        if(renderGBuffer){
            // GBuffer output for background
            int width = renderParam.width;
            int height = renderParam.height;
            int pixelIndex = segments[idx].index;
            // id
            renderBuffer.idBuffer[pixelIndex] = 0; // background id = 0
            // position
            renderBuffer.positionBuffer[3 * pixelIndex + 0] = 0.f;
            renderBuffer.positionBuffer[3 * pixelIndex + 1] = 0.f;
            renderBuffer.positionBuffer[3 * pixelIndex + 2] = 0.f;
        }
    }else {
        segmentValidFlags[idx] = 1;
        if(renderGBuffer){
            // GBuffer output
            int width = renderParam.width;
            int height = renderParam.height;
            int pixelIndex = segments[idx].index;
            // depth
            intersectionInfo[idx].hitTime = fminf(intersectionInfo[idx].hitTime, FLOATMAX);
            float depth = intersectionInfo[idx].hitTime;
            renderBuffer.normalDepthBuffer[pixelIndex].w = depth;
            // normal
            Vec3f normal = intersectionInfo[idx].normal.normalized();
            renderBuffer.normalDepthBuffer[pixelIndex].x = normal(0);
            renderBuffer.normalDepthBuffer[pixelIndex].y = normal(1);
            renderBuffer.normalDepthBuffer[pixelIndex].z = normal(2);
            // id
            renderBuffer.idBuffer[pixelIndex] = intersectionInfo[idx].meshID + 1; // +1 to distinguish from background
            // position
            Vec3f position = intersectionInfo[idx].hitPoint;
            renderBuffer.positionBuffer[3 * pixelIndex + 0] = position(0);
            renderBuffer.positionBuffer[3 * pixelIndex + 1] = position(1);
            renderBuffer.positionBuffer[3 * pixelIndex + 2] = position(2);
            // albedo
            Material *materials = renderParam.materials;
            MeshData *meshData = renderParam.meshData;
            Material &material = materials[meshData[intersectionInfo[idx].meshID].materialID];
            Vec3f albedo = material.basecolor;
            renderBuffer.albedoBuffer[3 * pixelIndex + 0] = albedo(0);
            renderBuffer.albedoBuffer[3 * pixelIndex + 1] = albedo(1);
            renderBuffer.albedoBuffer[3 * pixelIndex + 2] = albedo(2);
        }
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
            segment.color += material.ke; // If it's the first bounce, add emission directly
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
            segment.color += segment.weight.cwiseProduct(material.ke) * brdfMisWeight;
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
            segment.color += lightMisWeight * segment.weight.cwiseProduct(dirL);
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
                segment.color += lightMisWeight * segment.weight.cwiseProduct(dirL) * FtransmitOut;
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
    if(segmentValidFlags[idx] == 0)
        AccumulateColor(segments[idx].index, segments[idx].color, renderBuffer.framebuffer);
    segments[idx].remainingBounces--;
}

__device__ void WorldToLastFramePos(Vec3f worldPosition, CameraParam cameraParam, RenderParam renderParam, Vec2f &pixelPos)
{
    Vec3f lastCamPos = cameraParam.lastPosition;
    Vec3f lastLookat = cameraParam.lastLookat;
    Vec3f lastUp = cameraParam.lastUp;

    // view matrix
    Mat4f viewMatrix = LookAt(lastCamPos, lastLookat, lastUp);
    // projection matrix
    float aspect = cameraParam.ratio;
    float fovy = cameraParam.fovy;
    Bounds3D sceneBound = renderParam.sceneBounds;
    float r = sceneBound.DiagonalLength() * 0.5f;
    float nearPlane = 0.1f;
    float farPlane = (sceneBound.Center() - lastCamPos).norm() + 3 * r;
    // float farPlane = 1000.0f;
    Mat4f projectionMatrix = Perspective(fovy, aspect, nearPlane, farPlane);

    Mat4f vpMatrix = projectionMatrix * viewMatrix;
    Vec4f clipPos = vpMatrix * Vec4f(worldPosition(0), worldPosition(1), worldPosition(2), 1.0f);
    Vec3f ndcPos = Vec3f(clipPos(0) / clipPos(3), clipPos(1) / clipPos(3), clipPos(2) / clipPos(3));
    pixelPos(0) = 0.5f * (ndcPos(0) + 1.0f) * renderParam.width;
    pixelPos(1) = 0.5f * (ndcPos(1) + 1.0f) * renderParam.width;
    // pixelPos(1) = 0.5f * (1.0f - ndcPos(1)) * renderParam.height;
    // pixelPos(1) = (1.0f - (ndcPos(1) * 0.5f + 0.5f)) * renderParam.height;
}

__device__ bool GetLastFramePos(int index, CameraParam cameraParam, RenderParam renderParam, RenderBuffer renderBuffer, Vec2f &pixelPos)
{
    Vec3f worldPos = Vec3f(
        renderBuffer.positionBuffer[3 * index + 0],
        renderBuffer.positionBuffer[3 * index + 1],
        renderBuffer.positionBuffer[3 * index + 2]
    );

    if(worldPos.norm() < eps && renderBuffer.lastIdBuffer[index] == 0){
        pixelPos = Vec2f(-1, -1);
        return false;
    }

    WorldToLastFramePos(worldPos, cameraParam, renderParam, pixelPos);
    int width = renderParam.width, height = renderParam.height;
    if(pixelPos(0) < 0 || pixelPos(0) >= width || pixelPos(1) < 0 || pixelPos(1) >= height){
        pixelPos = Vec2f(-1, -1);
        return false;
    }
    int id = renderBuffer.idBuffer[index];
    int lastPixelIndex = static_cast<int>(pixelPos(1)) * width + static_cast<int>(pixelPos(0));
    
    int lastId = renderBuffer.lastIdBuffer[lastPixelIndex];
    if(id != lastId){
        // printf("id mismatch: %d vs %d, current index: %d, last pixel index: %d, last x: %f, last y: %f\n", id, lastId, index, lastPixelIndex, pixelPos(0), pixelPos(1));
        return false;
    }
        

    return true;

}

__device__ void TemporalDenoising(int index, float *currentFrame, float *lastFrame, DenoiseParam denoiseParam, CameraParam cameraParam, RenderParam renderParam, RenderBuffer renderBuffer)
{
    Vec2f pixelPos;
    bool valid = GetLastFramePos(index, cameraParam, renderParam, renderBuffer, pixelPos);

    denoiseParam.historyLength[index] = valid ? min(denoiseParam.historyLength[index] + 1, denoiseParam.maxHistoryLength) : 1;
    float alpha = valid ? 1.f / denoiseParam.historyLength[index] : 1.0f;
    int lastFrameIndex = static_cast<int>(pixelPos(1)) * renderParam.width + static_cast<int>(pixelPos(0));
    if(!valid){
        lastFrameIndex = index;
        denoiseParam.historyLength[index] = 0;
    }
    currentFrame[index * 3 + 0] = alpha * currentFrame[index * 3 + 0] + (1 - alpha) * lastFrame[lastFrameIndex * 3 + 0];
    currentFrame[index * 3 + 1] = alpha * currentFrame[index * 3 + 1] + (1 - alpha) * lastFrame[lastFrameIndex * 3 + 1];
    currentFrame[index * 3 + 2] = alpha * currentFrame[index * 3 + 2] + (1 - alpha) * lastFrame[lastFrameIndex * 3 + 2];

    // update moments
    float2 &moment = denoiseParam.moments[index];
    float2 lastMoment = valid ? denoiseParam.moments[lastFrameIndex] : make_float2(1000.f, 1000000.f);
    float luminance = Luminance(Vec3f(currentFrame[index * 3 + 0], currentFrame[index * 3 + 1], currentFrame[index * 3 + 2]));
    moment.x = alpha * luminance + (1 - alpha) * lastMoment.x;
    moment.y = alpha * luminance * luminance + (1 - alpha) * lastMoment.y;
    // printf("index: %d, valid: %d, historyLength: %d, alpha: %f, luminance: %f, moment.x: %f, moment.y: %f\n", index, valid, denoiseParam.historyLength[index], alpha, luminance, moment.x, moment.y);
}

__global__ void TemporalDenoiseKernel(DenoiseParam denoiseParam, CameraParam cameraParam, RenderParam renderParam, RenderBuffer renderBuffer)
{
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    int idx = j * width + i;
    float *currentFrame = renderBuffer.framebuffer;
    float *lastFrame = renderBuffer.lastFrame;
    TemporalDenoising(idx, currentFrame, lastFrame, denoiseParam, cameraParam, renderParam, renderBuffer);
}

__global__ void ComputeVarianceKernel(DenoiseParam denoiseParam, RenderParam renderParam, int kernelSize)
{
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    int idx = j * width + i;
    float2 moment = denoiseParam.moments[idx];
    float mean = moment.x;
    float mean2 = moment.y;
    if(mean < eps && mean2 < eps){
        denoiseParam.variance[idx] = 0.f;
        return;
    }
    float var = 0;
    float weight = 0;
    int halfKernel = kernelSize / 2;
    float neighborMean = 0.f, neighborMean2 = 0.f, neighborVar = 0.f;
    float weightSum = 0.f;
    for(int kj = -halfKernel; kj <= halfKernel; kj++){
        for(int ki = -halfKernel; ki <= halfKernel; ki++){
            int ni = i + ki;
            int nj = j + kj;
            if(ni >= 0 && ni < width && nj >= 0 && nj < height){
                int nidx = nj * width + ni;
                float2 neighborMoment = denoiseParam.moments[nidx];
                neighborMean += neighborMoment.x;
                neighborMean2 += neighborMoment.y;
                weightSum += 1.f;
            }
        }
    }
    var =  neighborMean2 / weightSum - (neighborMean / weightSum) * (neighborMean / weightSum);
    if(var < 0.f){
        // printf("idx: %d, mean: %f, mean2: %f, neighborMean: %f, neighborMean2: %f, var: %f\n", idx, mean, mean2, neighborMean / weightSum, neighborMean2 / weightSum, var);
        var = 0.f;
    }
    
    denoiseParam.variance[idx] = var;
}

__global__ void RemoveAlbedoKernel(RenderParam renderParam, RenderBuffer renderBuffer)
{
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    int idx = j * width + i;

    // remove albedo
    float3 albedo = make_float3(
        renderBuffer.albedoBuffer[3 * idx + 0],
        renderBuffer.albedoBuffer[3 * idx + 1],
        renderBuffer.albedoBuffer[3 * idx + 2]
    );

    float3 color = make_float3(
        renderBuffer.framebuffer[3 * idx + 0],
        renderBuffer.framebuffer[3 * idx + 1],
        renderBuffer.framebuffer[3 * idx + 2]
    );

    if(albedo.x > eps)color.x /= albedo.x;
    if(albedo.y > eps)color.y /= albedo.y;
    if(albedo.z > eps)color.z /= albedo.z;

    renderBuffer.framebuffer[3 * idx + 0] = color.x;
    renderBuffer.framebuffer[3 * idx + 1] = color.y;
    renderBuffer.framebuffer[3 * idx + 2] = color.z;
}

__global__ void RestoreAlbedoKernel(RenderParam renderParam, RenderBuffer renderBuffer)
{
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    int idx = j * width + i;

    // restore albedo
    float3 albedo = make_float3(
        renderBuffer.albedoBuffer[3 * idx + 0],
        renderBuffer.albedoBuffer[3 * idx + 1],
        renderBuffer.albedoBuffer[3 * idx + 2]
    );

    float3 color = make_float3(
        renderBuffer.framebuffer[3 * idx + 0],
        renderBuffer.framebuffer[3 * idx + 1],
        renderBuffer.framebuffer[3 * idx + 2]
    );

    if(albedo.x > eps)color.x *= albedo.x;
    if(albedo.y > eps)color.y *= albedo.y;
    if(albedo.z > eps)color.z *= albedo.z;

    renderBuffer.framebuffer[3 * idx + 0] = color.x;
    renderBuffer.framebuffer[3 * idx + 1] = color.y;
    renderBuffer.framebuffer[3 * idx + 2] = color.z;
}

__global__ void AtrousKernel(DenoiseParam denoiseParam, CameraParam cameraParam, RenderParam renderParam, RenderBuffer renderBuffer, int stepSize){
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    int idx = j * width + i;

    Vec3f centerColor = Vec3f(
        renderBuffer.framebuffer[3 * idx + 0],
        renderBuffer.framebuffer[3 * idx + 1],
        renderBuffer.framebuffer[3 * idx + 2]
    );
    float centerLuminance = Luminance(centerColor);
    Vec3f centerNormal = Vec3f(
        renderBuffer.normalDepthBuffer[idx].x,
        renderBuffer.normalDepthBuffer[idx].y,
        renderBuffer.normalDepthBuffer[idx].z
    ).normalized();
    float centerDepth = renderBuffer.normalDepthBuffer[idx].w;
    float centerVariance = denoiseParam.variance[idx];

    Vec3f finalColor = Vec3f(0.f, 0.f, 0.f);
    float totalWeight = 0.f;
    int kernelRadius = 1;
    float sigmaLight = denoiseParam.sigmaLight;
    float sigmaNormal = denoiseParam.sigmaNormal;
    float sigmaDepth = denoiseParam.sigmaDepth;

    // compute dz/dx and dz/dy
    float dzdx = 0.f, dzdy = 0.f;
    float depthLeft = (i > 0) ? renderBuffer.normalDepthBuffer[j * width + (i - 1)].w : centerDepth;
    float depthRight = (i < width - 1) ? renderBuffer.normalDepthBuffer[j * width + (i + 1)].w : centerDepth;
    float depthUp = (j > 0) ? renderBuffer.normalDepthBuffer[(j - 1) * width + i].w : centerDepth;
    float depthDown = (j < height - 1) ? renderBuffer.normalDepthBuffer[(j + 1) * width + i].w : centerDepth;
    dzdx = (depthRight - depthLeft) * 0.5f;
    dzdy = (depthDown - depthUp) * 0.5f;
    float gradientMagnitude = sqrtf(dzdx * dzdx + dzdy * dzdy) + eps;

    // for(int kj = -kernelRadius; kj <= kernelRadius; kj++){
    //     for(int ki = -kernelRadius; ki <= kernelRadius; ki++){
    //         int ni = i + ki * stepSize;
    //         int nj = j + kj * stepSize;
    //         if(ni >= 0 && ni < width && nj >= 0 && nj < height){
    //             int nidx = nj * width + ni;
    //             Vec3f neighborColor = Vec3f(
    //                 renderBuffer.framebuffer[3 * nidx + 0],
    //                 renderBuffer.framebuffer[3 * nidx + 1],
    //                 renderBuffer.framebuffer[3 * nidx + 2]
    //             );
    //             float neighborLuminance = Luminance(neighborColor);
    //             Vec3f neighborNormal = Vec3f(
    //                 renderBuffer.normalDepthBuffer[nidx].x,
    //                 renderBuffer.normalDepthBuffer[nidx].y,
    //                 renderBuffer.normalDepthBuffer[nidx].z
    //             ).normalized();
    //             float neighborDepth = renderBuffer.normalDepthBuffer[nidx].w;

    //             // compute weights
    //             float colorWeight = expf(- (centerLuminance - neighborLuminance) * (centerLuminance - neighborLuminance) / (sigmaLight * sigmaLight * centerVariance + eps));
    //             float normalWeight = powf(fmaxf(0.f, centerNormal.dot(neighborNormal)), sigmaNormal);
    //             float depthWeight = expf(- (fabsf(centerDepth - neighborDepth)) / (gradientMagnitude * sigmaDepth + eps));

    //             // colorWeight = 1;
    //             // normalWeight = 1;
    //             // depthWeight = 1;

    //             // printf("centerColor: %f, %f, %f; neighborColor: %f, %f, %f; colorWeight: %f, variance: %f\n", centerColor(0), centerColor(1), centerColor(2), neighborColor(0), neighborColor(1), neighborColor(2), colorWeight, centerVariance);
    //             // printf("colorWeight: %f, normalWeight: %f, depthWeight: %f\n", colorWeight, normalWeight, depthWeight);
    //             // if(centerNormal.norm() > eps && neighborNormal.norm() > eps)
    //             //     printf("centerNomral: %f, %f, %f; neighborNormal: %f, %f, %f; dot value: %f, normalWeight: %f\n", centerNormal(0), centerNormal(1), centerNormal(2), neighborNormal(0), neighborNormal(1), neighborNormal(2), centerNormal.dot(neighborNormal), normalWeight);

    //             float weight = colorWeight * normalWeight * depthWeight;

    //             finalColor += weight * neighborColor;
    //             totalWeight += weight;
    //         }
    //     }
    // }

    // using 5 * 1 && 1 * 5 kernel to approximate 5 * 5 kernel
    for(int ki = -kernelRadius; ki <= kernelRadius; ki++){
        int ni = i + ki * stepSize;
        int nj = j;
        if(ni >= 0 && ni < width){
            int nidx = nj * width + ni;
            Vec3f neighborColor = Vec3f(
                renderBuffer.framebuffer[3 * nidx + 0],
                renderBuffer.framebuffer[3 * nidx + 1],
                renderBuffer.framebuffer[3 * nidx + 2]
            );
            float neighborLuminance = Luminance(neighborColor);
            Vec3f neighborNormal = Vec3f(
                renderBuffer.normalDepthBuffer[nidx].x,
                renderBuffer.normalDepthBuffer[nidx].y,
                renderBuffer.normalDepthBuffer[nidx].z
            ).normalized();
            float neighborDepth = renderBuffer.normalDepthBuffer[nidx].w;

            // compute weights
            float colorWeight = expf(- (centerLuminance - neighborLuminance) * (centerLuminance - neighborLuminance) / (sigmaLight * sigmaLight * centerVariance + eps));
            float normalWeight = powf(fmaxf(0.f, centerNormal.dot(neighborNormal)), sigmaNormal);
            float depthWeight = expf(- (fabsf(centerDepth - neighborDepth)) / (gradientMagnitude * sigmaDepth + eps));

            float weight = colorWeight * normalWeight * depthWeight;

            finalColor += weight * neighborColor;
            totalWeight += weight;
        }
    }
    for(int kj = -kernelRadius; kj <= kernelRadius; kj++){
        int ni = i;
        int nj = j + kj * stepSize;
        if(nj >= 0 && nj < height){
            int nidx = nj * width + ni;
            Vec3f neighborColor = Vec3f(
                renderBuffer.framebuffer[3 * nidx + 0],
                renderBuffer.framebuffer[3 * nidx + 1],
                renderBuffer.framebuffer[3 * nidx + 2]
            );
            float neighborLuminance = Luminance(neighborColor);
            Vec3f neighborNormal = Vec3f(
                renderBuffer.normalDepthBuffer[nidx].x,
                renderBuffer.normalDepthBuffer[nidx].y,
                renderBuffer.normalDepthBuffer[nidx].z
            ).normalized();
            float neighborDepth = renderBuffer.normalDepthBuffer[nidx].w;

            // compute weights
            float colorWeight = expf(- (centerLuminance - neighborLuminance) * (centerLuminance - neighborLuminance) / (sigmaLight * sigmaLight * centerVariance + eps));
            float normalWeight = powf(fmaxf(0.f, centerNormal.dot(neighborNormal)), sigmaNormal);
            float depthWeight = expf(- (fabsf(centerDepth - neighborDepth)) / (gradientMagnitude * sigmaDepth + eps));

            float weight = colorWeight * normalWeight * depthWeight;

            finalColor += weight * neighborColor;
            totalWeight += weight;
        }
    }
    if(totalWeight < eps){
        finalColor = centerColor;
    }else{
        finalColor /= totalWeight;
        // if(finalColor(0) < eps && finalColor(1) < eps && finalColor(2) < eps)printf("totalWeight: %f, finalColor: %f, %f, %f, centerColor: %f, %f, %f\n", totalWeight, finalColor(0), finalColor(1), finalColor(2), centerColor(0), centerColor(1), centerColor(2));
    }

    

    renderBuffer.lastFrame[3 * idx + 0] = finalColor(0);
    renderBuffer.lastFrame[3 * idx + 1] = finalColor(1);
    renderBuffer.lastFrame[3 * idx + 2] = finalColor(2);
}

__global__ void GatherKernel(uchar4 *pixels, RenderBuffer renderBuffer,int spp, RenderParam renderParam, CameraParam cameraParam, DenoiseParam denoiseParam)
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
    float *frameBuffer = renderBuffer.framebuffer;
    float *lastFrame = renderBuffer.lastFrame;
    if(renderTargetMode == 0){
        // color
        if(renderParam.denoise){
            // TemporalDenoising(idx, frameBuffer, lastFrame, denoiseParam, cameraParam, renderParam, renderBuffer);
            pixels[segIndex] = make_uchar4(
                static_cast<unsigned char>(fminf(fmaxf(powf(frameBuffer[3 * idx], 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                static_cast<unsigned char>(fminf(fmaxf(powf(frameBuffer[3 * idx + 1], 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                static_cast<unsigned char>(fminf(fmaxf(powf(frameBuffer[3 * idx + 2], 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                255);
        }else{
            // using accumulated framebuffer
            Vec3f color = Vec3f(frameBuffer[3 * idx] / spp, frameBuffer[3 * idx + 1] / spp, frameBuffer[3 * idx + 2] / spp);
            pixels[segIndex] = make_uchar4(
                static_cast<unsigned char>(fminf(fmaxf(powf(color(0), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                static_cast<unsigned char>(fminf(fmaxf(powf(color(1), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                static_cast<unsigned char>(fminf(fmaxf(powf(color(2), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
                255);
        }
       
    }else if(renderTargetMode == 1){
        // depth
        float depth = renderBuffer.normalDepthBuffer[idx].w;
        Bounds3D sceneBound = renderParam.sceneBounds;
        Vec3f cameraPos = cameraParam.position;
        float r = sceneBound.DiagonalLength() * 0.5f;
        float nearPlane = (sceneBound.Center() - cameraPos).norm() - r;
        float farPlane = (sceneBound.Center() - cameraPos).norm() + r;
        float uniformDepth = (depth - nearPlane) / (farPlane - nearPlane);
        unsigned char depthUC = static_cast<unsigned char>(fminf(fmaxf(uniformDepth * 255.0f, 0.0f), 255.0f));
        pixels[segIndex] = make_uchar4(depthUC, depthUC, depthUC, 255);
    }else if(renderTargetMode == 2){
        // normal
        float nx = renderBuffer.normalDepthBuffer[idx].x;
        float ny = renderBuffer.normalDepthBuffer[idx].y;
        float nz = renderBuffer.normalDepthBuffer[idx].z;
        pixels[segIndex] = make_uchar4(
            static_cast<unsigned char>(fminf(fmaxf((nx * 0.5f + 0.5f) * 255.0f, 0.0f), 255.0f)),
            static_cast<unsigned char>(fminf(fmaxf((ny * 0.5f + 0.5f) * 255.0f, 0.0f), 255.0f)),
            static_cast<unsigned char>(fminf(fmaxf((nz * 0.5f + 0.5f) * 255.0f, 0.0f), 255.0f)),
            255);
    }else if(renderTargetMode == 3){
        // id
        int id = renderBuffer.idBuffer[idx];
        // int id = renderBuffer.lastIdBuffer[idx];
        // printf("pixel (%d, %d) id: %d\n", i, j, id);
        // Map the id to a color
        unsigned char r = (id * 37) % 256;
        unsigned char g = (id * 57) % 256;
        unsigned char b = (id * 97) % 256;
        pixels[segIndex] = make_uchar4(r, g, b, 255);
    }else if(renderTargetMode == 4){
        // position
        float px = renderBuffer.positionBuffer[3 * idx + 0];
        float py = renderBuffer.positionBuffer[3 * idx + 1];
        float pz = renderBuffer.positionBuffer[3 * idx + 2];
        // Map position to color (simple normalization based on scene bounds)
        Bounds3D sceneBound = renderParam.sceneBounds;
        Vec3f minPos = sceneBound.min;
        Vec3f maxPos = sceneBound.max;
        unsigned char r = static_cast<unsigned char>(fminf(fmaxf((px - minPos(0)) / (maxPos(0) - minPos(0)) * 255.0f, 0.0f), 255.0f));
        unsigned char g = static_cast<unsigned char>(fminf(fmaxf((py - minPos(1)) / (maxPos(1) - minPos(1)) * 255.0f, 0.0f), 255.0f));
        unsigned char b = static_cast<unsigned char>(fminf(fmaxf((pz - minPos(2)) / (maxPos(2) - minPos(2)) * 255.0f, 0.0f), 255.0f));
        pixels[segIndex] = make_uchar4(r, g, b, 255);
    }else if(renderTargetMode == 5){
        // albedo
        float ax = renderBuffer.albedoBuffer[3 * idx + 0];
        float ay = renderBuffer.albedoBuffer[3 * idx + 1];
        float az = renderBuffer.albedoBuffer[3 * idx + 2];
        pixels[segIndex] = make_uchar4(
            static_cast<unsigned char>(fminf(fmaxf(powf(ax, 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
            static_cast<unsigned char>(fminf(fmaxf(powf(ay, 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
            static_cast<unsigned char>(fminf(fmaxf(powf(az, 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
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
    }
    __host__ ~Renderer(){

    }

    __host__ void RenderLoop(){
        bool renderGBuffer = true;
        while (!glfwWindowShouldClose(ui -> window))
        {
            // LOG_DEBUG("here");
            ui -> GuiBegin(sppCounter, framebufferReset);
            if(framebufferReset){
                sppCounter = 0;
                cudaMemset(renderBuffer.framebuffer, 0, sizeof(float) * 3 * renderParam.width * renderParam.height);
                cudaMemset(renderBuffer.normalDepthBuffer, 0, sizeof(float) * 4 * renderParam.width * renderParam.height);
                cudaMemset(renderBuffer.idBuffer, 0, sizeof(int) * renderParam.width * renderParam.height);
                cudaMemset(renderBuffer.positionBuffer, 0, sizeof(float) * 3 * renderParam.width * renderParam.height);
                cudaMemset(renderBuffer.albedoBuffer, 0, sizeof(float) * 3 * renderParam.width * renderParam.height);
                framebufferReset = false;
                renderGBuffer = true;
            }

            if(renderParam.denoise){
                // reset framebuffer when denoising is enabled
                cudaMemset(renderBuffer.framebuffer, 0, sizeof(float) * 3 * renderParam.width * renderParam.height);
            }
        
            if(sppCounter < renderParam.spp || renderParam.denoise){
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
                    IntersectionKernel<<<renderGridSize, renderBlockSize>>>(renderSegments, renderParam, intersections, numSegments, segmentValidFlags, renderBuffer, renderGBuffer);
                    cudaDeviceSynchronize();
                    CHECK_CUDA_ERROR(cudaGetLastError());
                    renderGBuffer = false;

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

                // 4. Denoising
                if(renderParam.denoise){
                    denoiseGridSize = dim3((width + denoiseBlockSize.x - 1) / denoiseBlockSize.x, (height + denoiseBlockSize.y - 1) / denoiseBlockSize.y);
                    // Temporal denoising
                    TemporalDenoiseKernel<<<denoiseGridSize, denoiseBlockSize>>>(denoiseParam, cameraParam, renderParam, renderBuffer);

                    // Computing Variance
                    int varianceKernelSize = 7; // 7x7 kernel
                    ComputeVarianceKernel<<<denoiseGridSize, denoiseBlockSize>>>(denoiseParam, renderParam, varianceKernelSize);

                    // Remove albedo
                    RemoveAlbedoKernel<<<denoiseGridSize, denoiseBlockSize>>>(renderParam, renderBuffer);

                    // Atrous denoising
                    for(int i = 0; i < 5; i ++){
                        int stepSize = 1 << i;
                        AtrousKernel<<<denoiseGridSize, denoiseBlockSize>>>(denoiseParam, cameraParam, renderParam, renderBuffer, stepSize);
                        cudaDeviceSynchronize();
                        CHECK_CUDA_ERROR(cudaGetLastError());
                        std::swap(renderBuffer.framebuffer, renderBuffer.lastFrame);
                    }

                    // std::swap(renderBuffer.framebuffer, renderBuffer.lastFrame);

                    // Restore albedo
                    RestoreAlbedoKernel<<<denoiseGridSize, denoiseBlockSize>>>(renderParam, renderBuffer);

                    cudaDeviceSynchronize();
                    CHECK_CUDA_ERROR(cudaGetLastError());
                }

                // 5. Gather
                gatherGridSize = dim3((width + gatherBlockSize.x - 1) / gatherBlockSize.x, (height + gatherBlockSize.y - 1) / gatherBlockSize.y);
                GatherKernel<<<gatherGridSize, gatherBlockSize>>>(pixels, renderBuffer, sppCounter, renderParam, cameraParam, denoiseParam);
                cudaDeviceSynchronize();
                if(renderParam.denoise){
                    // save last id buffer
                    CHECK_CUDA_ERROR(cudaMemcpy(renderBuffer.lastIdBuffer, renderBuffer.idBuffer, sizeof(int) * width * height, cudaMemcpyDeviceToDevice));
                    cudaDeviceSynchronize();
                }
                CHECK_CUDA_ERROR(cudaGetLastError());
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

            // conserved last frame
            if(renderParam.denoise){
                std::swap(renderBuffer.framebuffer, renderBuffer.lastFrame);
            }
            // break;
        }
    }

    void CudaInit()
    {
        // Initialize
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
        cudaMalloc((void **)&denoiseParam.moments, sizeof(float2) * width * height);
        cudaMalloc((void **)&denoiseParam.historyLength, sizeof(int) * width * height);
        cudaMalloc((void**)&denoiseParam.variance, sizeof(float) * width * height);
        cudaMalloc((void**)&renderBuffer.albedoBuffer, sizeof(float) * 3 * width * height);

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
        cudaMalloc((void**)&renderBuffer.framebuffer, sizeof(float) * 3 * width * height);
        cudaMalloc((void**)&renderBuffer.lastFrame, sizeof(float) * 3 * width * height);
        cudaMalloc((void**)&renderBuffer.normalDepthBuffer, sizeof(float) * 4 * width * height);
        cudaMalloc((void**)&renderBuffer.idBuffer, sizeof(int) * width * height);
        cudaMalloc((void**)&renderBuffer.lastIdBuffer, sizeof(int) * width * height);
        cudaMalloc((void**)&renderBuffer.positionBuffer, sizeof(float) * width * height * 3);

        // bind textures

    }

    UI *ui;
    uchar4* pixels;
    RenderBuffer renderBuffer;
    size_t numBytes;
    int sppCounter = 0;
    bool framebufferReset = false;
};