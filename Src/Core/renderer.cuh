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
float *accumulator;
int *segmentValidFlags;
int *segmentPos;

dim3 generateRayBlockSize(32, 32);
dim3 renderBlockSize(128);
dim3 gatherBlockSize(32, 32);
dim3 generateRayGridSize;
dim3 renderGridSize;
dim3 gatherGridSize;


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

__global__ void IntersectionKernel(RenderSegment *segments, RenderParam renderParam, IntersectionInfo *intersectionInfo, int numSegments, int *segmentValidFlags, float *accumulator)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSegments)
        return;
    BVH *bvh = renderParam.bvh;
    Ray ray = segments[idx].ray; // Get the ray for this pixel
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
        AccumulateColor(index, segments[idx].color, accumulator);
        segments[idx].remainingBounces = 0; // Terminate the path
        segmentValidFlags[idx] = 0;
        return;
    }else {
        segmentValidFlags[idx] = 1;
    }
}

__device__ float MISWeight(float a, float b, int c){
    float t1 = powf(a, c), t2 = powf(b, c);
    return t1 / (t1 + t2);
}

__device__ void PathTracing(RenderSegment &segment, IntersectionInfo &info, RenderParam renderParam, float *accumulator, int *segmentValidFlags, int idx){
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
            // segmentValidFlags[idx] = 0;
            // segment.remainingBounces = 0; // Terminate the path if hitting light after first bounce
            // return ;
            // MIS
            int lightID = meshData[info.meshID].lightID;
            // printf("Hit light ID: %d\n", lightID);
            Light *light = lightManager -> GetLight(lightID);
            float pdfLight;
            light -> SampleSolidAnglePDF(segment.ray, info.hitPoint, info.normal, pdfLight);
            // printf("pdfLight on light hit: %f\n", pdfLight);

            // float r2 = (hitPoint - segment.ray.origin).squaredNorm();
            // // float cosTheta2 = fabs(hitNormal.dot(wo));
            // float cosTheta2 = fmaxf(hitNormal.dot(wo), 1e-6f);
            // float pdfLight2 = pdfSelectLight * 1.f / light -> area;
            // pdfLight2 = (r2 * pdfLight2) / cosTheta2;
            // printf("pdflight: %f, pdflight2: %f\n", pdfLight, pdfLight2);
            // printf("pdf select light: %f\n", pdfSelectLight);
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
            // printf("pdflight: %f\n", pdfLight);
            float cosTheta = hitNormal.dot(dirWi);
            Vec3f dirBrdf = material.Evaluate(wo, hitNormal, dirWi);
            Vec3f dirL = light -> emission.cwiseProduct(dirBrdf) * cosTheta / pdfLight;

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
    float indirCosTheta = hitNormal.dot(indirWi);
    if(!(indirWiPdf < eps || indirCosTheta < 0)){
        segment.weight = segment.weight.cwiseProduct(brdf) * indirCosTheta / indirWiPdf;
    }else{
        segment.remainingBounces = 0;
        segmentValidFlags[idx] = 0;
        return ;
    }

    // Russian roulette
    float p = sampler -> Get1D(idx);
    if(p > rr){
        segment.remainingBounces = 0; // Terminate the path
        segmentValidFlags[idx] = 0;
        return ;
    }

    segment.weight /= rr;
    segment.ray.origin = hitPoint; // Offset to avoid self-intersection
    segment.ray.direction = indirWi;

    segment.firstBounce = false; // After the first bounce
    segmentValidFlags[idx] = 1;
}

__global__ void ShadingKernel(RenderSegment *segments, RenderParam renderParam, IntersectionInfo *intersectionInfo, int numSegments, float *accumulator, int *segmentValidFlags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSegments)
        return;

    IntersectionInfo info = intersectionInfo[idx];
    if(segments[idx].remainingBounces > 0){
        PathTracing(segments[idx], info, renderParam, accumulator, segmentValidFlags, idx);
    }
    if(segmentValidFlags[idx] == 0)
        AccumulateColor(segments[idx].index, segments[idx].color, accumulator);
    segments[idx].remainingBounces--;
}

__global__ void GatherKernel(uchar4 *pixels, float *accumulator, int spp, RenderParam renderParam)
{
    int width = renderParam.width, height = renderParam.height;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return; // Ensure we don't access out of bounds
    int idx = j * width + i;
    Vec3f color = Vec3f(accumulator[3 * idx] / spp, accumulator[3 * idx + 1] / spp, accumulator[3 * idx + 2] / spp);
    j = height - j - 1; // Flip vertically for OpenGL
    int segIndex = j * width + i;

    // gamma
    pixels[segIndex] = make_uchar4(
        static_cast<unsigned char>(fminf(fmaxf(powf(color(0), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
        static_cast<unsigned char>(fminf(fmaxf(powf(color(1), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
        static_cast<unsigned char>(fminf(fmaxf(powf(color(2), 1.0f / 2.2f) * 255.0f, 0.0f), 255.0f)),
        255);
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
        while (!glfwWindowShouldClose(ui -> window))
        {
            // LOG_DEBUG("here");
            ui -> GuiBegin(sppCounter, framebufferReset);
            if(framebufferReset){
                sppCounter = 0;
                cudaMemset(accumulator, 0, sizeof(float) * 3 * renderParam.width * renderParam.height);
                framebufferReset = false;
            }
        
            if(sppCounter < renderParam.spp){
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
                    // printf("Bounce %d, Active segments: %d\n", i, numSegments);

                    // 2. Intersection
                    renderGridSize = dim3((numSegments + renderBlockSize.x - 1) / renderBlockSize.x);
                    IntersectionKernel<<<renderGridSize, renderBlockSize>>>(renderSegments, renderParam, intersections, numSegments, segmentValidFlags, accumulator);
                    cudaDeviceSynchronize();
                    CHECK_CUDA_ERROR(cudaGetLastError());

                    // stream compaction
                    numSegments = StreamCompaction(segmentValidFlags, segmentPos, numSegments);
                    // printf("After intersection stream compaction, Active segments: %d\n", numSegments);
                    if(numSegments <= 0)break;

                    // todo: sort intersections by materialID to improve memory coherence

                    // 3. Shading
                    renderGridSize = dim3((numSegments + renderBlockSize.x - 1) / renderBlockSize.x);
                    ShadingKernel<<<renderGridSize, renderBlockSize>>>(renderSegments, renderParam, intersections, numSegments, accumulator, segmentValidFlags);
                    cudaDeviceSynchronize();
                    CHECK_CUDA_ERROR(cudaGetLastError());

                    // stream compaction
                    numSegments = StreamCompaction(segmentValidFlags, segmentPos, numSegments);
                    // printf("After shading stream compaction, Active segments: %d\n", numSegments);
                }

                // 4. Gather
                gatherGridSize = dim3((width + gatherBlockSize.x - 1) / gatherBlockSize.x, (height + gatherBlockSize.y - 1) / gatherBlockSize.y);
                GatherKernel<<<gatherGridSize, gatherBlockSize>>>(pixels, accumulator, sppCounter, renderParam);
                cudaDeviceSynchronize();
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
        cudaMalloc(&accumulator, sizeof(float) * 3 * width * height);
        cudaMemset(accumulator, 0, sizeof(float) * 3 * width * height);
        cudaMalloc(&segmentValidFlags, sizeof(int) * width * height);
        cudaMalloc(&segmentPos, sizeof(int) * width * height);
        cudaMemset(segmentValidFlags, 0, sizeof(int) * height);
        cudaMemset(segmentPos, 0, sizeof(int) * width * height);
        cudaMemset(renderSegmentsBuffer, 0, sizeof(RenderSegment) * width * height);

        CHECK_CUDA_ERROR(cudaSetDevice(0)); // Set the device to use
        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&ui -> cudaPBOResource, ui -> PBO, cudaGraphicsMapFlagsWriteDiscard));
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &ui -> cudaPBOResource, 0));
        CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&pixels, &numBytes, ui -> cudaPBOResource));
        if (numBytes != width * height * sizeof(uchar4))
        {
            std::cout << "Mapped PBO size does not match expected size: " << width * height * sizeof(uchar4) << " bytes";
            return;
        }
    }

    UI *ui;
    uchar4* pixels;
    size_t numBytes;
    int sppCounter = 0;
    bool framebufferReset = false;
};