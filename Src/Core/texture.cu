#include "texture.h"
#include "mathutil.h"

__host__ HDRTexture::HDRTexture(std::string filename){
    Load(filename);
}

__host__ HDRTexture::~HDRTexture(){
    if(cudaTextureObj){
        cudaDestroyTextureObject(cudaTextureObj);
    }
    if(cudaArrayPtr){
        cudaFreeArray(cudaArrayPtr);
    }
}

__host__ void HDRTexture::Load(std::string filename){
    width = 0;
    height = 0;

    int channels;
    float *hostData = stbi_loadf(filename.c_str(), &width, &height, &channels, 4); // cuda texture needs 4 channels
    if(!hostData){
        throw std::runtime_error("Failed to load HDR image.");
    }

    // allocate cuda array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CHECK_CUDA_ERROR(cudaMallocArray(&cudaArrayPtr, &channelDesc, width, height));

    // host to device copy
    const size_t size = width * height * 4 * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpyToArray(cudaArrayPtr, 0, 0, hostData, size, cudaMemcpyHostToDevice));

    // compute CDFs for importance sampling
    // uCDF = new float[width * height];
    // vCDF = new float[height];
    uCDF = (float*)malloc(width * height * sizeof(float));
    vCDF = (float*)malloc(height * sizeof(float));
    float *hostUCDF = new float[width * height];
    float *hostVCDF = new float[height];

    for(int y = 0; y < height; y++){
        float vSum = 0.0f;
        for(int x = 0; x < width; x++){
            int idx = y * width + x;
            float r = hostData[idx * 4 + 0];
            float g = hostData[idx * 4 + 1];
            float b = hostData[idx * 4 + 2];
            float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            // consider spherical domain
            float theta = ( (float)y + 0.5f ) / (float)height * PI;
            luminance *= sinf(theta);
            vSum += luminance;
            hostUCDF[idx] = vSum;
            
        }
        hostVCDF[y] = vSum;
    }

    // normalize uCDF
    for(int y = 0; y < height; y++){
        float rowSum = hostVCDF[y];
        for(int x = 0; x < width; x++){
            int idx = y * width + x;
            if(rowSum > 0.0f){
                hostUCDF[idx] /= rowSum;
            }else{
                hostUCDF[idx] = 0.0f;
            }
        }
    }

    // normalize vCDF
    for(int i  = 1; i < height; i++){
        hostVCDF[i] += hostVCDF[i - 1];
    }
    float totalSum = hostVCDF[height - 1];
    for(int i = 0; i < height; i++){
        if(totalSum > 0.0f){
            hostVCDF[i] /= totalSum;
        }else{
            hostVCDF[i] = 0.0f;
        }
    }

    // copy CDFs to device memory
    CHECK_CUDA_ERROR(cudaMalloc(&uCDF, width * height * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&vCDF, height * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(uCDF, hostUCDF, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(vCDF, hostVCDF, height * sizeof(float), cudaMemcpyHostToDevice));

    delete[] hostUCDF;
    delete[] hostVCDF;

    stbi_image_free(hostData);

    // create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArrayPtr;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1; // access with normalized texture coordinates

    CHECK_CUDA_ERROR(cudaCreateTextureObject(&cudaTextureObj, &resDesc, &texDesc, nullptr));
    std::cout << "Loaded HDR texture: " << filename << " (" << width << "x" << height << ")\n";
}

__device__ void HDRTexture::Sample(float &u, float &v, Sampler *sampler, int idx) const {
    // binary search vCDF
    float rv = sampler->Get1D(idx);

    int vIdx = 0;
    int left = 0, right = height - 1;
    while(left <= right){
        int mid = (left + right) / 2;
        float cdfValue;
        cdfValue = vCDF[mid];
        if(cdfValue < rv){
            left = mid + 1;
        }else{
            right = mid - 1;
        }
    }
    vIdx = left;

    // binary search uCDF
    float ru = sampler->Get1D(idx);
    int uIdx = 0;
    left = 0, right = width - 1;
    while(left <= right){
        int mid = (left + right) / 2;
        float cdfValue;
        cdfValue = uCDF[vIdx * width + mid];
        if(cdfValue < ru){
            left = mid + 1;
        }else{
            right = mid - 1;
        }
    }
    uIdx = left;

    // convert to texture coordinates
    u = ( (float)uIdx + 0.5f ) / (float)width;
    v = ( (float)vIdx + 0.5f ) / (float)height;
}

__device__ void HDRTexture::SampleSolidAnglePDF(float &u, float &v, float &pdf) const{
    u = Clamp(0.0f, 0.999999f, u); // 确保不会越界
    v = Clamp(0.0f, 0.999999f, v);
    // (u, v) -> (uIdx, vIdx)
    int uIdx = min(int(u * width), width - 1);
    int vIdx = min(int(v * height), height - 1);

    float pRow = vIdx == 0 ? vCDF[0] : vCDF[vIdx] - vCDF[vIdx - 1];
    float pCol = uIdx == 0 ? uCDF[vIdx * width + 0] : uCDF[vIdx * width + uIdx] - uCDF[vIdx * width + uIdx - 1];
    float p = pRow * pCol;
    pdf = p * width * height; // uv space PDF
}

__device__ void HDRTexture::SampleSolidAnglePDF(float &u, float &v, float &pdf) const {
    u = Clamp(0.0f, 0.999999f, u); // 确保不会越界
    v = Clamp(0.0f, 0.999999f, v);
    // (u, v) -> (uIdx, vIdx)
    int uIdx = min(int(u * width), width - 1);
    int vIdx = min(int(v * height), height - 1);

    float pRow = vIdx == 0 ? vCDF[0] : vCDF[vIdx] - vCDF[vIdx - 1];
    float pCol = uIdx == 0 ? uCDF[vIdx * width + 0] : uCDF[vIdx * width + uIdx] - uCDF[vIdx * width + uIdx - 1];
    float p = pRow * pCol;
    float theta = v * PI;
    if(sinf(theta) < 1e-6f){
        pdf = 0.0f;
        return;
    }
    pdf = p * width * height / (2.0f * PI * PI * sinf(theta)); // convert to solid angle PDF
}

void CreateHDRTexture(HDRTexture *hostTexture, HDRTexture *deviceTexture){
    CHECK_CUDA_ERROR(cudaMemcpy(deviceTexture, hostTexture, sizeof(HDRTexture), cudaMemcpyHostToDevice));
}