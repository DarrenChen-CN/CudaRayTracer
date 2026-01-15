#include "texture.h"

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