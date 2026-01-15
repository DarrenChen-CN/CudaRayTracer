#pragma once
#include "stb_image.h"
#include "global.h"
#include <cuda_runtime.h>

class HDRTexture{
public:
    __host__ HDRTexture(std::string filename);
    __host__ ~HDRTexture();
    __host__ void Load(std::string filename);


    int width, height;
    cudaTextureObject_t cudaTextureObj = 0;
    cudaArray* cudaArrayPtr = nullptr;
};
