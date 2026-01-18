#pragma once
#include "stb_image.h"
#include "global.h"
#include <cuda_runtime.h>
#include "sampler.h"

class HDRTexture{
public:
    __host__ HDRTexture(std::string filename);
    __host__ ~HDRTexture();
    __host__ void Load(std::string filename);
    __device__ void Sample(float &u, float &v, Sampler *sampler, int idx) const;
    __device__ void SamplePDF(float &u, float &v, float &pdf) const;
    __device__ void SampleSolidAnglePDF(float &u, float &v, float &pdf) const;
    int width, height;
    cudaTextureObject_t cudaTextureObj = 0;
    cudaArray* cudaArrayPtr = nullptr;

    float *uCDF = nullptr; // width * height
    float *vCDF = nullptr; // height

};

void CreateHDRTexture(HDRTexture *hostTexture, HDRTexture *deviceTexture);