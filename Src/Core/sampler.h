#pragma once
#include "global.h"
#include <curand.h>
#include <curand_kernel.h>

class Sampler
{
public:
    __host__ Sampler(int seed, int numPixels);
    __host__ ~Sampler();

    __device__ float Get1D(int idx);
    __device__ Vec2f Get2D(int idx);

    int seed;
    int numPixels;
    curandState *devStates;
};

__global__ void InitKernel(curandState *state, int seed, int numPixels);
void CreateSampler(Sampler *hostSampler, Sampler *deviceSampler);