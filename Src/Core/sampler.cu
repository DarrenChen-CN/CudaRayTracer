#include "sampler.h"

__host__ Sampler::Sampler(int seed, int numPixels) : seed(seed), numPixels(numPixels) {
    cudaMalloc(&devStates, numPixels * sizeof(curandState));
    InitKernel<<<(numPixels + 255) / 256, 256>>>(devStates, seed, numPixels);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__host__ __device__ Sampler::~Sampler() {
    // if (devStates)
    // {
    //     cudaFree(devStates);
    //     devStates = nullptr;
    // }
}

__device__ float Sampler::Get1D(int idx) {
    float u = curand_uniform(&devStates[idx]);
    return u;
}

__device__ Vec2f Sampler::Get2D(int idx) {
    // 为什么直接返回Vec2f(cuda_uniform(&devStates[idx]), curand_uniform(&devStates[idx])) 会报错？
    float u = curand_uniform(&devStates[idx]);
    float v = curand_uniform(&devStates[idx]);
    // printf("RandomSampler Get 2D: idx = %d, u = %f, v = %f\n", idx, u, v);
    return Vec2f(u, v);
}

__global__ void InitKernel(curandState *state, int seed, int numPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels)
    {
        curand_init(seed, idx, 0, &state[idx]);
        
    }
}

void CreateSampler(Sampler *hostSampler, Sampler *deviceSampler){
    cudaMemcpy(deviceSampler, hostSampler, sizeof(Sampler), cudaMemcpyHostToDevice);
}