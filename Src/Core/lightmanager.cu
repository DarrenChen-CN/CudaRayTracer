#include "lightmanager.h"

__host__ LightManager::LightManager(int numLights):numLights(numLights){
    cudaMalloc((void**)&lights, sizeof(Light) * numLights);
}

__host__ LightManager::~LightManager(){
    if(lights){
        cudaFree(lights);
    }
}

__device__ Light* LightManager::GetLight(int idx){
    // printf("GetLight idx: %d, numLights: %d\n", idx, numLights);
    if(idx < 0 || idx >= numLights) return nullptr;
    return &lights[idx];
}

__device__ Light* LightManager::SampleLight(Sampler *sampler, int idx, float &pdf){
    if(numLights == 0) return nullptr;
    int lightID;
    if(strategy == UNIFORM){
        lightID = static_cast<int>(sampler->Get1D(idx) * numLights);
        // printf("sampled lightID: %d, light area: %f\n", lightID, lights[lightID].area);
        if(lightID == numLights) lightID = numLights - 1; // Clamp
        pdf = 1.0f / numLights;
    }
    // printf("Sampled lightID: %d\n", lightID);
    return GetLight(lightID);
}

void CreateLightManager(LightManager *hostLightManager, LightManager *deviceLightManager){
    cudaMemcpy(deviceLightManager, hostLightManager, sizeof(LightManager), cudaMemcpyHostToDevice);
 }

