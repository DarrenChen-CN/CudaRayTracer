#pragma once
#include "global.h"
#include "light.h"

enum LightSelectStrategy{
    UNIFORM,
    POWER
};

class LightManager{
public:
    __host__ LightManager(int numLights);
    __host__ ~LightManager();

    __device__ Light* GetLight(int idx);
    __device__ Light* SampleLight(Sampler *sampler, int idx, float &pdf);

    Light* lights = nullptr;
    int numLights = 0;
    LightSelectStrategy strategy = UNIFORM;
};

void CreateLightManager(LightManager *hostLightManager, LightManager *deviceLightManager);