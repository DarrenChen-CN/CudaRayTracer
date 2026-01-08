#pragma once
#include "global.h"
#include "ray.h"
#include "sampler.h"

enum CameraType{
    PINHOLE
};

class Camera{
public:
    __host__ __device__ Camera(){};
    __host__ __device__ Camera(Vec3f position, Vec3f lookat, Vec3f up, float fovy, int width, int height, CameraType type, int maxBounces = 5, float rr = 0.8, int spp = 16);
    __host__ __device__ ~Camera();
    __device__ void GeneratingRay(int i, int j, Sampler *sampler, Ray *ray);
    __device__ void GeneratingRay(int i, int j, Sampler *sampler, Vec3f *origin, Vec3f *direction);
    __device__ void PinholeGeneratingRay(int i, int j, Sampler *sampler, Ray *ray);
    __device__ void PinholeGeneratingRay(int i, int j, Sampler *sampler, Vec3f *origin, Vec3f *direction);

    int width, height;
    Vec3f position;
    Vec3f lookat;
    Vec3f up;
    float fovy;
    float focalLength;

    // Camera parameters
    float ratio;
    Vec3f leftDownPosition;
    Vec3f u, v, w; // x、y、z
    Vec3f du, dv;

    // Rendering parameters
    int maxBounces = 5; // Max depth of path tracing
    float rr = 0.8; // Russian roulette probability
    int spp = 16; // Samples per pixel

    CameraType type;
};

void CreateCamera(Camera *hostCamera, Camera *deviceCamera);