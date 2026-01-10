#pragma once
#include "global.h"
#include "ray.h"
#include "sampler.h"
#include "mathutil.h"

enum CameraType{
    PINHOLE
};

struct CameraParam{
    // For orbit camera
    Vec3f target;
    float distance;
    float theta;
    float phi;
    float rotateSpeed;
    float zoomSpeed;
    float minDistance = 0.1f;
    float moveSpeed;

    int width, height;
    Vec3f position;
    Vec3f lookat;
    Vec3f up;
    float fovy;
    float focalLength;
    float ratio;
    Vec3f leftDownPosition;
    Vec3f u, v, w; // x、y、z
    Vec3f du, dv;

    // Rendering parameters
    int maxBounces = 5; // Max depth of path tracing
    float rr = 0.8; // Russian roulette probability
    int spp = 16; // Samples per pixel
    int sppCounter = 0;

    // default direction
    Vec3f defaultDirection = Vec3f(0, 0, -1);

};

inline CameraParam cameraParam;

class Camera{
public:
    __host__ __device__ Camera(){};
    __host__ __device__ Camera(CameraType type);
    __host__ __device__ ~Camera();
    __device__ void GeneratingRay(int i, int j, Sampler *sampler, Ray *ray, CameraParam camParam);
    __device__ void PinholeGeneratingRay(int i, int j, Sampler *sampler, Ray *ray, CameraParam camParam);

        __host__ static void ComputeCameraParam(CameraParam &camParam){
        float theta = camParam.theta;
        float phi = camParam.phi;
        theta = AngleToRadian(theta);
        phi = AngleToRadian(phi);

        float x = camParam.distance * sin(theta) * sin(phi);
        float y = camParam.distance * cos(theta);
        float z = camParam.distance * sin(theta) * cos(phi);

        camParam.position = camParam.target + Vec3f(x, y, z);
        camParam.lookat = camParam.target;
        camParam.up = Vec3f(0, 1, 0);

        // Axis calculation
        camParam.w = -(camParam.lookat - camParam.position).normalized(); // z-axis
        camParam.u = camParam.up.normalized().cross(camParam.w);          // x-axis
        camParam.v = camParam.w.cross(camParam.u);                        // y-axis

        camParam.focalLength = (camParam.lookat - camParam.position).norm();
        camParam.ratio = 1.f * camParam.width / camParam.height;
        float h_ = 2 * camParam.focalLength * tan(AngleToRadian(camParam.fovy / 2));
        float w_ = h_ * camParam.ratio;
        camParam.du = w_ * camParam.u / camParam.width;
        camParam.dv = h_ * camParam.v / camParam.height;
        camParam.leftDownPosition = camParam.lookat - camParam.width / 2 * camParam.du - camParam.height / 2 * camParam.dv;
    }

    CameraType type;
};

void CreateCamera(Camera *hostCamera, Camera *deviceCamera);