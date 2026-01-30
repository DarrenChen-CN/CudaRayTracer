#include "mathutil.h"
#include "camera.h"

__host__ __device__ Camera::Camera(CameraType type): type(type)
{

}

__host__ __device__ Camera::~Camera(){

}

__device__ void Camera::GeneratingRay(int i, int j, Sampler *sampler, Ray *ray, CameraParam camParam)
{
    switch (type)
    {
        case PINHOLE:
            PinholeGeneratingRay(i, j, sampler, ray, camParam);
            break;
        default:
            PinholeGeneratingRay(i, j, sampler, ray, camParam);
            break;
    }
}


__device__ void Camera::PinholeGeneratingRay(int i, int j, Sampler *sampler, Ray *ray, CameraParam camParam)
{
    int idx = j * camParam.width + i;
    Vec2f pixelOffset = sampler->Get2D(idx);
    // pixelOffset = Vec2f(0.5f, 0.5f);
    Vec3f pixelPosition = camParam.leftDownPosition + ((i + pixelOffset(0))) * camParam.du + (j + pixelOffset(1)) * camParam.dv;
    Vec3f direction = (pixelPosition - camParam.position).normalized();
    ray->origin = camParam.position;
    ray->direction = direction;
}

void CreateCamera(Camera *hostCamera, Camera *deviceCamera){
    // cudaMalloc(&deviceCamera, sizeof(Camera));
    cudaMemcpy(deviceCamera, hostCamera, sizeof(Camera), cudaMemcpyHostToDevice);
}
