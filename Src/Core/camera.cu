#include "mathutil.h"
#include "camera.h"

__host__ __device__ Camera::Camera(Vec3f position, Vec3f lookat, Vec3f up, float fovy, int width, int height, CameraType type, int maxBounces, float rr, int spp) : position(position), lookat(lookat), up(up), fovy(fovy), width(width), height(height), type(type), maxBounces(maxBounces), rr(rr), spp(spp)
{
    // printf("Initializing camera...\n");
    // Compute camera parameters
    if (width <= 0 || height <= 0)
    {
        printf("Camera width and height must be positive integers.\n");
        return;
    }

    // Axis calculation
    w = -(lookat - position).normalized(); // z-axis
    u = up.normalized().cross(w);          // x-axis
    v = w.cross(u);                        // y-axis

    focalLength = (lookat - position).norm();
    ratio = 1.f * width / height;
    float h_ = 2 * focalLength * tan(AngleToRadian(fovy / 2));
    float w_ = h_ * ratio;
    du = w_ * u / width;
    dv = h_ * v / height;

    // printf("Camera parameters:\n");
    // printf("Position: (%f, %f, %f)\n", position(0), position(1), position(2));
    // printf("LookAt: (%f, %f, %f)\n", lookat(0), lookat(1), lookat(2));
    // printf("Up: (%f, %f, %f)\n", up(0), up(1), up(2));
    // printf("FOVY: %f\n", fovy); 
    // printf("du: (%f, %f, %f)\n", du(0), du(1), du(2));
    // printf("dv: (%f, %f, %f)\n", dv(0), dv(1), dv(2));

    leftDownPosition = lookat - width / 2 * du - height / 2 * dv;
}

__host__ __device__ Camera::~Camera(){

}

__device__ void Camera::GeneratingRay(int i, int j, Sampler *sampler, Ray *ray)
{
    switch (type)
    {
        case PINHOLE:
            PinholeGeneratingRay(i, j, sampler, ray);
            break;
        default:
            PinholeGeneratingRay(i, j, sampler, ray);
            break;
    }
}

__device__ void Camera::GeneratingRay(int i, int j, Sampler *sampler, Vec3f *origin, Vec3f *direction)
{
    switch (type)
    {
        case PINHOLE:
            PinholeGeneratingRay(i, j, sampler, origin, direction);
            break;
        default:
            PinholeGeneratingRay(i, j, sampler, origin, direction);
            break;
    }
}


__device__ void Camera::PinholeGeneratingRay(int i, int j, Sampler *sampler, Ray *ray)
{
    int idx = j * width + i;
    Vec2f pixelOffset = sampler->Get2D(idx);
    Vec3f pixelPosition = leftDownPosition + ((i + pixelOffset(0))) * du + (j + pixelOffset(1)) * dv;
    Vec3f direction = (pixelPosition - position).normalized();
    ray->origin = position;
    ray->direction = direction;
    // if(i > 200 && i < 210 && j > 130 && j < 140)
    //     printf("Generating Ray at pixel (%d, %d): origin = (%f, %f, %f), direction = (%f, %f, %f), pixelOffset = (%f, %f)\n", i, j, ray->origin(0), ray->origin(1), ray->origin(2), ray->direction(0), ray->direction(1), ray->direction(2), pixelOffset(0), pixelOffset(1));
}

__device__ void Camera::PinholeGeneratingRay(int i, int j, Sampler *sampler, Vec3f *origin, Vec3f *direction)
{
    int idx = j * width + i;
    Vec2f pixelOffset = sampler->Get2D(idx);
    Vec3f pixelPosition = leftDownPosition + ((i + pixelOffset(0))) * du + (j + pixelOffset(1)) * dv;
    *direction = (pixelPosition - position).normalized();
    *origin = position;
}

void CreateCamera(Camera *hostCamera, Camera *deviceCamera){
    // cudaMalloc(&deviceCamera, sizeof(Camera));
    cudaMemcpy(deviceCamera, hostCamera, sizeof(Camera), cudaMemcpyHostToDevice);
}
