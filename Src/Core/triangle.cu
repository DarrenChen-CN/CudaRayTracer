#include "triangle.h"

__host__ __device__ Triangle::Triangle(Vertex& v0, Vertex& v1, Vertex& v2, int meshID)
    : v0(v0), v1(v1), v2(v2), meshID(meshID)
{
    // Compute the normal and area of the triangle
    normal = (v1.position - v0.position).cross(v2.position - v0.position);
    area = 0.5f * normal.norm();
    normal = normal.normalized();
    bounds = GetBounds();
}

__host__ __device__ Triangle::Triangle() {}


__device__ bool Triangle::IsIntersect(const Ray& ray, IntersectionInfo& info, float tMin, float tMax) const
{
    // // Reference to PBRT V3
    // // Translating ray origin to (0, 0, 0)
    // Vec3f p0 = v0.position - ray.origin;
    // Vec3f p1 = v1.position - ray.origin;
    // Vec3f p2 = v2.position - ray.origin;

    // Ray tempRay = ray; // Copy the ray to avoid modifying the original
    // tempRay.origin -= ray.origin;

    // // Find the max dimension of the ray direction
    // int zDim = MaxDimension(Abs(ray.direction));
    // int xDim = zDim + 1;
    // if (xDim == 3)
    //     xDim = 0;
    // int yDim = xDim + 1;
    // if (yDim == 3)
    //     yDim = 0;

    // // Permute the vertices and ray direction
    // Vec3f rayDirPermuted = Permute(ray.direction, xDim, yDim, zDim);
    // p0 = Permute(p0, xDim, yDim, zDim);
    // p1 = Permute(p1, xDim, yDim, zDim);
    // p2 = Permute(p2, xDim, yDim, zDim);
    // Vec3f tempRayDirPermuted = Permute(tempRay.direction, xDim, yDim, zDim);

    // // Translating ray direction to (0, 0, 1) using shear transformation
    // float sx = -rayDirPermuted(0) / rayDirPermuted(2);
    // float sy = -rayDirPermuted(1) / rayDirPermuted(2);
    // float sz = 1.0f / rayDirPermuted(2);
    // p0(0) += sx * p0(2);
    // p0(1) += sy * p0(2);
    // p1(0) += sx * p1(2);
    // p1(1) += sy * p1(2);
    // p2(0) += sx * p2(2);
    // p2(1) += sy * p2(2);

    // // Judge if the ray intersects with the triangle using cross product
    // float e0 = (p1(0) * p2(1) - p2(0) * p1(1));
    // float e1 = (p2(0) * p0(1) - p0(0) * p2(1));
    // float e2 = (p0(0) * p1(1) - p1(0) * p0(1));

    // if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
    // {
    //     // The ray is outside the triangle
    //     return false;
    // }
    // float det = e0 + e1 + e2;
    // if (det == 0)
    // {
    //     // The ray is parallel to the triangle
    //     return false;
    // }

    // // Calculate the hit time
    // p0(2) *= sz;
    // p1(2) *= sz;
    // p2(2) *= sz;
    // float tScaled = e0 * p0(2) + e1 * p1(2) + e2 * p2(2);

    // // Check if the hit time is valid
    // float hitTime = info.hitTime;
    // if (det < 0 && (tScaled >= 0 || tScaled < hitTime * det))
    // {
    //     // The ray is outside the triangle
    //     return false;
    // }
    // else if (det > 0 && (tScaled <= 0 || tScaled > hitTime * det))
    // {
    //     // The ray is outside the triangle
    //     return false;
    // }
    // float invDet = 1.0f / det;
    // info.hitTime = tScaled * invDet;
    // // If need, calculate the barycentric coordinates and hit point...
    // float b0 = e0 * invDet;
    // float b1 = e1 * invDet;
    // float b2 = e2 * invDet;
    // info.hitPoint = b0 * v0.position + b1 * v1.position + b2 * v2.position;
    // info.shadingNormal = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
    // info.normal = normal;
    // info.texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;
    // info.meshID = meshID;
    // // info.material = mesh -> GetMaterial(materialID);
    // return true; // Intersection found

    Vec3f e1 = v1.position - v0.position;
    Vec3f e2 = v2.position - v0.position;
    Vec3f s = ray.origin - v0.position;
    Vec3f s1 = ray.direction.cross(e2);
    Vec3f s2 = s.cross(e1);

    double det = e1.dot(s1);
    if (det == 0 || det < 0)
        return false;

    float delta_inv = 1.f / det;
    float t = delta_inv * s2.dot(e2);
    if (t >= info.hitTime)return false;

    float b1 = delta_inv * s1.dot(s);
    if (b1 < 0 || b1 > 1)return false;
    float b2 = delta_inv * s2.dot(ray.direction);
    if (b2 < 0 || b1 + b2 > 1)return false;

    float b0 = 1 - b1 - b2;

    if (t < tMin || t > tMax)return false;

    info.hitTime = t;
    info.hitPoint = b0 * v0.position + b1 * v1.position + b2 * v2.position;
    info.shadingNormal = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
    info.normal = normal;
    info.texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;
    info.meshID = meshID;
    info.hit = true;
    return true;
}

__device__ void Triangle::Sample(TriangleSampleInfo &info, Sampler *sampler, int idx) const{
    Vec2f sample2D = sampler -> Get2D(idx);
    float u = sample2D(0), v = sample2D(1);
    if(u + v > 1){
        u = 1- u, v = 1 - v;
    }
    info.position = (1 - u - v) * v0.position + u * v1.position + v * v2.position;
    info.normal = normal;
}

__host__ __device__ Bounds3D Triangle::GetBounds() const {
    Bounds3D bounds(v0.position, v1.position, v2.position);
    return bounds;
}

void CreateTriangle(Triangle *hostTriangle, Triangle *deviceTriangle){
    // cudaMalloc(&deviceTriangle, sizeof(Triangle));
    cudaMemcpy(deviceTriangle, hostTriangle, sizeof(Triangle), cudaMemcpyHostToDevice);
}

void CreateMeshData(MeshData *hostMesh, MeshData *deviceMesh){
    // cudaMalloc(&deviceMesh, sizeof(MeshData));
    cudaMemcpy(deviceMesh, hostMesh, sizeof(MeshData), cudaMemcpyHostToDevice);
}