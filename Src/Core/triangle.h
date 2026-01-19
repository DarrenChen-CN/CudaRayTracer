#pragma once
#include <cuda_runtime.h>
#include "global.h"
#include "material.h"
#include "bounds.h"

struct IntersectionInfo
{
    Vec3f hitPoint;
    Vec3f normal;        // geometry normal
    Vec3f shadingNormal; // shading normal
    Vec2f texCoord;
    int meshID = -1;
    float hitTime = FLOATMAX;
    bool hit = false;
};

struct RenderSegment
{
    Ray ray;
    Vec3f color;
    Vec3f weight;
    int index;
    int remainingBounces;
    bool firstBounce = true;
    float pdfBrdf = 0.f; // For MIS
};

struct TriangleSampleInfo {
    Vec3f position;
    Vec3f normal;
};

struct Vertex
{
    Vec3f position;
    Vec3f normal;
    Vec2f texCoord;
};

class Triangle
{
public:
    int meshID;
    float area = 0.f;
    Vertex v0, v1, v2;
    Vec3f normal; // Geometry normal
    Bounds3D bounds;
    
    __host__ __device__ Triangle();
    __host__ __device__ Triangle(Vertex& v0, Vertex& v1, Vertex& v2, int meshID);

    __device__ bool IsIntersect(const Ray& ray, IntersectionInfo& info, float tMin = 0.f, float tMax = 1e8) const;
    __device__ void Sample(TriangleSampleInfo &info, Sampler *sampler, int idx) const;
    __host__ __device__ Bounds3D GetBounds() const;
};

void CreateTriangle(Triangle *hostTriangle, Triangle *deviceTriangle);

struct MeshData{
    std::string name;
    int meshID;
    int materialID;
    int startTriangleID; // in the global triangle list
    int numTriangles;
    float area = 0.f;
    Mat4f transform; // Transformation matrix for the mesh
    Mat4f transformInv;
    int lightID = -1;
};

void CreateMeshData(MeshData *hostMesh, MeshData *deviceMesh);
