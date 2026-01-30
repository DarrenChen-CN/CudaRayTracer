#pragma once
#include "global.h"
#include "triangle.h"
#include <cuda_runtime.h>

enum BVHSplitMethod
{
    SAH,         // Surface Area Heuristic
    EqualCounts, // Equal counts split
    HLBVH        // Hierarchical LBVH
};


struct BVHNode
{
    BVHNode* left = nullptr;
    BVHNode* right = nullptr;
    Bounds3D bounds;
    bool isLeaf;
    int depth = 0;
    int offset = 0;  
    int numTriangles = 0;
};

struct LinearBVHNode
{
    int triangleIndex = -1;
    union {
        int numTriangles;
        int rightChildIndex = -1;
    };
    Bounds3D bounds;
};

struct SAHBucket
{
    Bounds3D bounds;
    int count = 0;
};
#define BUCKET_COUNT 12

class BVH
{
public:
    __host__ BVH();
    __host__  BVH(Triangle *triangles, int numTriangles, int maxTrianglesInLeaf = 4, BVHSplitMethod method = SAH);
    __host__ ~BVH();
    __device__ bool IsIntersect(const Ray& ray, IntersectionInfo& info, float tMin = 0.f, float tMax = 1e8) const;
    __host__ int FlattenBVHTree(BVHNode* node, int* offset);
    __host__ BVHNode* RecursiveBuild(Triangle *triangles, int start, int end, std::vector<Triangle>&orderedTriangles, int depth = 0);
    __host__ void ReleaseBVHNode(BVHNode* node);

    int totalNode = 0;
    Triangle* orderedTriangles; // device
    LinearBVHNode* linearNodes; // device
    BVHNode* root = nullptr; // host
    Triangle *hostOrderedTriangles = nullptr; // host
    LinearBVHNode* hostLinearNodes = nullptr; // host
    int numTriangles = 0;
    int maxDepth = 0;
    BVHSplitMethod method = SAH;

    int maxTrianglesInLeaf = 4;
};

void CreateBVH(BVH *hostBVH, BVH *deviceBVH);