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
    BVHNode* left = nullptr;         // Left child
    BVHNode* right = nullptr;        // Right child
    Bounds3D bounds;                 // Bounding box of the node
    bool isLeaf;                     // True if the node is a leaf node
    int depth = 0;
    int offset = 0;  
    int numTriangles = 0;            // Number of triangles in the leaf node   
};

struct LinearBVHNode
{
    int triangleIndex = -1;   // Index of the triangle if it's a leaf node
    union {
        int numTriangles;    // Number of triangles in the leaf node
        int rightChildIndex = -1; // Index of the right child node, notice that left child is always the next node in the array
    };
    Bounds3D bounds;          // Bounding box of the node
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
    __device__ bool IsIntersect(const Ray& ray, IntersectionInfo& info) const;
    __host__ int FlattenBVHTree(BVHNode* node, int* offset);
    __host__ BVHNode* RecursiveBuild(Triangle *triangles, int start, int end, std::vector<Triangle>&orderedTriangles, int depth = 0);

    int totalNode = 0; // Total number of nodes in the BVH
    Triangle* orderedTriangles;
    LinearBVHNode* linearNodes; // Linear BVH nodes for fast traversal
    BVHNode* root = nullptr;
    int numTriangles = 0;
    int maxDepth = 0;
    BVHSplitMethod method = SAH;

    int maxTrianglesInLeaf = 4; // Maximum number of triangles in a leaf node
};

void CreateBVH(BVH *hostBVH, BVH *deviceBVH);