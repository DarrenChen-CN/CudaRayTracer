#include "bvh.h"
#include "triangle.h"
#include <iostream>
#include <queue>
#include "bounds.h"

__host__ __device__ BVH::BVH(){}

__host__  BVH::BVH(Triangle *triangles, int numTriangles, int maxTrianglesInLeaf, BVHSplitMethod method){
    // Build the BVH
    if(numTriangles <= 0){
        std::cout << "No triangles to build BVH." << std::endl;
        return;
    }
    std::cout << "Building BVH with " << numTriangles << " triangles..." << std::endl;
    hostOrderedTriangles = new Triangle[numTriangles];
    this -> numTriangles = numTriangles;
    this -> maxTrianglesInLeaf = maxTrianglesInLeaf;
    this -> method = method;
    std::vector<Triangle> tempOrderedTriangles;
    this -> root = RecursiveBuild(triangles, 0, numTriangles, tempOrderedTriangles);
    printf("Total BVH nodes created: %d\n", totalNode);
    printf("orderedTriangles size: %zu\n", tempOrderedTriangles.size());
    printf("max depth of BVH tree: %d\n", maxDepth);
    memcpy(hostOrderedTriangles, tempOrderedTriangles.data(), sizeof(Triangle) * numTriangles);

    // Flatten the BVH tree
    std::cout << "Flattening BVH tree with total nodes: " << totalNode << " ..." << std::endl;
    hostLinearNodes = new LinearBVHNode[totalNode];
    int offset = 0;
    FlattenBVHTree(this -> root, &offset);
}

__host__ BVH::~BVH(){
    // std::cout << "Destroying BVH..." << std::endl;
    ReleaseBVHNode(root);
    delete[] hostOrderedTriangles;
    delete[] hostLinearNodes;
    cudaFree(orderedTriangles);
    cudaFree(linearNodes);
    std::cout << "BVH destroyed." << std::endl;
}

__host__  __device__ BVHNode* BVH::RecursiveBuild(Triangle *triangles, int start, int end, std::vector<Triangle>& orderedTriangles, int depth)
{
    
    BVHNode *node = new BVHNode();
    totalNode++;
    if (!node)
    {
        printf("Failed to allocate memory for BVHNode.\n");
        return nullptr;
    }
    node->depth = depth;
    if(depth > maxDepth)maxDepth = depth;

    if (start >= end)
    {
        printf("Warning: start index >= end index in BVH build, start: %d, end: %d\n", start, end);
        node -> isLeaf = true;
        return node;
        
    }

    Bounds3D bounds;
    for (int i = start; i < end; i++)
    {
        bounds = bounds.Expand(triangles[i].bounds);
    }

    node->bounds = bounds;

    if (end - start <= maxTrianglesInLeaf)
    {
        node->isLeaf = true;
        node->left = nullptr;
        node->right = nullptr;
        node->offset = orderedTriangles.size();
        node->numTriangles = end - start;
        for (int i = start; i < end; i++) {
            orderedTriangles.push_back(triangles[i]);
        }
        return node;
    }

    node->isLeaf = false;
    int splitAxis = bounds.MaxExtent();

    // Split
    if (method == EqualCounts)
    {
        int mid = (start + end) / 2;
        std::nth_element(&triangles[start], &triangles[mid], &triangles[end - 1] + 1, [splitAxis](const Triangle a, const Triangle b)
            { return a.GetBounds().Center()(splitAxis) < b.GetBounds().Center()(splitAxis); });
        node->left = RecursiveBuild(triangles, start, mid, orderedTriangles, depth + 1);
        node->right = RecursiveBuild(triangles, mid, end, orderedTriangles, depth + 1);
    }
    else if (method == SAH)
    {
        // Surface Area Heuristic (SAH) split
        Bounds3D centroidBounds;
        for (int i = start; i < end; i++){
            Vec3f center = triangles[i].bounds.Center();
            centroidBounds = centroidBounds.Expand(center);
        }

        splitAxis = centroidBounds.MaxExtent();

        std::vector<SAHBucket> buckets(BUCKET_COUNT);
        for (int i = start; i < end; i++){
            Vec3f center = triangles[i].bounds.Center();
            int b = BUCKET_COUNT * centroidBounds.Offset(center)(splitAxis);
            if (b == BUCKET_COUNT) b = BUCKET_COUNT - 1;
            buckets[b].count++;
            buckets[b].bounds = buckets[b].bounds.Expand(triangles[i].bounds);
        }


        int nSplit = BUCKET_COUNT - 1;
        std::vector<float> costs(nSplit, 0.0f);

        for(int i = 0; i < nSplit; i ++){
            Bounds3D b0, b1;
            int count0 = 0, count1 = 0;
            for(int j = 0; j <= i; j ++){
                b0 = buckets[j].bounds.Expand(b0);
                count0 += buckets[j].count;
            }
            for(int j = i + 1; j < nSplit; j ++){
                b1 = buckets[j].bounds.Expand(b1);
                count1 += buckets[j].count;
            }
            costs[i] = .125f + (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) / bounds.SurfaceArea();
        }

        // Find the best split bucket
        float minCost = FLOATMAX;
        int bestSplit = -1;
        for (int i = 1; i < nSplit; i++){
            if(costs[i] < minCost){
                minCost = costs[i];
                bestSplit = i;
            }
        }

        float triangleCost = end - start;
        if(minCost > triangleCost && (end - start) <= maxTrianglesInLeaf){
            node->isLeaf = true;
            node->left = nullptr;
            node->right = nullptr;
            node->offset = orderedTriangles.size();  
            node->numTriangles = end - start;      
            for (int i = start; i < end; i++) {
                orderedTriangles.push_back(triangles[i]);
            }
            return node;
        }

        auto midPtr = std::partition(&triangles[start], &triangles[end - 1] + 1, [=](const Triangle& tri) mutable
            {
                Vec3f center = tri.GetBounds().Center();
                int b = BUCKET_COUNT * centroidBounds.Offset(center)[splitAxis];
                if (b == BUCKET_COUNT) b = BUCKET_COUNT - 1;
                return b <= bestSplit;
            });
        int mid = midPtr - &triangles[0];
        node->left = RecursiveBuild(triangles, start, mid, orderedTriangles, depth + 1);
        node->right = RecursiveBuild(triangles, mid, end, orderedTriangles, depth + 1);
    }
    else if (method == HLBVH){
        // Hierarchical LBVH split
        // todo...
    }
    return node;
}

__host__ void BVH::ReleaseBVHNode(BVHNode* node)
{
    if (!node)
        return;
    ReleaseBVHNode(node->left);
    ReleaseBVHNode(node->right);
    delete node;
}

int BVH::FlattenBVHTree(BVHNode* node, int* offset)
{
    // Flatten the BVH tree into a linear array
    if (!node)
        return 0;

    LinearBVHNode& linearNode = hostLinearNodes[*offset];
    linearNode.bounds = node->bounds;
    int currentIndex = (*offset)++;
    if (node->isLeaf)
    {
        linearNode.triangleIndex = node->offset;
        linearNode.numTriangles = node->numTriangles;
    }
    else
    {
        FlattenBVHTree(node->left, offset);
        linearNode.rightChildIndex = FlattenBVHTree(node->right, offset);
    }
    return currentIndex;
}

// __device__ bool BVH::IsIntersect(const Ray& ray, IntersectionInfo& info) const
// {
//     int cnt = 0;
//     bool hit = false;
//     Vec3f invDir(1.f / ray.direction(0), 1.f / ray.direction(1), 1.f / ray.direction(2));
//     int dirIsNeg[3] = { invDir(0) < 0, invDir(1) < 0, invDir(2) < 0 };

//     int toVisitOffset = 0, currentOffset = 0;
//     int nodeToVisit[64];

//     while (true){
//         cnt++;
//         LinearBVHNode node = linearNodes[currentOffset];

//         if (node.bounds.IsIntersect(ray, invDir, dirIsNeg)){
//             // printf("Ray intersects with node %d\n", currentOffset);
//             if (node.triangleIndex != -1)
//             {
//                 for(int i = 0; i < linearNodes[currentOffset].numTriangles; i++){
//                     if (orderedTriangles[node.triangleIndex + i].IsIntersect(ray, info))
//                     {
//                         hit = true;
//                     }
//                 }
//                 if (toVisitOffset == 0)
//                     break;
//                 currentOffset = nodeToVisit[--toVisitOffset];
//             }
//             else
//             {
//                 nodeToVisit[toVisitOffset++] = node.rightChildIndex;
//                 currentOffset = currentOffset + 1;
//             }
//         }
//         else
//         {
//             if (toVisitOffset == 0)
//                 break;
//             currentOffset = nodeToVisit[--toVisitOffset];
//         }
//     }
//     // printf("BVH traversal count: %d\n, hit = %d\n", cnt, hit);
//     return hit;
// }

__device__ bool BVH::IsIntersect(const Ray& ray, IntersectionInfo& info, float tMin, float tMax) const
{
    int cnt = 0;
    bool hit = false;
    Vec3f invDir(1.f / ray.direction(0), 1.f / ray.direction(1), 1.f / ray.direction(2));
    int dirIsNeg[3] = { invDir(0) < 0, invDir(1) < 0, invDir(2) < 0 };

    int toVisitOffset = 0, currentOffset = 0;
    int nodeToVisit[64];

    info.hitTime = tMax;

    while (true){
        cnt++;
        LinearBVHNode node = linearNodes[currentOffset];

        if (node.triangleIndex != -1){
            for(int i = 0; i < linearNodes[currentOffset].numTriangles; i++){
                if (orderedTriangles[node.triangleIndex + i].IsIntersect(ray, info)){
                    hit = true;
                }
            }
        }
        else
        {
            // Find nearest child first
            int firstChild, secondChild;
            firstChild = node.rightChildIndex;
            secondChild = currentOffset + 1;
            float tFirst, tSecond;
            bool hitFirst = linearNodes[firstChild].bounds.IsIntersect(ray, invDir, dirIsNeg, tFirst, tMin, info.hitTime);
            bool hitSecond = linearNodes[secondChild].bounds.IsIntersect(ray, invDir, dirIsNeg, tSecond, tMin, info.hitTime);
            if(hitFirst && hitSecond){
                if(tFirst < tSecond){
                    nodeToVisit[toVisitOffset++] = secondChild;
                    nodeToVisit[toVisitOffset++] = firstChild;
                }else{
                    nodeToVisit[toVisitOffset++] = firstChild;
                    nodeToVisit[toVisitOffset++] = secondChild;
                }
            }else if(hitFirst){
                nodeToVisit[toVisitOffset++] = firstChild;
            }else if(hitSecond){
                nodeToVisit[toVisitOffset++] = secondChild;
            }
        }

        if (toVisitOffset == 0)
            break;
        currentOffset = nodeToVisit[--toVisitOffset];
    }

    return hit;
}

void CreateBVH(BVH *hostBVH, BVH *deviceBVH){
    cudaMalloc(&hostBVH->orderedTriangles, sizeof(Triangle) * hostBVH->numTriangles);
    cudaMemcpy(hostBVH->orderedTriangles, hostBVH->hostOrderedTriangles, sizeof(Triangle) * hostBVH->numTriangles, cudaMemcpyHostToDevice);
    cudaMalloc(&hostBVH->linearNodes, sizeof(LinearBVHNode) * hostBVH->totalNode);
    cudaMemcpy(hostBVH->linearNodes, hostBVH->hostLinearNodes, sizeof(LinearBVHNode) * hostBVH->totalNode, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBVH, hostBVH, sizeof(BVH), cudaMemcpyHostToDevice);
}


