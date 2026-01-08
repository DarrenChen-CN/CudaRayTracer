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
    orderedTriangles = new Triangle[numTriangles];
    this -> numTriangles = numTriangles;
    this -> maxTrianglesInLeaf = maxTrianglesInLeaf;
    this -> method = method;
    std::vector<Triangle> tempOrderedTriangles;
    this -> root = RecursiveBuild(triangles, 0, numTriangles, tempOrderedTriangles);
    printf("Total BVH nodes created: %d\n", totalNode);
    printf("orderedTriangles size: %zu\n", tempOrderedTriangles.size());
    printf("max depth of BVH tree: %d\n", maxDepth);
    memcpy(orderedTriangles, tempOrderedTriangles.data(), sizeof(Triangle) * numTriangles);

    // Flatten the BVH tree
    std::cout << "Flattening BVH tree with total nodes: " << totalNode << " ..." << std::endl;
    linearNodes = new LinearBVHNode[totalNode];
    int offset = 0;
    FlattenBVHTree(this -> root, &offset);
}

__host__  __device__ BVHNode* BVH::RecursiveBuild(Triangle *triangles, int start, int end, std::vector<Triangle>& orderedTriangles, int depth)
{
    
    // std::cout << "Building BVH node at depth " << depth << " with triangles from " << start << " to " << end << std::endl;
    // Create a node
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

    // Compute Bounding box
    Bounds3D bounds;
    for (int i = start; i < end; i++)
    {
        bounds = bounds.Expand(triangles[i].bounds);
    }

    node->bounds = bounds; // Set the bounding box of the node

    if (end - start <= maxTrianglesInLeaf)
    {
        // printf("Creating leaf node at depth %d with triangle index %d\n", depth, start);
        node->isLeaf = true;
        node->left = nullptr;
        node->right = nullptr;
        node->offset = orderedTriangles.size();       // Set the offset for the triangle in the ordered list
        node->numTriangles = end - start;                        // Set the number of triangles in the leaf node
        for (int i = start; i < end; i++) {
            orderedTriangles.push_back(triangles[i]); // Add the triangle to the ordered list
        }
        return node;
    }

    // Not leaf node
    node->isLeaf = false;

    // Find split axis
    int splitAxis = bounds.MaxExtent();
    // printf("Splitting node at depth %d along axis %d with %d triangles\n", depth, splitAxis, end - start);

    // Split
    if (method == EqualCounts)
    {
        // Using std::nth_element to find the median
        int mid = (start + end) / 2;
        std::nth_element(&triangles[start], &triangles[mid], &triangles[end - 1] + 1, [splitAxis](const Triangle a, const Triangle b)
            { return a.GetBounds().Center()(splitAxis) < b.GetBounds().Center()(splitAxis); });
        // Build left and right children
        node->left = RecursiveBuild(triangles, start, mid, orderedTriangles, depth + 1);
        node->right = RecursiveBuild(triangles, mid, end, orderedTriangles, depth + 1);
    }
    else if (method == SAH)
    {
        // Surface Area Heuristic (SAH) split
        // todo...
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
            // printf("Triangle %d center: (%f, %f, %f), offset: %f, bucket: %d\n", i, center(0), center(1), center(2), bounds.Offset(center)(splitAxis), b);
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
            // Compute the cost of splitting at this bucket
            if(costs[i] < minCost){
                minCost = costs[i];
                bestSplit = i;
            }
        }

        float triangleCost = end - start;
        if(minCost > triangleCost && (end - start) <= maxTrianglesInLeaf){
            // Create leaf node
            // printf("Creating leaf node at depth %d with triangle index %d\n", depth, start);
            node->isLeaf = true;
            node->left = nullptr;
            node->right = nullptr;
            node->offset = orderedTriangles.size();       // Set the offset for the triangle in the ordered list
            node->numTriangles = end - start;                        // Set the number of triangles in the leaf node
            for (int i = start; i < end; i++) {
                orderedTriangles.push_back(triangles[i]);
                    // Add the triangle to the ordered list
            }
            return node;
        }

        // Partition the triangles based on the best split
        // printf("Best split at bucket %d with cost %f\n", bestSplit, minCost);
        auto midPtr = std::partition(&triangles[start], &triangles[end - 1] + 1, [=](const Triangle& tri) mutable
            {
                Vec3f center = tri.GetBounds().Center();
                int b = BUCKET_COUNT * centroidBounds.Offset(center)[splitAxis];
                if (b == BUCKET_COUNT) b = BUCKET_COUNT - 1;
                return b <= bestSplit;
            });
        int mid = midPtr - &triangles[0];
        // printf("mid: %d\n", mid);
        node->left = RecursiveBuild(triangles, start, mid, orderedTriangles, depth + 1);
        node->right = RecursiveBuild(triangles, mid, end, orderedTriangles, depth + 1);
    }
    else if (method == HLBVH){
        // Hierarchical LBVH split
        // todo...
    }
    return node;
}

int BVH::FlattenBVHTree(BVHNode* node, int* offset)
{
    // printf("Flattening node at depth %d, current offset: %d\n", node->depth, *offset);
    // Flatten the BVH tree into a linear array
    if (!node)
        return 0;

    LinearBVHNode& linearNode = linearNodes[*offset];
    linearNode.bounds = node->bounds;
    // if(node -> depth < 8)
    //     printf("flattening node at depth %d, current offset: %d\n", node->depth, *offset);
    int currentIndex = (*offset)++;
    // linearNodes.push_back(linearNode);
    if (node->isLeaf)
    {
        // printf("Leaf node created at index %d, triangle index: %d\n", currentIndex, node->offset);
        linearNode.triangleIndex = node->offset; // Set the triangle index for leaf nodes
        linearNode.numTriangles = node->numTriangles;
    }
    else
    {
        // left child is always the next node in the array
        FlattenBVHTree(node->left, offset);
        linearNode.rightChildIndex = FlattenBVHTree(node->right, offset);
    }
    return currentIndex;
}

__device__ bool BVH::IsIntersect(const Ray& ray, IntersectionInfo& info) const
{
    int cnt = 0;
    bool hit = false;
    Vec3f invDir(1.f / ray.direction(0), 1.f / ray.direction(1), 1.f / ray.direction(2));
    int dirIsNeg[3] = { invDir(0) < 0, invDir(1) < 0, invDir(2) < 0 };

    int toVisitOffset = 0, currentOffset = 0;
    int nodeToVisit[64]; // Stack to hold nodes to visit

    // #pragma unroll
    while (true){
        cnt++;
        LinearBVHNode node = linearNodes[currentOffset];

        // Check if the ray intersects with the bounding box of the node
        if (node.bounds.IsIntersect(ray, invDir, dirIsNeg)){
            // printf("Ray intersects with node %d\n", currentOffset);
            if (node.triangleIndex != -1)
            {
                for(int i = 0; i < linearNodes[currentOffset].numTriangles; i++){
                    if (orderedTriangles[node.triangleIndex + i].IsIntersect(ray, info))
                    {
                        hit = true;
                    }
                }
                if (toVisitOffset == 0)
                    break;                                    // Stack is empty, no more nodes to visit
                currentOffset = nodeToVisit[--toVisitOffset]; // Pop the last node to visit
            }
            else
            {
                nodeToVisit[toVisitOffset++] = node.rightChildIndex; // Push right child node to visit stack
                currentOffset = currentOffset + 1; // Move to the left node in the linear array
            }
        }
        else
        {
            if (toVisitOffset == 0)
                break;                                    // No more nodes to visit
            currentOffset = nodeToVisit[--toVisitOffset]; // Pop the last node to visit
        }
    }
    // printf("BVH traversal count: %d\n, hit = %d\n", cnt, hit);
    return hit;
}

// __device__ bool BVH::IsIntersect(const Ray& ray, IntersectionInfo& info) const
// {
//     int cnt = 0;
//     bool hit = false;
//     Vec3f invDir(1.f / ray.direction(0), 1.f / ray.direction(1), 1.f / ray.direction(2));
//     int dirIsNeg[3] = { invDir(0) < 0, invDir(1) < 0, invDir(2) < 0 };

//     int toVisitOffset = 0, currentOffset = 0;
//     int nodeToVisit[64]; // Stack to hold nodes to visit

//     while (true){
//         cnt++;
//         LinearBVHNode node = linearNodes[currentOffset];

//         // printf("Ray intersects with node %d\n", currentOffset);
//         if (node.triangleIndex != -1){
//             for(int i = 0; i < linearNodes[currentOffset].numTriangles; i++){
//                 if (orderedTriangles[node.triangleIndex + i].IsIntersect(ray, info)){
//                     hit = true;
//                 }
//             }
//         }
//         else
//         {
//             // Find nearest child first
//             int firstChild, secondChild;
//             firstChild = node.rightChildIndex;
//             secondChild = currentOffset + 1;
//             float tFirst, tSecond;
//             bool hitFirst = linearNodes[firstChild].bounds.IsIntersect(ray, invDir, dirIsNeg, tFirst);
//             bool hitSecond = linearNodes[secondChild].bounds.IsIntersect(ray, invDir, dirIsNeg, tSecond);
//             if(hitFirst && hitSecond){
//                 if(tFirst < tSecond){
//                     nodeToVisit[toVisitOffset++] = secondChild;
//                     nodeToVisit[toVisitOffset++] = firstChild; // Push right child node to visit stack
//                 }else{
//                     nodeToVisit[toVisitOffset++] = firstChild; // Push right child node to visit stack
//                     nodeToVisit[toVisitOffset++] = secondChild; // Push right child node to visit stack
//                 }
//             }else if(hitFirst){
//                 nodeToVisit[toVisitOffset++] = firstChild; // Push right child node to visit stack
//             }else if(hitSecond){
//                 nodeToVisit[toVisitOffset++] = secondChild; // Push right child node to visit stack
//             }
//         }

//         if (toVisitOffset == 0)
//             break;                                    // No more nodes to visit
//         currentOffset = nodeToVisit[--toVisitOffset]; // Pop the last node to visit   
//     }

//     return hit;
// }

void CreateBVH(BVH *hostBVH, BVH *deviceBVH){
    // cudaMalloc(&deviceBVH, sizeof(BVH));
    Triangle *orderedTriangles;
    cudaMalloc(&orderedTriangles, sizeof(Triangle) * hostBVH->numTriangles);
    cudaMemcpy(orderedTriangles, hostBVH->orderedTriangles, sizeof(Triangle) * hostBVH->numTriangles, cudaMemcpyHostToDevice);
    LinearBVHNode *linearNodes;
    cudaMalloc(&linearNodes, sizeof(LinearBVHNode) * hostBVH->totalNode);
    cudaMemcpy(linearNodes, hostBVH->linearNodes, sizeof(LinearBVHNode) * hostBVH->totalNode, cudaMemcpyHostToDevice);
    hostBVH->orderedTriangles = orderedTriangles;
    hostBVH->linearNodes = linearNodes;
    cudaMemcpy(deviceBVH, hostBVH, sizeof(BVH), cudaMemcpyHostToDevice);
}


