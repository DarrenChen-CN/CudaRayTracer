#pragma once
#include "global.h"
#include "triangle.h"
#include "bvh.h"
#include "cuda_runtime.h"
#include "sampler.h"
#include "camera.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <json.hpp>

using json = nlohmann::json;
// #include "light.cuh"

class Light;

class Scene
{
public:
    __host__ __device__ Scene(const std::string &sceneFilePath);
    __host__ __device__ ~Scene();

    // __host__ void LoadScene(const std::string &sceneFilePath);
    __host__ __device__ std::vector<Triangle> LoadObject(const std::string &objectFilePath, int id, Mat4f transform = Mat4f::Identity(), Mat4f transformInv = Mat4f::Identity());
    __host__ __device__ void BuildBVH();
    __host__ __device__ bool ParseSceneFile(const std::string &sceneFilePath);
    __host__ __device__ bool ParseCamera(const json& cameraJson);
    __host__ __device__ bool ParseMaterials(const json& materialsJson);
    __host__ __device__ bool ParseObjects(const json& objectsJson);
    __host__ __device__ bool ParseLights(const json& lightsJson);

    Material *hostMaterials;
    Material *materials;
    int numMaterials;
    std::unordered_map<std::string, int> materialMap;
    Triangle *triangles;
    Triangle *hostTriangles;
    int numTriangles;
    Light *lights;
    int numLights;
    BVH *bvh;
    MeshData *meshes;
    int numMeshes;
    Camera *camera;
    Sampler *sampler;

    int width, height;
    float rr;
    int spp;
    int maxBounces;
};