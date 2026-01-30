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
#include "texture.h"
#include "lightmanager.h"

using json = nlohmann::json;
// #include "light.cuh"

class Light;

struct RenderParam{
    int width, height;
    // Device pointers
    Triangle *triangles;
    MeshData *meshData;
    Material *materials;
    LightManager *lightManager;
    BVH *bvh;
    Camera *camera;
    Sampler *sampler;
    Bounds3D sceneBounds;
    float rr;
    int spp;
    int maxBounces;
    int renderTargetMode = 0; // 0: color, 1: depth, 2: normal, 3: id
    bool denoise = false;
    int currentRenderBufferIndex = 0;
    int currentGBufferIndex = 0;

    // host pointers
    Triangle *hostTriangles;
    MeshData *hostMeshes;
    Material *hostMaterials;
    int numMeshes;
};

inline RenderParam renderParam;

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
    __host__ __device__ bool ParseTextures(const json& texturesJson);
    __host__ void InitCameraParam();
    __host__ void SetRenderParam();

    Material *hostMaterials;
    Material *materials;
    int numMaterials;
    std::unordered_map<std::string, int> materialMap;
    Triangle *triangles;
    Triangle *hostTriangles;
    int numTriangles;
    LightManager *hostLightManager;
    LightManager *lightManager;
    BVH *bvh;
    MeshData *hostMeshes;
    MeshData *meshes;
    std::unordered_map<std::string, int> meshMap;
    int numMeshes;
    Camera *camera;
    Sampler *sampler;
    Bounds3D sceneBounds;

    int width, height;
    float fovy;
    float rr;
    int spp;
    int maxBounces;
    bool denoise = false;
};