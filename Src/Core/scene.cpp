#include "scene.h"
#include "objloader.h"
#include "light.h"
#include <new>
#include <fstream>
#include "GLFW/glfw3.h"
#include <imgui_impl_opengl3_loader.h>
#include <glad/glad.h>

namespace {

std::string ResolveMaterialName(
    const json &objJson,
    const LoadedOBJMesh &loadedMesh,
    const std::unordered_map<std::string, int> &materialMap)
{
    if (objJson.contains("material")) {
        return objJson["material"];
    }

    if (objJson.contains("material_overrides") && objJson["material_overrides"].is_object()) {
        const json &overrides = objJson["material_overrides"];
        if (!loadedMesh.materialName.empty() && overrides.contains(loadedMesh.materialName)) {
            return overrides[loadedMesh.materialName];
        }
    }

    if (!loadedMesh.materialName.empty() && materialMap.find(loadedMesh.materialName) != materialMap.end()) {
        return loadedMesh.materialName;
    }

    if (objJson.contains("default_material")) {
        return objJson["default_material"];
    }

    return "";
}

std::string BuildSceneMeshName(const json &objJson, const LoadedOBJMesh &loadedMesh, bool expanded)
{
    std::string objectName = objJson["name"];
    if (!expanded) {
        return objectName;
    }
    return objectName + "::" + loadedMesh.name;
}

} // namespace

__host__ __device__ Scene::Scene(const std::string &sceneFilePath){
    if(!ParseSceneFile(sceneFilePath)){
        throw std::runtime_error("Failed to initialize scene from file: " + sceneFilePath);
    }
    BuildBVH();
    InitCameraParam();
    SetRenderParam();
}

__host__ __device__ Scene::~Scene(){
    // host
    if(hostMaterials){
        delete[] hostMaterials;
    }
    if(hostTextures){
        delete[] hostTextures;
    }
    if(hostLightManager){
        delete hostLightManager;
    }
    if(hostBVH){
        delete hostBVH;
    }
    if(hostTriangles){
        delete[] hostTriangles;
    }
    if(hostMeshes){
        delete[] hostMeshes;
    }
    if(hostSampler){
        delete hostSampler;
    }


    // devcie
    if (meshes) {
        cudaFree(meshes);
    }
    if (triangles) {
        cudaFree(triangles);
    }
    if (lightManager) {
        cudaFree(lightManager);
    }
    if (bvh) {
        cudaFree(bvh);
    }
    if(materials){
        cudaFree(materials);
    }
    if(textures){
        cudaFree(textures);
    }
    if(sampler){
        cudaFree(sampler);
    }
    if(camera){
        cudaFree(camera);
    }
}

__host__ std::vector<LoadedOBJMesh> Scene::LoadObject(const std::string &objectFilePath, Mat4f transform, Mat4f transformInv){
    return OBJLoader::LoadObject(objectFilePath, transform, transformInv);
}

__host__ __device__ void Scene::BuildBVH(){
    if (numTriangles == 0)
    {
        printf("No triangles to build BVH");
        return;
    }
    hostBVH = new BVH(hostTriangles, numTriangles, 4);
    if(hostBVH -> root){
        sceneBounds = hostBVH -> root -> bounds;
    }
    cudaMalloc(&bvh, sizeof(BVH));
    CreateBVH(hostBVH, bvh);
}

__host__ __device__ bool Scene::ParseSceneFile(const std::string& filename) {
    std::cout << "Parsing scene file: " << filename << std::endl;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Error: Cannot open scene file " << filename << std::endl;
        return false;
    }
    
    try {
        json sceneJson;
        file >> sceneJson;

        // 解析相机
        if (!ParseCamera(sceneJson["camera"])) {
            return false;
        }

        // 解析纹理
        if (!ParseTextures(sceneJson["textures"])) {
            return false;
        }
        
        // 解析材质
        if (!ParseMaterials(sceneJson["materials"])) {
            return false;
        }
        
        // 解析模型
        if (!ParseObjects(sceneJson["objects"])) {
            return false;
        }

        // 解析光源
        if (!ParseLights(sceneJson["lights"])) {
            return false;
        }

        // Sampler
        hostSampler = new Sampler(1234, width * height);
        cudaMalloc(&sampler, sizeof(Sampler));
        CreateSampler(hostSampler, sampler);

        std::cout << "Scene loaded successfully!" << std::endl;
        std::cout << "Materials: " << numMaterials << std::endl;
        std::cout << "Objects: " << numMeshes << std::endl;
        std::cout << "Triangles: " << numTriangles << std::endl;
        std::cout << "Lights: " << hostLightManager -> numLights << std::endl;

        return true;
    } catch (const json::exception& e) {
        std::cout << "JSON parsing error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cout << "Scene parsing/runtime error: " << e.what() << std::endl;
        return false;
    }
}

bool Scene::ParseCamera(const json& cameraJson) {
    if (!cameraJson.contains("type")) {
        std::cout << "Error: Camera missing 'type' field" << std::endl;
        return false;
    }

    int w, h;
    Vec3f position;
    Vec3f lookat;
    Vec3f up;
    float fovy;
    CameraType type;

    int maxBounces_ = 5;
    float rr_ = 0.8f;
    int spp_ = 16;

    if(cameraJson["type"] == "pinhole"){
        type = PINHOLE;
        std::cout << "Camera type: PINHOLE" << std::endl;
    }else{
        std::cout << "Error: Unsupported camera type " << cameraJson["type"] << std::endl;
        return false;
    }

    // 解析基本参数
    if (cameraJson.contains("resolution")) {
        auto res = cameraJson["resolution"];
        w = res[0];
        h = res[1];
        width = w, height = h;
    }

    if(cameraJson.contains("fov")){
        auto fov = cameraJson["fov"];
        this -> fovy = fov;
    }

    if(cameraJson.contains("max_bounces")){
        maxBounces_ = cameraJson["max_bounces"];
        this -> maxBounces = maxBounces_;
    }
    if(cameraJson.contains("rr")){
        rr_ = cameraJson["rr"];
        this -> rr = rr_;
    }
    if(cameraJson.contains("spp")){
        spp_ = cameraJson["spp"];
        this -> spp = spp_;
    }
    if(cameraJson.contains("direction")){
        auto pos = cameraJson["direction"];
        cameraParam.defaultDirection = Vec3f(pos[0], pos[1], pos[2]);
    }

    if(cameraJson.contains("denoise")){
        denoise = cameraJson["denoise"];
    }

    cudaMalloc(&camera, sizeof(Camera));
    Camera *hostCamera = new Camera(type);
    CreateCamera(hostCamera, camera);
    delete hostCamera;

    return true;
}

bool Scene::ParseMaterials(const json& materialsJson) {
    if (!materialsJson.is_array()) {
        std::cout << "Error: 'materials' should be an array" << std::endl;
        return false;
    }

    int materialSize = materialsJson.size();
    cudaMalloc(&materials, materialSize * sizeof(Material));
    hostMaterials = new Material[materialSize];
    numMaterials = materialSize;
    
    for(int i = 0; i < materialSize; i++){
        auto& matJson = materialsJson[i];

        if (!matJson.contains("name") || !matJson.contains("type")) {
            std::cout << "Error: Material missing 'name' or 'type' field" << std::endl;
            continue;
        }

        Material mat;
        
        std::string name = matJson["name"];
        mat.name = name;

        if(matJson["type"] == "diffuse"){
            mat.type = DIFFUSE;
        }else if(matJson["type"] == "light"){
            mat.type = LIGHT;
        }else if(matJson["type"] == "pbr"){
            mat.type = PBR;
        }else if(matJson["type"] == "subsurface"){
            mat.type = SUBSURFACE;
        }else if(matJson["type"] == "dielectric" || matJson["type"] == "glass"){
            mat.type = DIELECTRIC;
        }
        else{
            std::cout << "Error: Unsupported material type " << matJson["type"] << std::endl;
            continue;
        }

        if(mat.type == DIFFUSE){
            if (matJson.contains("basecolor")) {
                auto basecolor = matJson["basecolor"];
                mat.basecolor = Vec3f(basecolor[0], basecolor[1], basecolor[2]);
            } else {
                mat.basecolor = Vec3f(0.0f, 0.0f, 0.0f);
            }
        }else if(mat.type == LIGHT){
            if (matJson.contains("emission")) {
                auto emission = matJson["emission"];
                mat.ke = Vec3f(emission[0], emission[1], emission[2]);
            } else {
                mat.ke = Vec3f(0.0f, 0.0f, 0.0f);
            }
        }else if(mat.type == PBR){
            if (matJson.contains("basecolor")) {
                auto basecolor = matJson["basecolor"];
                mat.basecolor = Vec3f(basecolor[0], basecolor[1], basecolor[2]);
            } else {
                mat.basecolor = Vec3f(0.0f, 0.0f, 0.0f);
            }

            if(matJson.contains("metallic")){
                mat.metallic = matJson["metallic"];
            }else{
                mat.metallic = 0.0f;
            }

            if(matJson.contains("roughness")){
                mat.roughness = matJson["roughness"];
            }else{
                mat.roughness = 0.5f;
            }
        }else if(mat.type == SUBSURFACE){
            if (matJson.contains("basecolor")) {
                auto basecolor = matJson["basecolor"];
                mat.basecolor = Vec3f(basecolor[0], basecolor[1], basecolor[2]);
            } else {
                mat.basecolor = Vec3f(0.0f, 0.0f, 0.0f);
            }

            if(matJson.contains("scatter_distance")){
                auto scatter = matJson["scatter_distance"];
                mat.scatterDistance = Vec3f(scatter[0], scatter[1], scatter[2]);
            }else{
                mat.scatterDistance = Vec3f(1.0f, 1.0f, 1.0f);
            }

            if(matJson.contains("ior")){
                mat.ior = matJson["ior"];
            }

            if(matJson.contains("roughness")){
                mat.roughness = matJson["roughness"];
            }

            mat.metallic = 0.0f;
        }else if(mat.type == DIELECTRIC){
            if (matJson.contains("basecolor")) {
                auto basecolor = matJson["basecolor"];
                mat.basecolor = Vec3f(basecolor[0], basecolor[1], basecolor[2]);
            } else {
                mat.basecolor = Vec3f(1.0f, 1.0f, 1.0f);
            }

            if(matJson.contains("ior")){
                mat.ior = matJson["ior"];
            }else{
                mat.ior = 1.5f;
            }

            if(matJson.contains("transmission")){
                mat.transmission = matJson["transmission"];
            }else{
                mat.transmission = 1.0f;
            }

            if(matJson.contains("roughness")){
                mat.roughness = matJson["roughness"];
            }else{
                mat.roughness = 0.0f;
            }

            mat.metallic = 0.0f;
        }

        if(matJson.contains("diffuse_texture")){
            std::string texName = matJson["diffuse_texture"];
            if(textureMap.find(texName) != textureMap.end()){
                mat.diffuseTexture = textures + textureMap[texName];
                mat.usingDiffuseTexture = true;
            }else{
                std::cout << "Warning: Texture '" << texName << "' not found for material '" << mat.name << "'" << std::endl;
            }
        }

        materialMap[mat.name] = i;
        CreateMaterial(&mat, &materials[i]);
        hostMaterials[i] = mat;
    }

    return true;
}

 bool Scene::ParseObjects(const json& objectJson) {
    if (!objectJson.is_array()) {
        std::cout << "Error: 'objects' should be an array" << std::endl;
        return false;
    }

    std::vector<Triangle> tris;
    std::vector<MeshData> meshList;

    printf("Vertex size: %llu, offset of Normal: %llu\n", sizeof(Vertex), offsetof(Vertex, normal));

    for (int objectIndex = 0; objectIndex < objectJson.size(); objectIndex++) {
        auto &objJson = objectJson[objectIndex];

        if (!objJson.contains("name") || !objJson.contains("path")) {
            std::cout << "Error: Object missing 'name' or 'path' field" << std::endl;
            continue;
        }

        std::string path = objJson["path"];
        Mat4f transform = Mat4f::Identity();
        Mat4f translate = Mat4f::Identity();
        Mat4f rotate = Mat4f::Identity();
        Mat4f scale = Mat4f::Identity();
        if (objJson.contains("translate")) {
            auto trans = objJson["translate"];
            translate = Translate(Vec3f(trans[0], trans[1], trans[2]));
        }

        if(objJson.contains("rotate")){
            auto rot = objJson["rotate"];
            for(auto& r : rot){
                if(r.contains("rotateX")){
                    float angle = r["rotateX"];
                    rotate = RotateX(angle) * rotate;
                }else if(r.contains("rotateY")){
                    float angle = r["rotateY"];
                    rotate = RotateY(angle) * rotate;
                }else if(r.contains("rotateZ")){
                    float angle = r["rotateZ"];
                    rotate = RotateZ(angle) * rotate;
                }
            }
        }

        if(objJson.contains("scale")){
            auto s = objJson["scale"];
            scale = Scale(Vec3f(s[0], s[1], s[2]));
        }

        transform = translate * rotate * scale;
        Mat4f transformInv = transform.inverse();

        std::vector<LoadedOBJMesh> loadedMeshes = LoadObject(path, transform, transformInv);
        if (loadedMeshes.empty()) {
            std::cout << "Warning: Object '" << objJson["name"] << "' produced no meshes" << std::endl;
            continue;
        }

        bool expanded = loadedMeshes.size() > 1;
        for (LoadedOBJMesh &loadedMesh : loadedMeshes) {
            if (loadedMesh.triangles.empty()) {
                continue;
            }

            std::string materialName = ResolveMaterialName(objJson, loadedMesh, materialMap);
            if (materialName.empty() || materialMap.find(materialName) == materialMap.end()) {
                std::cout << "Warning: No valid material binding for imported mesh '" << loadedMesh.name
                          << "' from object '" << objJson["name"] << "'" << std::endl;
                continue;
            }

            MeshData mesh;
            mesh.name = BuildSceneMeshName(objJson, loadedMesh, expanded);
            mesh.meshID = static_cast<int>(meshList.size());
            mesh.materialID = materialMap[materialName];
            mesh.startTriangleID = static_cast<int>(tris.size());
            mesh.numTriangles = static_cast<int>(loadedMesh.triangles.size());
            mesh.transform = transform;
            mesh.transformInv = transformInv;

            for (Triangle &tri : loadedMesh.triangles) {
                tri.meshID = mesh.meshID;
                mesh.area += tri.area;
                tris.push_back(tri);
            }

            glGenVertexArrays(1, &mesh.vao);
            glGenBuffers(1, &mesh.vbo);
            glBindVertexArray(mesh.vao);
            glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
            glBufferData(GL_ARRAY_BUFFER, loadedMesh.vertices.size() * sizeof(Vertex), loadedMesh.vertices.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoord));
            glEnableVertexAttribArray(3);
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, barycentricCoords));
            glBindVertexArray(0);

            std::cout << "Loaded object: " << mesh.name
                      << ", triangles: " << mesh.numTriangles
                      << ", meshID: " << mesh.meshID
                      << ", material: " << materialName << std::endl;
            meshMap[mesh.name] = mesh.meshID;
            meshList.push_back(mesh);
        }
    }

    numMeshes = static_cast<int>(meshList.size());
    std::cout << "Allocating mesh buffer for " << numMeshes << " meshes" << std::endl;
    CHECK_CUDA_ERROR(cudaMalloc(&meshes, numMeshes * sizeof(MeshData)));
    hostMeshes = new MeshData[numMeshes];
    for (int i = 0; i < numMeshes; i++) {
        hostMeshes[i] = meshList[i];
        CreateMeshData(&hostMeshes[i], &meshes[i]);
    }

    numTriangles = static_cast<int>(tris.size());
    size_t triangleBytes = static_cast<size_t>(numTriangles) * sizeof(Triangle);
    std::cout << "Preparing triangle buffer for " << numTriangles
              << " triangles, sizeof(Triangle)=" << sizeof(Triangle)
              << ", total bytes=" << triangleBytes << std::endl;
    try {
        hostTriangles = new Triangle[numTriangles];
        std::cout << "Allocated host triangle buffer" << std::endl;
    } catch (const std::bad_alloc&) {
        std::cout << "Failed to allocate host triangle buffer" << std::endl;
        throw;
    }
    memcpy(hostTriangles, tris.data(), triangleBytes);
    std::cout << "Copied triangle data to host buffer" << std::endl;
    CHECK_CUDA_ERROR(cudaMalloc(&triangles, triangleBytes));
    std::cout << "Allocated device triangle buffer" << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(triangles, hostTriangles, triangleBytes, cudaMemcpyHostToDevice));
    std::cout << "Copied triangle data to device buffer" << std::endl;
    return true;
}

__host__ __device__ bool Scene::ParseLights(const json& lightsJson) {
    if (!lightsJson.is_array()) {
        std::cout << "Error: 'lights' should be an array" << std::endl;
        return false;
    }

    int lightSize = lightsJson.size();
    hostLightManager = new LightManager(lightSize);

    for(int i = 0; i < lightSize; i++){
        Light light;

        auto& lightJson = lightsJson[i];

        if (!lightJson.contains("type")) {
            std::cout << "Error: Light missing 'type' field" << std::endl;
            continue;
        }

        if(lightJson["type"] == "area"){
           light.type = AREA_LIGHT;
            if(lightJson.contains("ke")){
                auto em = lightJson["ke"];
                light.emission = Vec3f(em[0], em[1], em[2]);
            }

            if(lightJson.contains("material")){
                std::string matName = lightJson["material"];
                if (materialMap.find(matName) != materialMap.end()) {
                    int matId = materialMap[matName];
                    Material mat = hostMaterials[matId];
                    light.emission = mat.ke;
                } else {
                    std::cout << "Warning: Material '" << matName << "' not found for light" << std::endl;
                }
            }

            std::string lightMeshName = lightJson["lightMeshName"];
            if(meshMap.find(lightMeshName) == meshMap.end()){
                std::cout << "Error: Light mesh '" << lightMeshName << "' not found" << std::endl;
                continue;
            }
            int meshId = meshMap[lightMeshName];
            MeshData *hostMesh = &hostMeshes[meshId];
            hostMesh -> lightID = i;
            CreateMeshData(hostMesh, &meshes[meshId]);
            light.mesh = meshes + meshId;

            CreateLight(&light, &(hostLightManager -> lights[i]));
        }else if(lightJson["type"] == "env"){
            light.type = ENV_LIGHT;
            if(lightJson.contains("path")){
                std::string path = lightJson["path"];
                std::cout << "Loading environment light from: " << path << std::endl;
                HDRTexture *hostEnvironmentMap = new HDRTexture(path);
                printf("cudaTextureObj: %llu\n", hostEnvironmentMap -> cudaTextureObj);
                HDRTexture *deviceEnvironmentMap;
                CHECK_CUDA_ERROR(cudaMalloc(&deviceEnvironmentMap, sizeof(HDRTexture)));
                CreateHDRTexture(hostEnvironmentMap, deviceEnvironmentMap);
                light.envMap = deviceEnvironmentMap;
            }else{
                std::cout << "Error: Env light missing 'path' field" << std::endl;
                continue;
            }
            CreateLight(&light, &(hostLightManager -> lights[i]));
            hostLightManager -> envMapLightIdx = i;
        }
        else{
            std::cout << "Error: Unsupported light type " << lightJson["type"] << std::endl;
            continue;
        }
    }

    cudaMalloc(&lightManager, sizeof(LightManager));
    CreateLightManager(hostLightManager, lightManager);

    return true;
}

__host__ __device__ bool Scene::ParseTextures(const json& texturesJson) {
    if (!texturesJson.is_array()) {
        std::cout << "Error: 'textures' should be an array" << std::endl;
        return false;
    }

    int textuerSize = texturesJson.size();
    hostTextures = new Texture[textuerSize];
    cudaMalloc((void**)&textures, textuerSize * sizeof(Texture));
    for(int i = 0; i < textuerSize; i++){
        auto& textureJson = texturesJson[i];

        if (!textureJson.contains("path") || !textureJson.contains("name")) {
            std::cout  << "Error: Texture missing 'path' or 'name' field" << std::endl;
            continue;
        }

        std::string path = textureJson["path"];
        std::string name = textureJson["name"];
        Texture &tex = hostTextures[i];
        tex.textureName = name;
        tex.Load(path);
        CreateTexture(&tex, &textures[i]);
        textureMap[name] = i;
    }
    return true;
}

__host__ void Scene::InitCameraParam(){
    cameraParam.target = sceneBounds.Center();
    cameraParam.fovy = this -> fovy;
    cameraParam.width = this -> width;
    cameraParam.height = this -> height;
    cameraParam.maxBounces = this -> maxBounces;
    cameraParam.rr = this -> rr;
    cameraParam.spp = this -> spp;
    cameraParam.distance = sceneBounds.DiagonalLength() * 0.5f / sin(AngleToRadian(cameraParam.fovy) / 2.f)  * 0.6f;
    Vec3f dir = cameraParam.defaultDirection.normalized();
    cameraParam.theta = acosf(dir(1) / dir.norm());
    cameraParam.phi = atan2f(dir(0), dir(2));
    cameraParam.theta = RadianToAngle(cameraParam.theta);
    cameraParam.phi = RadianToAngle(cameraParam.phi);
    cameraParam.rotateSpeed = 0.3f;
    cameraParam.zoomSpeed = 0.1f;
    cameraParam.moveSpeed = 0.1f;
    Camera::ComputeCameraParam(cameraParam);
    cameraParam.lastLookat = cameraParam.lookat;
    cameraParam.lastPosition = cameraParam.position;
    cameraParam.lastUp = cameraParam.up; // avoid nan
}

__host__ void Scene::SetRenderParam(){
    renderParam.width = width;
    renderParam.height = height;
    renderParam.triangles = triangles;
    renderParam.meshData = meshes;
    renderParam.materials = materials;
    renderParam.lightManager = lightManager;
    renderParam.bvh = bvh;
    renderParam.camera = camera;
    renderParam.sampler = sampler;
    renderParam.sceneBounds = sceneBounds;
    renderParam.rr = rr;
    renderParam.maxBounces = maxBounces;
    renderParam.spp = spp;
    renderParam.currentRenderBufferIndex = 0;
    renderParam.currentGBufferIndex = 0;
    renderParam.hostTriangles = hostTriangles;
    renderParam.hostMeshes = hostMeshes;
    renderParam.hostMaterials = hostMaterials;
    renderParam.numMeshes = numMeshes;
    renderParam.denoise = denoise;
}
