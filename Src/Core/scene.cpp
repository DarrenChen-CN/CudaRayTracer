#include "scene.h"
#include "objloader.h"
#include "light.h"
#include <new>
#include <fstream>


__host__ __device__ Scene::Scene(const std::string &sceneFilePath){
    ParseSceneFile(sceneFilePath);
    BuildBVH();
    InitCameraParam();
    SetRenderParam();
}

__host__ __device__ Scene::~Scene(){
    // Free device memory
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
}

__host__ __device__ std::vector<Triangle> Scene::LoadObject(const std::string &objectFilePath, int id, Mat4f transform, Mat4f transformInv){
    return OBJLoader::LoadObject(id, objectFilePath, transform);
}

__host__ __device__ void Scene::BuildBVH(){
    if (numTriangles == 0)
    {
        printf("No triangles to build BVH");
        return;
    }
    BVH *hostBVH = new BVH(hostTriangles, numTriangles, 4);
    if(hostBVH -> root){
        sceneBounds = hostBVH -> root -> bounds;
    }
    cudaMalloc(&bvh, sizeof(BVH));
    CreateBVH(hostBVH, bvh);
    delete hostBVH;
}

__host__ __device__ bool Scene::ParseSceneFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open scene file " << filename << std::endl;
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
        Sampler *hostSampler = new Sampler(1234, width * height);
        cudaMalloc(&sampler, sizeof(Sampler));
        CreateSampler(hostSampler, sampler);

        std::cout << "Scene loaded successfully!" << std::endl;
        // std::cout << "  Camera: " << camera -> type << std::endl;
        std::cout << "Materials: " << numMaterials << std::endl;
        std::cout << "Objects: " << numMeshes << std::endl;
        std::cout << "Triangles: " << numTriangles << std::endl;
        std::cout << "Lights: " << hostLightManager -> numLights << std::endl;

        return true;
    } catch (const json::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
        return false;
    }
}

bool Scene::ParseCamera(const json& cameraJson) {
    if (!cameraJson.contains("type")) {
        std::cerr << "Error: Camera missing 'type' field" << std::endl;
        return false;
    }

    int w, h;
    Vec3f position;
    Vec3f lookat;
    Vec3f up;
    float fovy;
    CameraType type;

    int maxBounces_ = 5; // Default max bounces
    float rr_ = 0.8f; // Default Russian roulette probability
    int spp_ = 16; // Default samples per pixel

    if(cameraJson["type"] == "pinhole"){
        type = PINHOLE;
        std::cout << "Camera type: PINHOLE" << std::endl;
    }else{
        std::cerr << "Error: Unsupported camera type " << cameraJson["type"] << std::endl;
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

    cudaMalloc(&camera, sizeof(Camera));
    Camera *hostCamera = new Camera(type);
    CreateCamera(hostCamera, camera);
    delete hostCamera;

    return true;
}

bool Scene::ParseMaterials(const json& materialsJson) {
    if (!materialsJson.is_array()) {
        std::cerr << "Error: 'materials' should be an array" << std::endl;
        return false;
    }

    int materialSize = materialsJson.size();
    cudaMalloc(&materials, materialSize * sizeof(Material));
    hostMaterials = new Material[materialSize];
    numMaterials = materialSize;
    
    for(int i = 0; i < materialSize; i++){
        auto& matJson = materialsJson[i];

        if (!matJson.contains("name") || !matJson.contains("type")) {
            std::cerr << "Error: Material missing 'name' or 'type' field" << std::endl;
            continue;
        }

        Material mat;
        
        std::string name = matJson["name"];
        // materials[i].name = name;
        mat.name = name;
        // std::cout << "Loading material: " << name << std::endl;

        if(matJson["type"] == "diffuse"){
            // materials[i].type = DIFFUSE;
            mat.type = DIFFUSE;
        }else if(matJson["type"] == "light"){
            // materials[i].type = LIGHT;
            mat.type = LIGHT;
        }else if(matJson["type"] == "pbr"){
            mat.type = PBR;
        }
        else{
            std::cerr << "Error: Unsupported material type " << matJson["type"] << std::endl;
            continue;
        }

        if(mat.type == DIFFUSE){
            if (matJson.contains("basecolor")) {
                auto basecolor = matJson["basecolor"];
                mat.basecolor = Vec3f(basecolor[0], basecolor[1], basecolor[2]);
                mat.kd = mat.basecolor;
            } else {
                mat.basecolor = Vec3f(0.0f, 0.0f, 0.0f);
                mat.kd = Vec3f(0.0f, 0.0f, 0.0f);
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
            std::cout << "Material " << mat.name << ": basecolor=(" << mat.basecolor(0) << "," << mat.basecolor(1) << "," << mat.basecolor(2) << "), metallic=" << mat.metallic << ", roughness=" << mat.roughness << std::endl;
            mat.F0 = Lerp(Vec3f(0.04f, 0.04f, 0.04f), mat.basecolor, mat.metallic);
            std::cout << "F0: " << mat.F0(0) << ", " << mat.F0(1) << ", " << mat.F0(2) << std::endl;
            // mat.kd = (Vec3f(1.f, 1.f, 1.f) - mat.ks) * (1.0f - mat.metallic);
        }

        materialMap[mat.name] = i;
        CreateMaterial(&mat, &materials[i]);
        hostMaterials[i] = mat;
    }

    return true;
}

 bool Scene::ParseObjects(const json& objectJson) {
    if (!objectJson.is_array()) {
        std::cerr << "Error: 'objects' should be an array" << std::endl;
        return false;
    }

    int objectSize = objectJson.size();
    numMeshes = objectSize;
    cudaMalloc(&meshes, objectSize * sizeof(MeshData));
    hostMeshes = new MeshData[objectSize];

    std::vector<Triangle> tris;

    for(int i = 0; i < numMeshes; i ++){
        MeshData mesh;
        auto& objJson = objectJson[i];

        if (!objJson.contains("name") || !objJson.contains("path")) {
            std::cerr << "Error: Object missing 'name' or 'path' field" << std::endl;
            continue;
        }

        int materialId = -1;
        if (objJson.contains("material")) {
            std::string matName = objJson["material"];
            if (materialMap.find(matName) != materialMap.end()) {
                materialId = materialMap[matName];
            } else {
                std::cerr << "Warning: Material '" << matName << "' not found for object '" << objJson["name"] << "'" << std::endl;
            }
        } else {
            std::cerr << "Warning: Object '" << objJson["name"] << "' has no material specified" << std::endl;
        }

        mesh.name = objJson["name"];
        mesh.meshID = i;
        meshMap[mesh.name] = i;
        mesh.materialID = materialId;

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
        std::vector<Triangle> objectTris = LoadObject(path, i, transform, transformInv);
        mesh.startTriangleID = tris.size();
        mesh.numTriangles = objectTris.size();
        for(auto& tri : objectTris)mesh.area += tri.area;
        mesh.transform = transform;
        mesh.transformInv = transformInv;
        hostMeshes[i] = mesh;
        tris.insert(tris.end(), objectTris.begin(), objectTris.end());
        CreateMeshData(&mesh, &meshes[i]);
    }

    numTriangles = tris.size();
    cudaMalloc(&triangles, numTriangles * sizeof(Triangle));
    hostTriangles = new Triangle[numTriangles];
    // for(int i = 0; i < numTriangles; i++){
    //     printf("Triangle %d: v0=(%f,%f,%f), v1=(%f,%f,%f), v2=(%f,%f,%f), normal=(%f,%f,%f), area=%f, meshID=%d\n", i,
    //            tris[i].v0.position(0), tris[i].v0.position(1), tris[i].v0.position(2),
    //            tris[i].v1.position(0), tris[i].v1.position(1), tris[i].v1.position(2),
    //            tris[i].v2.position(0), tris[i].v2.position(1), tris[i].v2.position(2),
    //            tris[i].normal(0), tris[i].normal(1), tris[i].normal(2),
    //            tris[i].area, tris[i].meshID);
    // }
    memcpy(hostTriangles, tris.data(), numTriangles * sizeof(Triangle));
    for(int i = 0; i < numTriangles; i++){
        CreateTriangle(&hostTriangles[i], &triangles[i]);
    }
    return true;
}

__host__ __device__ bool Scene::ParseLights(const json& lightsJson) {
    if (!lightsJson.is_array()) {
        std::cerr << "Error: 'lights' should be an array" << std::endl;
        return false;
    }

    int lightSize = lightsJson.size();
    hostLightManager = new LightManager(lightSize);

    for(int i = 0; i < lightSize; i++){
        Light light;

        auto& lightJson = lightsJson[i];

        if (!lightJson.contains("type")) {
            std::cerr << "Error: Light missing 'type' field" << std::endl;
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
                    // Ensure the material is of type LIGHT
                    Material mat = hostMaterials[matId];
                    light.emission = mat.ke;
                } else {
                    std::cerr << "Warning: Material '" << matName << "' not found for light" << std::endl;
                }
            }

            std::string lightMeshName = lightJson["lightMeshName"];
            if(meshMap.find(lightMeshName) == meshMap.end()){
                std::cerr << "Error: Light mesh '" << lightMeshName << "' not found" << std::endl;
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
                HDRTexture *hostEnvironmentMap = new HDRTexture(path);
                printf("cudaTextureObj: %llu\n", hostEnvironmentMap -> cudaTextureObj);
                HDRTexture *deviceEnvironmentMap;
                CHECK_CUDA_ERROR(cudaMalloc(&deviceEnvironmentMap, sizeof(HDRTexture)));
                CreateHDRTexture(hostEnvironmentMap, deviceEnvironmentMap);
                light.envMap = deviceEnvironmentMap;
            }else{
                std::cerr << "Error: Env light missing 'path' field" << std::endl;
                continue;
            }
            CreateLight(&light, &(hostLightManager -> lights[i]));
            hostLightManager -> envMapLightIdx = i;
        }
        else{
            std::cerr << "Error: Unsupported light type " << lightJson["type"] << std::endl;
            continue;
        }
    }

    cudaMalloc(&lightManager, sizeof(LightManager));
    CreateLightManager(hostLightManager, lightManager);

    return true;
}

__host__ __device__ bool Scene::ParseTextures(const json& texturesJson) {
    if (!texturesJson.is_array()) {
        std::cerr << "Error: 'textures' should be an array" << std::endl;
        return false;
    }

    int textuerSize = texturesJson.size();
    for(int i = 0; i < textuerSize; i++){
        auto& textureJson = texturesJson[i];

        if (!textureJson.contains("type")) {
            std::cerr << "Error: Texture missing 'type' field" << std::endl;
            continue;
        }
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
    cameraParam.distance = sceneBounds.DiagonalLength() * 0.5f / sin(AngleToRadian(cameraParam.fovy) / 2.f);
    // cameraParam.theta = 90.f;
    // cameraParam.phi = 180.f;
    Vec3f dir = cameraParam.defaultDirection.normalized();
    cameraParam.theta = acosf(dir(1) / dir.norm());
    cameraParam.phi = atan2f(dir(0), dir(2));
    cameraParam.theta = RadianToAngle(cameraParam.theta);
    cameraParam.phi = RadianToAngle(cameraParam.phi);
    cameraParam.rotateSpeed = 0.3f;
    cameraParam.zoomSpeed = 0.1f;
    cameraParam.moveSpeed = 0.1f;
    Camera::ComputeCameraParam(cameraParam);
    std::cout << "Camera initialized. Position: (" << cameraParam.position(0) << ", " << cameraParam.position(1) << ", " << cameraParam.position(2) << ")" << std::endl;
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
}