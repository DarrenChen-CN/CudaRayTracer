// #include "global.h"
// #include "ui.h"
// #include <glad/glad.h>
// #include <GLFW/glfw3.h>
// #include <cuda_runtime.h>
// #include <cuda_gl_interop.h>
// #include "integrator.h"
// #include "pipeline.h"
// #include "triangle.h"
// #include "objloader.h"
// #include "scene.h"
// #include <thrust/device_vector.h>

// // 生成每个像素的多个随机数（每个像素多个采样）
// static __global__ void generate_kernel(curandState *state, float *output, int numPixels, int numSamplesPerPixel)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < numPixels)
//     {
//         int startIdx = idx * numSamplesPerPixel; // 每个像素的起始位置

//         for (int i = 0; i < numSamplesPerPixel; i++)
//         {
//             output[startIdx + i] = curand_uniform(&state[idx]);
//         }
//     }
// }

// void Init()
// {
//     // initialize logger
//     Logger::Init();
// }

// void UITest()
// {
//     int width = 1024, height = 1024;
//     Vec3f position(278.0, 273.0, -800.0);
//     Vec3f lookat(278.0, 273.0, -799.0);
//     Vec3f up(0, 1, 0);

//     // Vec3f position(0, 1, 4);
//     // Vec3f lookat(0, 1, 0);
//     // Vec3f up(0, 1, 0);
//     float fovy = 39.3077f;
//     UI *ui = new UI(width, height);
//     Integrator *hostIntegrator = new Integrator(32, 0.8f);
//     Integrator *deviceIntegrator;
//     CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceIntegrator, sizeof(Integrator)));
//     CHECK_CUDA_ERROR(cudaMemcpy(deviceIntegrator, hostIntegrator, sizeof(Integrator), cudaMemcpyHostToDevice));
//     Pipeline *pipeline = new Pipeline(width, height, deviceIntegrator, ui->GetPBO());

//     // Main loop
//     while (!glfwWindowShouldClose(ui->GetWindow()))
//     {
//         // LOG_DEBUG("here");
//         ui->GuiBegin();
//         ImGui::Begin("CPURayTracer GUI", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
//         ImGui::SetWindowPos(ImVec2(0, 0));
//         ImGui::SetWindowSize(ImVec2(350, 50));
//         ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
//         // ImGui::Text("spp %d", hostIntegrator -> GetSPP());
//         ImGui::End();
//         // LOG_DEBUG("Rendering time: {}ms", 1000.0f / ImGui::GetIO().Framerate);
//         // Rendering...
//         // pipeline->RunCuda();
//         cudaDeviceSynchronize(); // Render the frame buffer
//         ui->UpdateTexture();
//         ui->RenderFrameBuffer();
//         ui->GuiEnd();

//         // Swap buffers and poll events
//         glfwSwapBuffers(ui->GetWindow());
//         glfwSwapInterval(0); // Disable VSync
//         glfwPollEvents();
//     }
// }

// Ray *deviceRay;

// // void TriangleTest()
// // {
// //     // size_t heapLimit = 128 * 1024 * 1024; // 设置为 128MB
// //     // cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapLimit);
// //     int width = 1024, height = 1024;
// //     Vec3f position(0, 0, 4);
// //     Vec3f lookat(0, 0, 0);
// //     Vec3f up(0, 1, 0);

// //     float fovy = 39.3077f;
// //     UI *ui = new UI(width, height);

// //     // Create a triangle mesh
// //     Vertex v0 = {Vec3f(-1.0f, -1.0f, 0.0f), Vec3f(0.0f, 0.0f, 1.0f), Vec2f(0.0f, 0.0f), Vec3f(1.0f, 0.0f, 0.0f)};
// //     Vertex v1 = {Vec3f(1.0f, -1.0f, 0.0f), Vec3f(0.0f, 0.0f, 1.0f), Vec2f(1.0f, 0.0f), Vec3f(0.0f, 1.0f, 0.0f)};
// //     Vertex v2 = {Vec3f(0.0f, 1.0f, 0.0f), Vec3f(0.0f, 0.0f, 1.0f), Vec2f(0.5f, 1.0f), Vec3f(0.0f, 0.0f, 1.0f)};
// //     CPUTriangle *triangle = new CPUTriangle(v0, v1, v2);
// //     CPUTriangleMesh *triangleMesh = new CPUTriangleMesh({triangle});

// //     // printf("CPUTriangle size: %zu\n", sizeof(CPUTriangle));
// //     // printf("Device Triangle size: %zu\n", sizeof(Triangle));

// //     // Create a triangle mesh on the device
// //     TriangleMesh *deviceTriangleMesh = CreateTriangleMesh(triangleMesh);

// //     Camera *camera = CreatePinholeCamera(position, lookat, up, fovy, width, height);

// //     Integrator *hostIntegrator = new Integrator(32, 0.8f);
// //     Integrator *deviceIntegrator;
// //     CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceIntegrator, sizeof(Integrator)));
// //     CHECK_CUDA_ERROR(cudaMemcpy(deviceIntegrator, hostIntegrator, sizeof(Integrator), cudaMemcpyHostToDevice));

// //     RandomSampler *sampler = CreateRandomSampler(0, width, height);

// //     CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceRay, width * height * sizeof(Ray)));
// //     CHECK_CUDA_ERROR(cudaDeviceSynchronize());

// //     Pipeline *pipeline = new Pipeline(width, height, deviceIntegrator, ui->GetPBO());

// //     // Main loop
// //     while (!glfwWindowShouldClose(ui->GetWindow()))
// //     {
// //         // LOG_DEBUG("here");
// //         ui->GuiBegin();
// //         ImGui::Begin("CPURayTracer GUI", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
// //         ImGui::SetWindowPos(ImVec2(0, 0));
// //         ImGui::SetWindowSize(ImVec2(350, 50));
// //         ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
// //         // ImGui::Text("spp %d", hostIntegrator -> GetSPP());
// //         ImGui::End();
// //         // LOG_DEBUG("Rendering time: {}ms", 1000.0f / ImGui::GetIO().Framerate);
// //         // Rendering...
// //         pipeline->RunCuda(camera, deviceTriangleMesh, sampler, deviceRay);
// //         cudaDeviceSynchronize(); // Render the frame buffer
// //         ui->UpdateTexture();
// //         ui->RenderFrameBuffer();
// //         ui->GuiEnd();

// //         // Swap buffers and poll events
// //         glfwSwapBuffers(ui->GetWindow());
// //         glfwSwapInterval(0); // Disable VSync
// //         glfwPollEvents();
// //     }
// // }

// // void BVHTest()
// // {
// //     int width = 1024, height = 1024;
// //     Vec3f position(278.0, 273.0, -800.0);
// //     Vec3f lookat(278.0, 273.0, -799.0);
// //     Vec3f up(0, 1, 0);

// //     float fovy = 39.3077f;
// //     UI *ui = new UI(width, height);

// //     std::string objPath = "E:/code/projects/CPURayTracer/Model/Cornellbox/cornell-box.obj";
// //     std::vector<CPUTriangleMesh *> meshes = OBJLoader::LoadObject(objPath);

// //     int meshCount = meshes.size();

// //     TriangleMesh **deviceTriangleMeshes;
// //     CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceTriangleMeshes, meshCount * sizeof(TriangleMesh *)));
// //     TriangleMesh **hostTriangleMeshes = new TriangleMesh *[meshCount];

// //     for (int i = 0; i < meshCount; i++)
// //     {
// //         CPUTriangleMesh *cpuMesh = meshes[i];
// //         TriangleMesh *deviceMesh = CreateTriangleMesh(cpuMesh);
// //         hostTriangleMeshes[i] = deviceMesh;
// //     }

// //     CHECK_CUDA_ERROR(cudaMemcpy(deviceTriangleMeshes, hostTriangleMeshes, meshCount * sizeof(TriangleMesh *), cudaMemcpyHostToDevice));

// //     Camera *camera = CreatePinholeCamera(position, lookat, up, fovy, width, height);

// //     Integrator *hostIntegrator = new Integrator(32, 0.8f);
// //     Integrator *deviceIntegrator;
// //     CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceIntegrator, sizeof(Integrator)));
// //     CHECK_CUDA_ERROR(cudaMemcpy(deviceIntegrator, hostIntegrator, sizeof(Integrator), cudaMemcpyHostToDevice));

// //     RandomSampler *sampler = CreateRandomSampler(0, width, height);

// //     CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceRay, width * height * sizeof(Ray)));
// //     CHECK_CUDA_ERROR(cudaDeviceSynchronize());

// //     Pipeline *pipeline = new Pipeline(width, height, deviceIntegrator, ui->GetPBO());

// //     // Main loop
// //     while (!glfwWindowShouldClose(ui->GetWindow()))
// //     {
// //         // LOG_DEBUG("here");
// //         ui->GuiBegin();
// //         ImGui::Begin("CPURayTracer GUI", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
// //         ImGui::SetWindowPos(ImVec2(0, 0));
// //         ImGui::SetWindowSize(ImVec2(350, 50));
// //         ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
// //         // ImGui::Text("spp %d", hostIntegrator -> GetSPP());
// //         ImGui::End();
// //         // LOG_DEBUG("Rendering time: {}ms", 1000.0f / ImGui::GetIO().Framerate);
// //         // Rendering...
// //         pipeline->RunCuda(camera, deviceTriangleMeshes, meshCount, sampler, deviceRay);
// //         cudaDeviceSynchronize(); // Render the frame buffer
// //         ui->UpdateTexture();
// //         ui->RenderFrameBuffer();
// //         ui->GuiEnd();

// //         // Swap buffers and poll events
// //         glfwSwapBuffers(ui->GetWindow());
// //         glfwSwapInterval(0); // Disable VSync
// //         glfwPollEvents();
// //     }
// // }

// void SceneTest()
// {
//     int width = 1024, height = 1024;
//     Vec3f position(278.0, 273.0, -800.0);
//     Vec3f lookat(278.0, 273.0, -799.0);
//     Vec3f up(0, 1, 0);

//     float fovy = 39.3077f;
//     Camera *camera = CreatePinholeCamera(position, lookat, up, fovy, width, height);
//     UI *ui = new UI(width, height);

//     std::string objPath = "E:/code/projects/CPURayTracer/Model/Cornellbox/cornell-box.obj";
//     Scene *scene = new Scene();
//     scene->LoadObject(objPath);
//     scene->camera = camera;

//     Scene *deviceScene;
//     CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceScene, sizeof(Scene)));
//     CHECK_CUDA_ERROR(cudaMemcpy(deviceScene, scene, sizeof(Scene), cudaMemcpyHostToDevice));

//     Integrator *hostIntegrator = new Integrator(8, 0.8f);
//     Integrator *deviceIntegrator;
//     CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceIntegrator, sizeof(Integrator)));
//     CHECK_CUDA_ERROR(cudaMemcpy(deviceIntegrator, hostIntegrator, sizeof(Integrator), cudaMemcpyHostToDevice));

//     RandomSampler *sampler = CreateRandomSampler(0, width, height);

//     Pipeline *pipeline = new Pipeline(width, height, deviceIntegrator, ui->GetPBO());

//     // Main loop
//     while (!glfwWindowShouldClose(ui->GetWindow()))
//     {
//         // LOG_DEBUG("here");
//         ui->GuiBegin();
//         ImGui::Begin("CPURayTracer GUI", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
//         ImGui::SetWindowPos(ImVec2(0, 0));
//         ImGui::SetWindowSize(ImVec2(350, 50));
//         ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
//         // ImGui::Text("spp %d", hostIntegrator -> GetSPP());
//         ImGui::End();
//         // LOG_DEBUG("Rendering time: {}ms", 1000.0f / ImGui::GetIO().Framerate);
//         // Rendering...
//         pipeline->RunCuda(deviceScene, camera, sampler);
//         cudaDeviceSynchronize(); // Render the frame buffer
//         ui->UpdateTexture();
//         ui->RenderFrameBuffer();
//         ui->GuiEnd();

//         // Swap buffers and poll events
//         glfwSwapBuffers(ui->GetWindow());
//         glfwSwapInterval(0); // Disable VSync
//         glfwPollEvents();
//     }
// }