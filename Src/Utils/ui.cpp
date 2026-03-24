#include <ui.h>
#include <camera.h>
#include "mathutil.h"
#include <scene.h>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
UI::UI(int width, int height) : width(width), height(height)
{
    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    windowWidth = width + sidebarWidth;
    windowHeight = height;
    renderOffsetX = sidebarWidth;
    window = glfwCreateWindow(windowWidth, windowHeight, "GPURayTracer", nullptr, nullptr);
    if (!window)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }
    Init();
    std::cout << "UI initialized successfully with width: " << width << ", height: " << height << std::endl;
}

UI::UI(std::string configPath)
{
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open scene file " << configPath << std::endl;
    }
    try {
        json sceneJson;
        file >> sceneJson;
        int w = 800, h = 600; // default values
        if (sceneJson.contains("camera")) {
            auto& cameraJson = sceneJson["camera"];
            if (cameraJson.contains("resolution")) {
                auto res = cameraJson["resolution"];
                w = res[0];
                h = res[1];
            }
        }
        width = w;
        height = h;
        windowWidth = width + sidebarWidth;
        windowHeight = height;
        renderOffsetX = sidebarWidth;

    } catch (const json::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
    }
    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(windowWidth, windowHeight, "GPURayTracer", nullptr, nullptr);
    if (!window)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }
    Init();
    std::cout << "UI initialized successfully with width: " << width << ", height: " << height << std::endl;
}

UI::~UI()
{
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaPBOResource, 0));
    delete shader;
    glDeleteBuffers(1, &PBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteTextures(1, &screenTexture);
}

void UI::Resize(int newWidth, int newHeight)
{
    width = newWidth;
    height = newHeight;
    windowWidth = width + sidebarWidth;
    windowHeight = height;
    renderOffsetX = sidebarWidth;
    glfwSetWindowSize(window, windowWidth, windowHeight);
    std::cout << "UI resized to render size: " << width << "x" << height
              << ", window size: " << windowWidth << "x" << windowHeight << std::endl;
}

void UI::Init()
{
    // OpenGL initialization
    float vertices[] = {
        // positions        // texture coords
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        1.0f, 1.0f, 0.0f, 1.0f, 1.0f,   // top right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
        1.0f, 1.0f, 0.0f, 1.0f, 1.0f,   // top right
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f   // top left
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // render target PBO
    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // render target texture
    glGenTextures(1, &screenTexture);
    glBindTexture(GL_TEXTURE_2D, screenTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    shader = new Shader(); // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // defaultCameraParam = cameraParam; // Store the default camera parameters for reset

    CHECK_CUDA_ERROR(cudaSetDevice(0));
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&cudaPBOResource, PBO, cudaGraphicsMapFlagsWriteDiscard));
    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaPBOResource, 0));
    CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&pixels, &numBytes, cudaPBOResource));
    if (numBytes != width * height * sizeof(uchar4))
    {
        std::cout << "Mapped PBO size does not match expected size: " << width * height * sizeof(uchar4) << " bytes";
        return;
    }

    std::cout << "ImGui initialized successfully." << std::endl;
}
void UI::UpdateTexture()
{
    glBindTexture(GL_TEXTURE_2D, screenTexture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
void UI::RenderFrameBuffer()
{
    glViewport(0, 0, windowWidth, windowHeight);
    glClear(GL_COLOR_BUFFER_BIT);
    shader->Use();
    glBindVertexArray(VAO);
    glBindTexture(GL_TEXTURE_2D, screenTexture);
    glViewport(renderOffsetX, 0, width, height);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glViewport(0, 0, windowWidth, windowHeight);
    shader->Unuse();
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);

}
void UI::GuiBegin(int spp, bool &framebufferReset, RenderParam &renderParam, DenoiseParam &denoiseParam, DenoiseTimingStats &denoiseTimingStats, bool &hardResetDenoiseHistory)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(320.0f, 255.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.84f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);

    ImGui::Text("Frame %.2f ms | %.1f FPS", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("SPP %d", spp);
    ImGui::TextDisabled("RMB rotate | MMB pan | Wheel zoom");
    ImGui::Spacing();
    ImGui::TextDisabled("Camera");
    ImGui::Separator();
    ImGui::PushItemWidth(-1);

    if(ImGui::DragFloat3("Target", &cameraParam.target(0), 0.05f, -1000.f, 1000.f, "%.2f"))
    {
        framebufferReset = true;
    }
    if(ImGui::DragFloat("Distance", &cameraParam.distance, 0.05f, cameraParam.minDistance, 1000.f, "%.2f"))
    {
        cameraParam.distance = std::max(cameraParam.distance, cameraParam.minDistance);
        framebufferReset = true;

    }
    ImGui::TextColored(ImVec4(0.72f, 0.72f, 0.72f, 1.0f), "Theta %.1f | Phi %.1f", cameraParam.theta, cameraParam.phi);
    // if (ImGui::Button("Reset Camera", ImVec2(-1, 0))) { // -1 表示宽度填满
    //     cameraParam.target = defaultCameraParam.target;
    //     cameraParam.distance = defaultCameraParam.distance;
    //     cameraParam.theta = defaultCameraParam.theta;
    //     cameraParam.phi = defaultCameraParam.phi;
    //     framebufferReset = true;
    // }
    ImGui::Spacing();
    ImGui::TextDisabled("Interaction");
    ImGui::Separator();
    ImGui::SliderFloat("Rotate Speed", &cameraParam.rotateSpeed, 0.1f, 1.0f, "%.2f deg/px");
    ImGui::SliderFloat("Zoom Speed", &cameraParam.zoomSpeed, 0.01f, 0.3f, "%.2f x Dist");
    ImGui::SliderFloat("Move Speed", &cameraParam.moveSpeed, 0.01f, 0.3f, "%.2f x Dist");

    ImGui::Spacing();
    ImGui::TextDisabled("Display");
    ImGui::Separator();
    const char* items[] = { "Color", "Depth", "Normal", "ID", "Position"};
    static int item_current = 0;
    if (ImGui::Combo("Mode", &item_current, items, IM_ARRAYSIZE(items))) {
        renderParam.renderTargetMode = item_current;
        framebufferReset = true;
    }
    ImGui::PopItemWidth();
    ImGui::End();
    ImGui::PopStyleVar(3);

    bool denoiseParamChanged = false;
    ImGui::SetNextWindowPos(ImVec2(16.0f, 286.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.84f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    ImGui::Begin("SVGF", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PushItemWidth(220.0f);
    if(ImGui::Checkbox("Enable Denoise", &renderParam.denoise)){
        denoiseParamChanged = true;
    }
    if(renderParam.denoise){
        denoiseParamChanged |= ImGui::SliderInt("History Max", &denoiseParam.maxHistoryLength, 1, 64);
        denoiseParamChanged |= ImGui::SliderFloat("Sigma Light", &denoiseParam.sigmaLight, 1.0f, 64.0f, "%.2f");
        denoiseParamChanged |= ImGui::SliderFloat("Sigma Normal", &denoiseParam.sigmaNormal, 16.0f, 512.0f, "%.1f");
        denoiseParamChanged |= ImGui::SliderFloat("Sigma Depth", &denoiseParam.sigmaDepth, 1.0f, 256.0f, "%.1f");
        denoiseParamChanged |= ImGui::SliderFloat("Reproj Depth", &denoiseParam.reprojectionDepthFactor, 0.001f, 0.2f, "%.4f");
        denoiseParamChanged |= ImGui::SliderInt("Atrous Passes", &denoiseParam.atrousIterations, 1, 8);
        if(ImGui::Button("Reset SVGF Defaults", ImVec2(-1, 0))){
            denoiseParam.maxHistoryLength = 24;
            denoiseParam.sigmaLight = 8.f;
            denoiseParam.sigmaNormal = 256.f;
            denoiseParam.sigmaDepth = 1.f;
            denoiseParam.reprojectionDepthFactor = 0.1f;
            denoiseParam.atrousIterations = 5;
            denoiseParamChanged = true;
        }

        ImGui::Separator();
        if(denoiseTimingStats.valid){
            ImGui::Text("SVGF Total %.2f ms", denoiseTimingStats.svgfTotalMs);
        }else{
            ImGui::Text("SVGF Total --");
        }
        if(ImGui::CollapsingHeader("Timing Details")){
            ImGui::Text("GBuffer %.2f", denoiseTimingStats.gbufferMs);
            ImGui::Text("Path Trace %.2f", denoiseTimingStats.pathTraceMs);
            ImGui::Text("Temporal %.2f", denoiseTimingStats.temporalMs);
            ImGui::Text("Variance %.2f", denoiseTimingStats.varianceMs);
            ImGui::Text("Atrous %.2f", denoiseTimingStats.atrousMs);
            ImGui::Text("Gather %.2f", denoiseTimingStats.gatherMs);
            ImGui::Text("Frame Total %.2f", denoiseTimingStats.totalMs);
        }
    }
    ImGui::PopItemWidth();
    ImGui::End();
    ImGui::PopStyleVar(2);

    if(denoiseParamChanged){
        framebufferReset = true;
        hardResetDenoiseHistory = true;
    }

    ImGuiIO& io = ImGui::GetIO();

    if (!io.WantCaptureMouse) {

        cameraParam.lastPosition = cameraParam.position;
        cameraParam.lastLookat = cameraParam.lookat;
        cameraParam.lastUp = cameraParam.up;
        
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
            ImVec2 delta = io.MouseDelta;
            
            cameraParam.phi -= delta.x * cameraParam.rotateSpeed;
            cameraParam.theta -= delta.y * cameraParam.rotateSpeed;

            cameraParam.theta = Clamp(0.1f, 179.9f, cameraParam.theta);
            framebufferReset = true;
        }

        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            ImVec2 delta = io.MouseDelta;
            
            cameraParam.target -= delta.x * cameraParam.du;
            cameraParam.target += delta.y * cameraParam.dv;

            framebufferReset = true;
        }


        if (io.MouseWheel != 0.0f) {
            float zoomAmount = io.MouseWheel * cameraParam.distance * cameraParam.zoomSpeed;
            cameraParam.distance -= zoomAmount;
            
            cameraParam.distance = std::max(cameraParam.distance, cameraParam.minDistance);
            framebufferReset = true;
        }

    }

    // screenshot
    if(ImGui::IsKeyPressed(ImGuiKey_P)) {
        time_t now = time(0);
        tm *ltm = localtime(&now);
        char filename[64];
        sprintf(filename, "../../output/screenshot_%04d%02d%02d_%02d%02d%02d.png",
                1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday,
                ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
        SaveScreenshot(std::string(filename));
    }

    Camera::ComputeCameraParam(cameraParam);
}


void UI::GuiEnd()
{
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void UI::SaveScreenshot(const std::string &filename)
{
    std::vector<unsigned char> hostPixels(width * height * 4);
    cudaMemcpy(hostPixels.data(), pixels, width * height * 4, cudaMemcpyDeviceToHost);
    // stbi_flip_vertically_on_write(1); 
    
    if (stbi_write_png(filename.c_str(), width, height, 4, hostPixels.data(), width * 4)) {
        std::cout << "Screenshot saved to " << filename << std::endl;
    } else {
        std::cerr << "Failed to save screenshot." << std::endl;
    }
}
