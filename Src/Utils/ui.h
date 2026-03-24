#pragma once
#include "glad/glad.h"
#include "global.h"
#include <iostream>
#include <memory>
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"
#include "shader.h"
#include "camera.h"
#include "scene.h"

struct DenoiseParam{
    float2 *directMoments[2];
    float2 *indirectMoments[2];
    int *historyLength[2];
    int currentTemporalBufferIndex = 0;
    int maxHistoryLength = 24;
    float sigmaLight = 8.f;
    float sigmaNormal = 256.f;
    float sigmaDepth = 1000.f;
    float reprojectionDepthFactor = 0.1f;
    int atrousIterations = 5;
};

struct DenoiseTimingStats{
    float gbufferMs = 0.f;
    float pathTraceMs = 0.f;
    float temporalMs = 0.f;
    float varianceMs = 0.f;
    float atrousMs = 0.f;
    float svgfTotalMs = 0.f;
    float gatherMs = 0.f;
    float totalMs = 0.f;
    bool valid = false;
};

class UI {
public:
    UI(int width, int height);
    UI(std::string filepath);
    ~UI();
    void Resize(int width, int height);
    void Init();
    void UpdateTexture();
    void RenderFrameBuffer();
    void GuiBegin(int spp, bool &framebufferReset, RenderParam &renderParam, DenoiseParam &denoiseParam, DenoiseTimingStats &denoiseTimingStats, bool &hardResetDenoiseHistory);
    void GuiEnd();
    void SaveScreenshot(const std::string &filename);
    GLFWwindow* window;
    Shader* shader;
    int width, height;
    int windowWidth, windowHeight;
    int sidebarWidth = 352;
    int renderOffsetX = 352;
    GLuint VAO, VBO, PBO, screenTexture;
    cudaGraphicsResource* cudaPBOResource;
    // // For reset camera movement
    // CameraParam defaultCameraParam;
    uchar4* pixels;
    size_t numBytes;
};
