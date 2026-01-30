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

class UI {
public:
    UI(int width, int height);
    UI(std::string filepath);
    ~UI();
    void Resize(int width, int height);
    void Init();
    void UpdateTexture();
    void RenderFrameBuffer();
    void GuiBegin(int spp, bool &framebufferReset);
    void GuiEnd();
    void SaveScreenshot(const std::string &filename);
    GLFWwindow* window;
    Shader* shader;
    int width, height;
    GLuint VAO, VBO, PBO, screenTexture;
    cudaGraphicsResource* cudaPBOResource;
    // // For reset camera movement
    // CameraParam defaultCameraParam;
    uchar4* pixels;
    size_t numBytes;
};