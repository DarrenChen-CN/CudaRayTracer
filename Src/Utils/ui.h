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

class UI {
public:
    UI(int width, int height);
    ~UI();
    void Resize(int width, int height);
    void Init();
    void UpdateTexture();
    void RenderFrameBuffer();
    void GuiBegin(int spp);
    void GuiEnd();
    GLFWwindow* window;
    Shader* shader;
    int width, height;
    GLuint VAO, VBO, PBO, screenTexture;
    cudaGraphicsResource* cudaPBOResource;
    float *accumulateColors;
};