#include <ui.h>
#include <camera.h>
#include "mathutil.h"
UI::UI(int width, int height) : width(width), height(height)
{
    // Initialize GLFW
    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(width, height, "GPURayTracer", nullptr, nullptr);
    if (!window)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    } // Make the window's context current
    glfwMakeContextCurrent(window);
    // Initialize GLAD
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
}

void UI::Resize(int newWidth, int newHeight)
{
    width = newWidth;
    height = newHeight;
    glfwSetWindowSize(window, width, height);
    std::cout << "UI resized to width: " << width << ", height: " << height << std::endl;
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

    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glGenTextures(1, &screenTexture);
    glBindTexture(GL_TEXTURE_2D, screenTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    // shader = std::make_shared<Shader>("../../../Src/Shader/shader.vert", "../../../Src/Shader/shader.frag");
    // shader = new Shader("E:\\code\\projects\\CPURayTracer\\Src\\Shader\\shader.vert", "E:\\code\\projects\\CPURayTracer\\Src\\Shader\\shader.frag");
    // shader = std::make_shared<Shader>("..\\..\\Src\\Shader\\shader.vert", "..\\..\\Src\\Shader\\shader.frag");
    shader = new Shader(); // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    defaultCameraParam = cameraParam; // Store the default camera parameters for reset

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
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);
    // Use the shader program
    shader->Use();
    // Bind the VAO and draw the screen quad
    glBindVertexArray(VAO);
    // Bind the texture
    glBindTexture(GL_TEXTURE_2D, screenTexture);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    // Unbind everything
    shader->Unuse();
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
}
void UI::GuiBegin(int spp, bool &framebufferReset)
{
    // Start a new ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::SetWindowPos(ImVec2(0, 0));
    ImGui::SetWindowSize(ImVec2(350, 320));
    ImGui::Begin("CudaRayTracer GUI", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("spp %d", spp);
    ImGui::Separator();

    // Camera
    ImGui::Text("Camera Transform");
    if(ImGui::DragFloat3("Target", &cameraParam.target(0), 0.05f, -1000.f, 1000.f, "%.2f"))
    {
        framebufferReset = true;
    }
    if(ImGui::DragFloat("Distance", &cameraParam.distance, 0.05f, cameraParam.minDistance, 1000.f, "%.2f"))
    {
        cameraParam.distance = std::max(cameraParam.distance, cameraParam.minDistance);
        framebufferReset = true;

    }
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Theta: %.1f | Phi: %.1f", cameraParam.theta, cameraParam.phi);
    if (ImGui::Button("Reset Camera", ImVec2(-1, 0))) { // -1 表示宽度填满
        cameraParam.target = defaultCameraParam.target;
        cameraParam.distance = defaultCameraParam.distance;
        cameraParam.theta = defaultCameraParam.theta;
        cameraParam.phi = defaultCameraParam.phi;
        framebufferReset = true;
    }
    ImGui::Separator();

    // Input Sensitivity
    ImGui::Text("Input Sensitivity");
    ImGui::SliderFloat("Rotate Speed", &cameraParam.rotateSpeed, 0.1f, 1.0f, "%.2f deg/px");
    ImGui::SliderFloat("Zoom Speed", &cameraParam.zoomSpeed, 0.01f, 0.3f, "%.2f x Dist");
    ImGui::End();

    ImGuiIO& io = ImGui::GetIO();

    if (!io.WantCaptureMouse) {
        
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
            ImVec2 delta = io.MouseDelta;
            
            cameraParam.phi -= delta.x * cameraParam.rotateSpeed;
            cameraParam.theta -= delta.y * cameraParam.rotateSpeed;

            // cameraParam.theta = Clamp(0.f, 180.0f, cameraParam.theta);
            framebufferReset = true;
        }

        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            ImVec2 delta = io.MouseDelta;
            
            cameraParam.target -= delta.x * cameraParam.du;
            cameraParam.target += delta.y * cameraParam.dv;

            // cameraParam.theta = Clamp(0.f, 180.0f, cameraParam.theta);
            framebufferReset = true;
        }


        if (io.MouseWheel != 0.0f) {
            float zoomAmount = io.MouseWheel * cameraParam.distance * cameraParam.zoomSpeed;
            cameraParam.distance -= zoomAmount;
            
            cameraParam.distance = std::max(cameraParam.distance, cameraParam.minDistance);
            framebufferReset = true;
        }

    }

    // Update camera parameters
    Camera::ComputeCameraParam(cameraParam);
    // framebufferReset = true;
}


void UI::GuiEnd()
{
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}