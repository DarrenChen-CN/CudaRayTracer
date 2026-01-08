#include <ui.h>
UI::UI(int width, int height) : width(width), height(height)
{
    accumulateColors = new float[width * height * 3](); // Initialize to zero
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
void UI::GuiBegin(int spp)
{
    // Start a new ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("CPURayTracer GUI", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    ImGui::SetWindowPos(ImVec2(0, 0));
    ImGui::SetWindowSize(ImVec2(350, 100));
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("spp %d", spp);
    ImGui::End();
}

void UI::GuiEnd()
{
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}