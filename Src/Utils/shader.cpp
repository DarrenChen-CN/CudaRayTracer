#include "glad/glad.h"
#include "global.h"
#include "shader.h"
#include "GLFW/glfw3.h"
#include < fstream>
#include <iostream>
Shader::Shader(const char *vertexPath, const char *fragmentPath)
{
    // Get shader source code from files
    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vertexShaderFile;
    std::ifstream fragmentShaderFile;
    // vertexShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    // fragmentShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    if (vertexPath != nullptr && vertexPath[0] != '\0' && fragmentPath != nullptr && fragmentPath[0] != '\0')
    {
        try
        { // Open shader files
            vertexShaderFile.open(vertexPath);
            fragmentShaderFile.open(fragmentPath);
            // Check if files are opened successfully
            if (!vertexShaderFile.is_open())
            {
                std::cout << "Failed to open vertex shader file: " << vertexPath << std::endl;
            }
            if (!fragmentShaderFile.is_open())
            {
                std::cout << "Failed to open fragment shader file: " << fragmentPath << std::endl;
            }
            std::stringstream vertexShaderStream, fragmentShaderStream;
            // Read shader files into string streams
            vertexShaderStream << vertexShaderFile.rdbuf();
            fragmentShaderStream << fragmentShaderFile.rdbuf();
            // Close the files
            vertexShaderFile.close();
            fragmentShaderFile.close();
            // Stream to string conversion
            vertexCode = vertexShaderStream.str();
            fragmentCode = fragmentShaderStream.str();
        }
        catch (std::ifstream::failure &e)
        {
            std::cerr << "ERROR::SHADER::FILE_NOT_READ: " << e.what() << std::endl;
        }
    }
    else
    { // Default shader code if paths are not provided
        // Using default shader code
        vertexCode = "#version 330 core \n"
                     "layout (location = 0) in vec3 aPos;\n"
                     "layout (location = 1) in vec2 aTexCoord;\n"
                     "\n"
                     "out vec2 TexCoord;\n"
                     "\n"
                     "void main()\n"
                     "{\n"
                     "    gl_Position = vec4(aPos, 1.0);\n"
                     "    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);\n"
                     "}\n";
        fragmentCode = "#version 330 core \n"
                       "out vec4 FragColor;\n"
                       "in vec2 TexCoord;\n"
                       "uniform sampler2D texture1;\n"
                       "void main()\n"
                       "{\n"
                       "    FragColor = texture(texture1, TexCoord);\n"
                       "}\n";
    }
    const char *vertexShaderCode = vertexCode.c_str();
    const char *fragmentShaderCode = fragmentCode.c_str();
    // Compile vertex shader
    vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShaderId, 1, &vertexShaderCode, nullptr);
    // LOG_DEBUG("Vertex shader source code:\n{}", vertexCode);
    glCompileShader(vertexShaderId);
    // LOG_DEBUG("Vertex Shader compiled with ID: {}", vertexShaderId);
    // Check for vertex shader compile errors
    int success;
    glGetShaderiv(vertexShaderId, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(vertexShaderId, 512, nullptr, infoLog);
        std::cerr << "Vertex Shader compilation failed: " << infoLog << std::endl;
    }
    // Compile fragment shader
    fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderId, 1, &fragmentShaderCode, nullptr);
    // LOG_DEBUG("Fragment shader source code:\n{}", fragmentCode);
    glCompileShader(fragmentShaderId);
    // LOG_DEBUG("Fragment Shader compiled with ID: {}", fragmentShaderId);
    // Check for fragment shader compile errors
    glGetShaderiv(fragmentShaderId, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShaderId, 512, nullptr, infoLog);
        std::cerr << "Fragment Shader compilation failed: " << infoLog << std::endl;
    }
    // Link shaders into a program
    id = glCreateProgram();
    glAttachShader(id, vertexShaderId);
    glAttachShader(id, fragmentShaderId);
    glLinkProgram(id);
    glValidateProgram(id);
    // LOG_DEBUG("Shader program created with ID: {}", id);
    // Check for linking errors
    glGetProgramiv(id, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(id, 512, nullptr, infoLog);
        std::cerr << "Program linking failed: " << infoLog << std::endl;
    }
}
Shader::~Shader()
{
    // Clean up shaders and program
    glDeleteShader(vertexShaderId);
    glDeleteShader(fragmentShaderId);
    glDeleteProgram(id);
}
void Shader::Use() const
{
    glUseProgram(id);
}
void Shader::Unuse() const
{
    glUseProgram(0);
}
GLint Shader::GetID() const
{
    return id;
}
GLint Shader::GetVertexShaderID() const
{
    return vertexShaderId;
}
GLint Shader::GetFragmentShaderID() const
{
    return fragmentShaderId;
}