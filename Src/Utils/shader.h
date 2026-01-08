#pragma once
class Shader
{
public:
    Shader(const char *vertexPath = "", const char *fragmentPath = "");
    ~Shader();
    void Use() const;
    void Unuse() const;
    GLint GetID() const;
    GLint GetVertexShaderID() const;
    GLint GetFragmentShaderID() const;

private:
    GLint id;               // Shader program ID
    GLint vertexShaderId;   // Vertex shader ID
    GLint fragmentShaderId; // Fragment shader ID
};