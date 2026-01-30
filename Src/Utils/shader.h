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
    void SetUniformMat4(const char* name, const float* matrix) const;
    void SetUniformInt(const char* name, int value) const;
    void SetUniformFloat(const char* name, float value) const;
    

private:
    GLint id;               // Shader program ID
    GLint vertexShaderId;   // Vertex shader ID
    GLint fragmentShaderId; // Fragment shader ID
};