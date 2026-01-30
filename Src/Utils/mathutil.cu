#include "mathutil.h"
#include <random>

float AngleToRadian(float angle)
{
    return angle / 180 * PI;
}

float RadianToAngle(float radian)
{
    return radian * 180 / PI;
}

int MaxDimension(const Vec3f &v)
{
    if (v(0) >= v(1) && v(0) >= v(2))
    {
        return 0; // x is the largest
    }
    else if (v(1) >= v(0) && v(1) >= v(2))
    {
        return 1; // y is the largest
    }
    else
    {
        return 2; // z is the largest
    }
}

Vec3f Permute(const Vec3f &v, int dim0, int dim1, int dim2)
{
    Vec3f permuted;
    permuted(0) = v(dim0);
    permuted(1) = v(dim1);
    permuted(2) = v(dim2);
    return permuted;
}

Vec3f Abs(const Vec3f &v)
{
    Vec3f absV;
    absV(0) = std::abs(v(0));
    absV(1) = std::abs(v(1));
    absV(2) = std::abs(v(2));
    return absV;
}

Vec3f LocalToWorld(Vec3f v, Vec3f normal)
{
    Vec3f n = normal.normalized();
    Vec3f up;
    if (fabs(n(0)) > fabs(n(1)))
        up = {0, 1, 0};
    else
        up = {1, 0, 0};
    Vec3f a, b;
    a = up.cross(n).normalized();
    b = n.cross(a).normalized();
    return a * v(0) + b * v(1) + n * v(2);
}

Vec4f Vec3ToVec4(Vec3f v)
{
    return Vec4f(v(0), v(1), v(2), 1.f);
}

Vec3f Vec4ToVec3(Vec4f v)
{
    return Vec3f(v(0), v(1), v(2));
}

float Clamp(float a, float b, float c)
{
    if (c < a)
        return a;
    if (c > b)
        return b;
    return c;
}

Vec3f Min(Vec3f v1, Vec3f v2)
{
    return Vec3f(std::min(v1(0), v2(0)), std::min(v1(1), v2(1)), std::min(v1(2), v2(2)));
}

Vec3f Max(Vec3f v1, Vec3f v2)
{
    return Vec3f(std::max(v1(0), v2(0)), std::max(v1(1), v2(1)), std::max(v1(2), v2(2)));
}

__host__ __device__ Mat4f Translate(const Vec3f &trans){
    Mat4f mat = Mat4f::Identity();
    mat(0, 3) = trans(0);
    mat(1, 3) = trans(1);
    mat(2, 3) = trans(2);
    return mat;
}

__host__ __device__ Mat4f Scale(const Vec3f &scale){
    Mat4f mat = Mat4f::Identity();
    mat(0, 0) = scale(0);
    mat(1, 1) = scale(1);
    mat(2, 2) = scale(2);
    return mat;
}

__host__ __device__ Mat4f RotateX(float angle){
    float rad = AngleToRadian(angle);
    Mat4f mat = Mat4f::Identity();
    mat(1, 1) = cos(rad);
    mat(1, 2) = -sin(rad);
    mat(2, 1) = sin(rad);
    mat(2, 2) = cos(rad);
    return mat;
}
__host__ __device__ Mat4f RotateY(float angle){
    float rad = AngleToRadian(angle);
    Mat4f mat = Mat4f::Identity();
    mat(0, 0) = cos(rad);
    mat(0, 2) = sin(rad);
    mat(2, 0) = -sin(rad);
    mat(2, 2) = cos(rad);
    return mat;
}
__host__ __device__ Mat4f RotateZ(float angle){
    float rad = AngleToRadian(angle);
    Mat4f mat = Mat4f::Identity();
    mat(0, 0) = cos(rad);
    mat(0, 1) = -sin(rad);
    mat(1, 0) = sin(rad);
    mat(1, 1) = cos(rad);
    return mat;
}

__host__ __device__ Vec3f Lerp(const Vec3f &v1, const Vec3f &v2, float t){
    return v1 * (1.0f - t) + v2 * t;
}

__host__ __device__ Vec3f Reflect(const Vec3f &I, const Vec3f &N){
    return I - 2.0f * I.dot(N) * N;
}

__host__ __device__ float Luminance(const Vec3f& color) {
    // 使用 Rec. 709 标准权重
    return 0.2126f * color(0) + 0.7152f * color(1) + 0.0722f * color(2);
}

__device__ void CreateONB(const Vec3f& normal, Vec3f& tangent, Vec3f& bitangent){
    if (fabs(normal(0)) > fabs(normal(1))){
        tangent = Vec3f(-normal(2), 0, normal(0)).normalized();
    } else {
        tangent = Vec3f(0, normal(2), -normal(1)).normalized();
    }
    bitangent = normal.cross(tangent).normalized();
}

__host__ __device__ unsigned int PCGHash(unsigned int input) {
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__host__ __device__ Mat4f Perspective(float fovy, float aspect, float near, float far){
    float f = 1.0f / tan(AngleToRadian(fovy) / 2.0f);
    Mat4f mat = Mat4f::Zero();
    mat(0, 0) = f / aspect;
    mat(1, 1) = f;
    mat(2, 2) = (far + near) / (near - far);
    mat(2, 3) = (2 * far * near) / (near - far);
    mat(3, 2) = -1.0f;
    return mat;
}


__host__ __device__ Mat4f LookAt(const Vec3f &eye, const Vec3f &center, const Vec3f &up){
    Vec3f f = (center - eye).normalized();
    Vec3f s = f.cross(up).normalized();
    Vec3f u = s.cross(f);

    Mat4f mat = Mat4f::Identity();
    mat(0, 0) = s(0);
    mat(0, 1) = s(1);
    mat(0, 2) = s(2);
    mat(1, 0) = u(0);
    mat(1, 1) = u(1);
    mat(1, 2) = u(2);
    mat(2, 0) = -f(0);
    mat(2, 1) = -f(1);
    mat(2, 2) = -f(2);
    mat(0, 3) = -s.dot(eye);
    mat(1, 3) = -u.dot(eye);
    mat(2, 3) = f.dot(eye);
    return mat;
}