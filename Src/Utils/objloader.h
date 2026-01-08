#pragma once
#include <vector>
#include <iostream>
#include <memory>
#include "global.h"
#include "stringutil.h"
#include "triangle.h"


class OBJLoader
{
public:
    __host__ __device__ static std::vector<Triangle> LoadObject(int id, std::string objpath, Mat4f transform = Mat4f::Identity(), Mat4f transformInv = Mat4f::Identity());
};