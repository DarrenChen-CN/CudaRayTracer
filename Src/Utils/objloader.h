#pragma once
#include <vector>
#include <iostream>
#include <memory>
#include "global.h"
#include "stringutil.h"
#include "triangle.h"

struct LoadedOBJMesh
{
    std::string name;
    std::string materialName;
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
};

class OBJLoader
{
public:
    __host__ static std::vector<LoadedOBJMesh> LoadObject(
        std::string objpath,
        Mat4f transform = Mat4f::Identity(),
        Mat4f transformInv = Mat4f::Identity());
};
