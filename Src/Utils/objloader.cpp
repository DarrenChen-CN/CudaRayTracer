#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include "objloader.h"
#include "mathutil.h"
#include <unordered_map>
#include <iostream>
__host__ __device__ std::vector<Triangle> OBJLoader::LoadObject(int id, std::string path, Mat4f transform, Mat4f transformInv)
{
    std::vector<Triangle> triangles;
    tinyobj::attrib_t attrib;                              // Storage for attributes (vertices, normals, etc.)
    std::vector<tinyobj::shape_t> shapes;                  // Storage for shapes (meshes)
    std::vector<tinyobj::material_t> materials;            // Storage for materials, not used in this project
    std::string mtl_basepath = GetDirectoryFromPath(path); // .mtl file base path, not used in this project
    std::string warn, err;                                 // Storage for warnings and errors

    std::cout << "Load OBJ File: " << path << std::endl;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str(), mtl_basepath.c_str());
    if (!warn.empty())
    {
        std::cout << "OBJ Warning: " << warn << std::endl;
    }
    if (!err.empty())
    {
        std::cout << "OBJ Error: " << err << std::endl;
    }
    if (!ret)
    {
        std::cout << "Failed to load OBJ file: " << path << std::endl;
        return triangles;
    }

    if(shapes.size() > 1)
    {
        std::cout << "OBJ file has " << shapes.size() << " shapes, only support 1 shape" << std::endl;
        return triangles;
    }

    // std::cout << "Number of vertex: " << attrib.vertices.size() / 3 << std::endl;
    // std::cout << "Number of triangle: " << shapes[0].mesh.indices.size() / 3 << std::endl;

    // Vertices
    std::vector<Vertex> vertices_;
    for (int i = 0; i < attrib.vertices.size() / 3; i++)
    {
        // LOG_DEBUG("Loading vertex {}: ({}, {}, {})", i, attrib.vertices[3 * i + 0], attrib.vertices[3 * i + 1], attrib.vertices[3 * i + 2]);
        Vertex v;
        // auto vertex = attrib.vertices[i];

        v.position = { attrib.vertices[3 * i + 0], attrib.vertices[3 * i + 1], attrib.vertices[3 * i + 2] };
        Vec4f pos = Vec3ToVec4(v.position);
        Vec4f transformedPos = transform * pos;
        v.position = Vec4ToVec3(transformedPos);
        if (attrib.normals.size() > 0)
        {
            v.normal = { attrib.normals[3 * i + 0], attrib.normals[3 * i + 1], attrib.normals[3 * i + 2] };
            Vec4f normal = Vec3ToVec4(v.normal);
            Vec4f transformedNormal = transformInv.transpose() * normal;
            v.normal = Vec4ToVec3(transformedNormal);
        };
        if (attrib.texcoords.size() > 0)
            v.texCoord = { attrib.texcoords[2 * i + 0], attrib.texcoords[2 * i + 1] };
        if(i % 3==0) {
            v.barycentricCoords = Vec3f(1.0f, 0.0f, 0.0f);
        } else if(i %3==1) {
            v.barycentricCoords = Vec3f(0.0f, 1.0f, 0.0f);
        } else {
            v.barycentricCoords = Vec3f(0.0f, 0.0f, 1.0f);
        }
        vertices_.push_back(v);
    }

    // Meshes
    std::vector<unsigned int> indices_;
    std::vector<unsigned int> materials_id_;

    // Indices
    for (int j = 0; j < shapes[0].mesh.indices.size(); j++)
    {
        indices_.push_back((unsigned int)(shapes[0].mesh.indices[j].vertex_index));
    }

    // Create triangle
    for (int j = 0; j < shapes[0].mesh.indices.size() / 3; j++)
    {
        triangles.push_back(Triangle(
            vertices_[indices_[3 * j + 0]],
            vertices_[indices_[3 * j + 1]],
            vertices_[indices_[3 * j + 2]],
            id));
    }
    
    return triangles;
}