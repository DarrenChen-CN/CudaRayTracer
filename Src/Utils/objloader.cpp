#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include "objloader.h"
#include "mathutil.h"
#include <unordered_map>
#include <iostream>

namespace {

Vertex BuildVertex(
    const tinyobj::attrib_t &attrib,
    const tinyobj::index_t &index,
    const Mat4f &transform,
    const Mat4f &transformInv,
    const std::vector<Vec3f> &generatedNormals,
    const Vec3f &barycentric)
{
    Vertex vertex;
    vertex.barycentricCoords = barycentric;
    vertex.texCoord = Vec2f(0.0f, 0.0f);
    vertex.normal = Vec3f(0.0f, 0.0f, 0.0f);

    if (index.vertex_index >= 0) {
        vertex.position = Vec3f(
            attrib.vertices[3 * index.vertex_index + 0],
            attrib.vertices[3 * index.vertex_index + 1],
            attrib.vertices[3 * index.vertex_index + 2]);
        vertex.position = Vec4ToVec3(transform * Vec3ToVec4(vertex.position));
    }

    if (index.normal_index >= 0) {
        Vec3f sourceNormal(
            attrib.normals[3 * index.normal_index + 0],
            attrib.normals[3 * index.normal_index + 1],
            attrib.normals[3 * index.normal_index + 2]);
        vertex.normal = Vec4ToVec3(transformInv.transpose() * Vec3ToVec4(sourceNormal)).normalized();
    } else if (index.vertex_index >= 0 && index.vertex_index < static_cast<int>(generatedNormals.size())) {
        Vec3f generatedNormal = generatedNormals[index.vertex_index];
        if (generatedNormal.squaredNorm() > 0.0f) {
            vertex.normal = Vec4ToVec3(transformInv.transpose() * Vec3ToVec4(generatedNormal.normalized())).normalized();
        }
    }

    if (index.texcoord_index >= 0) {
        vertex.texCoord = Vec2f(
            attrib.texcoords[2 * index.texcoord_index + 0],
            attrib.texcoords[2 * index.texcoord_index + 1]);
    }

    return vertex;
}

std::string BuildMeshName(const std::string &path, const tinyobj::shape_t &shape, int shapeIndex, const std::string &materialName)
{
    std::string shapeName = shape.name.empty()
        ? GetFilenameFromPath(path) + "_shape_" + std::to_string(shapeIndex)
        : shape.name;
    if (!materialName.empty()) {
        return shapeName + "::" + materialName;
    }
    return shapeName;
}

} // namespace

std::vector<LoadedOBJMesh> OBJLoader::LoadObject(std::string path, Mat4f transform, Mat4f transformInv)
{
    std::vector<LoadedOBJMesh> loadedMeshes;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string mtlBasePath = GetDirectoryFromPath(path);
    std::string warn, err;

    std::cout << "Load OBJ File: " << path << std::endl;

    bool ret = tinyobj::LoadObj(
        &attrib,
        &shapes,
        &materials,
        &warn,
        &err,
        path.c_str(),
        mtlBasePath.c_str(),
        true);

    if (!warn.empty()) {
        std::cout << "OBJ Warning: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cout << "OBJ Error: " << err << std::endl;
    }
    if (!ret) {
        std::cout << "Failed to load OBJ file: " << path << std::endl;
        return loadedMeshes;
    }

    if (shapes.empty()) {
        std::cout << "OBJ file has no shapes: " << path << std::endl;
        return loadedMeshes;
    }

    // Generate smooth normals in object space when the OBJ does not provide them.
    std::vector<Vec3f> generatedNormals(attrib.vertices.size() / 3, Vec3f::Zero());
    for (const tinyobj::shape_t &shape : shapes) {
        size_t faceIndexOffset = 0;
        for (int faceIndex = 0; faceIndex < static_cast<int>(shape.mesh.num_face_vertices.size()); ++faceIndex) {
            int faceVertexCount = shape.mesh.num_face_vertices[faceIndex];
            if (faceVertexCount != 3) {
                faceIndexOffset += faceVertexCount;
                continue;
            }

            tinyobj::index_t i0 = shape.mesh.indices[faceIndexOffset + 0];
            tinyobj::index_t i1 = shape.mesh.indices[faceIndexOffset + 1];
            tinyobj::index_t i2 = shape.mesh.indices[faceIndexOffset + 2];
            faceIndexOffset += faceVertexCount;

            if (i0.vertex_index < 0 || i1.vertex_index < 0 || i2.vertex_index < 0) {
                continue;
            }

            Vec3f p0(
                attrib.vertices[3 * i0.vertex_index + 0],
                attrib.vertices[3 * i0.vertex_index + 1],
                attrib.vertices[3 * i0.vertex_index + 2]);
            Vec3f p1(
                attrib.vertices[3 * i1.vertex_index + 0],
                attrib.vertices[3 * i1.vertex_index + 1],
                attrib.vertices[3 * i1.vertex_index + 2]);
            Vec3f p2(
                attrib.vertices[3 * i2.vertex_index + 0],
                attrib.vertices[3 * i2.vertex_index + 1],
                attrib.vertices[3 * i2.vertex_index + 2]);
            Vec3f faceNormal = (p1 - p0).cross(p2 - p0);
            if (faceNormal.squaredNorm() <= 0.0f) {
                continue;
            }

            generatedNormals[i0.vertex_index] += faceNormal;
            generatedNormals[i1.vertex_index] += faceNormal;
            generatedNormals[i2.vertex_index] += faceNormal;
        }
    }

    std::unordered_map<std::string, size_t> meshLookup;

    for (int shapeIndex = 0; shapeIndex < static_cast<int>(shapes.size()); ++shapeIndex) {
        const tinyobj::shape_t &shape = shapes[shapeIndex];
        size_t indexOffset = 0;

        for (int faceIndex = 0; faceIndex < static_cast<int>(shape.mesh.num_face_vertices.size()); ++faceIndex) {
            int faceVertexCount = shape.mesh.num_face_vertices[faceIndex];
            if (faceVertexCount != 3) {
                indexOffset += faceVertexCount;
                continue;
            }

            int materialIndex = -1;
            if (faceIndex < static_cast<int>(shape.mesh.material_ids.size())) {
                materialIndex = shape.mesh.material_ids[faceIndex];
            }

            std::string materialName;
            if (materialIndex >= 0 && materialIndex < static_cast<int>(materials.size())) {
                materialName = materials[materialIndex].name;
            }

            std::string meshKey = std::to_string(shapeIndex) + ":" + std::to_string(materialIndex);
            size_t meshIndex = 0;
            auto found = meshLookup.find(meshKey);
            if (found == meshLookup.end()) {
                LoadedOBJMesh mesh;
                mesh.name = BuildMeshName(path, shape, shapeIndex, materialName);
                mesh.materialName = materialName;
                meshIndex = loadedMeshes.size();
                meshLookup[meshKey] = meshIndex;
                loadedMeshes.push_back(mesh);
            } else {
                meshIndex = found->second;
            }

            tinyobj::index_t i0 = shape.mesh.indices[indexOffset + 0];
            tinyobj::index_t i1 = shape.mesh.indices[indexOffset + 1];
            tinyobj::index_t i2 = shape.mesh.indices[indexOffset + 2];
            indexOffset += faceVertexCount;

            Vertex v0 = BuildVertex(attrib, i0, transform, transformInv, generatedNormals, Vec3f(1.0f, 0.0f, 0.0f));
            Vertex v1 = BuildVertex(attrib, i1, transform, transformInv, generatedNormals, Vec3f(0.0f, 1.0f, 0.0f));
            Vertex v2 = BuildVertex(attrib, i2, transform, transformInv, generatedNormals, Vec3f(0.0f, 0.0f, 1.0f));

            bool hasNormals = i0.normal_index >= 0 && i1.normal_index >= 0 && i2.normal_index >= 0;
            if (!hasNormals) {
                Vec3f faceNormal = (v1.position - v0.position).cross(v2.position - v0.position).normalized();
                v0.normal = faceNormal;
                v1.normal = faceNormal;
                v2.normal = faceNormal;
            }

            Triangle triangle(v0, v1, v2, -1);
            LoadedOBJMesh &mesh = loadedMeshes[meshIndex];
            mesh.vertices.push_back(v0);
            mesh.vertices.push_back(v1);
            mesh.vertices.push_back(v2);
            mesh.triangles.push_back(triangle);
        }
    }

    return loadedMeshes;
}
