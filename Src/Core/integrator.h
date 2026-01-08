// #pragma once
// #include "global.h"
// #include <memory>
// #include "triangle.h"
// #include "mathutil.h"
// #include <thrust/device_vector.h>
// #include "scene.h"
// #include "light.h"    
// enum IntegratorType
// {
//     Path_Tracing
// };

// class Integrator
// {
// public:
//     __host__ __device__ Integrator(int maxDepth = 32, float rr = 0.8) : maxDepth(maxDepth), rr(rr)
//     {
//         spp = 1; // Samples per pixel
//         type = Path_Tracing;
//     }

//     __host__ __device__ virtual ~Integrator()
//     {
//     }
//     __device__ void Render(uchar4 *pixels, int i, int j, int width, int height, TriangleMesh **triangleMeshes, int numTriangleMeshes, Ray ray)
//     {
//         // For testing
//         // int idx = j * width + i;
//         // bool hit = false;
//         // for (int k = 0; k < numTriangleMeshes; ++k)
//         // {
//         //     TriangleMesh *mesh = triangleMeshes[k];
//         //     if (mesh->IsIntersect(ray))
//         //     {
//         //         hit = true;
//         //         // break;
//         //     }
//         // }
//         // if (hit)
//         // {
//         //     Vec3f color = Vec3f(1.0f, 0.0f, 0.0f); // Example color
//         //     pixels[idx] = make_uchar4(color(0) * 255, color(1) * 255, color(2) * 255, 255);
//         // }
//         // else
//         // {
//         //     // LOG_DEBUG("Ray {} does not intersect with triangle mesh", idx);
//         //     pixels[idx] = make_uchar4(0, 0, 0, 255); // Background color
//         // }
//     }

//     __device__ Vec3f Trace(Ray ray, Scene *scene, Sampler *sampler, int idx){
//         auto format = [](Vec3f &a) {
//             for(int i = 0; i < 3; i ++)if(a[i] < 0)a[i] = 0;
//         };

//         Vec3f l = {0, 0, 0};

//         Vec3f weight = {1, 1, 1};
//         Vec3f brdf;

//         // Sampler *sampler = scene -> GetSampler();
//         for(int bounce = 0; bounce < maxDepth; bounce ++){
//             IntersectionInfo info;
//             bool hit = scene -> IsIntersect(ray, info);
//             if(!hit)break;

//             Material *material = info.material;
//             Vec3f hitPoint = info.hitPoint;
//             Vec3f hitNormal = info.normal;
//             Vec2f uv = info.texCoord;
//             Vec3f wo = -ray.direction;

//             if(material -> IsLight()){
//                 // printf("light\n");
//                 // Hit light
//                 if(bounce == 0){
//                     // printf("emission: %f, %f, %f\n", material->e(0), material->e(1), material->e(2));
//                     return material -> e;
//                 }else break;
//             }

//             // Sample light
//             float sampleLightPDF;
//             Light *light = scene -> UniformSampleLight(sampleLightPDF, sampler, idx);
//             float sampleLightPointPDF;
//             TriangleSampleInfo sampleLightPointInfo;
//             light -> SamplePoint(sampleLightPointInfo, sampleLightPointPDF, sampler, idx);
            
//             bool visible = light -> Visible(info, sampleLightPointInfo, scene);
//             if(visible || 1){
//                 Vec3f dirWi = (sampleLightPointInfo.position - hitPoint).normalized();
//                 float r2 = (hitPoint - sampleLightPointInfo.position).squaredNorm();
//                 float cosTheta = hitNormal.dot(dirWi);
//                 float cosTheta2 = sampleLightPointInfo.normal.dot(-dirWi);
//                 float pdfLight = sampleLightPDF * sampleLightPointPDF;
//                 pdfLight = (r2 * pdfLight) / cosTheta2;
            
//                 Vec3f dirBrdf = material -> Evaluate(wo, hitNormal, dirWi);
//                 Vec3f dir_l = light -> Emission().cwiseProduct(dirBrdf) * cosTheta / pdfLight;
//                 format(dir_l);

//                 l += weight.cwiseProduct(dir_l);
//                 // printf("light contribution: %f, %f, %f\n", dir_l(0), dir_l(1), dir_l(2));
//             }
//             // break;

//             // indir
//             Vec3f indirWi;
//             float indirWiPdf;
//             material -> Sample(wo, hitNormal, indirWi, indirWiPdf, sampler, idx);
//             brdf = material -> Evaluate(wo, hitNormal, indirWi);
//             float indirCosTheta = hitNormal.dot(indirWi);
//             // LOG_DEBUG(indirWiPdf);
//             if(indirWiPdf < eps || indirCosTheta < 0)break;
//             weight = weight.cwiseProduct(brdf) * indirCosTheta / indirWiPdf;
//             format(weight);
//             // else break;

//             // rr
//             float r = sampler -> Get1D(idx);
//             if(r > rr)break;
//             weight /= rr;

//             ray = {hitPoint + 1e-4 * hitNormal, indirWi};
//         }

//         return l;
//     }

//     __device__ void Render(uchar4 *pixels, float* accumulateColors, int i, int j, int width, int height, Scene *scene, Ray ray, Sampler *sampler)
//     {
//         // For testing
//         int idx = (height - j - 1) * width + i;
//         // IntersectionInfo info;
//         // if (scene->IsIntersect(ray, info))
//         // {
//         //     // Vec3f color = Vec3f(1.0f, 0.0f, 0.0f); // Example color
//         //     Vec3f color = info.material->kd; // Example color
//         //     // printf("material kd: %f, %f, %f\n", info.material->kd(0), info.material->kd(1), info.material->kd(2));
//         //     pixels[idx] = make_uchar4(color(0) * 255, color(1) * 255, color(2) * 255, 255);
//         // }
//         // else
//         // {
//         //     pixels[idx] = make_uchar4(0, 0, 0, 255); // Background color
//         // }
//         Vec3f color = Trace(ray, scene, sampler, idx);
//         // printf("pixel (%d, %d), color: %f, %f, %f\n", i, j, color(0), color(1), color(2));

//         accumulateColors[4 * idx + 0] += color(0);
//         accumulateColors[4 * idx + 1] += color(1);
//         accumulateColors[4 * idx + 2] += color(2);

//         // printf("accumulateColors: %f, %f, %f\n", accumulateColors[4 * idx + 0], accumulateColors[4 * idx + 1], accumulateColors[4 * idx + 2]);

//         // Update Pixels
//         pixels[idx] = make_uchar4(
//             accumulateColors[4 * idx + 0] / spp * 255,
//             accumulateColors[4 * idx + 1] / spp * 255,
//             accumulateColors[4 * idx + 2] / spp * 255,
//             255
//         );

//         spp++;

//     }
//     __device__ int GetSPP()
//     {
//         return spp;
//     }
//     __device__ IntegratorType GetType()
//     {
//         return type;
//     }

// protected:
//     int spp;
//     int maxDepth = 32;
//     float rr = 0.8;
//     IntegratorType type;
// };