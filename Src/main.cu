#define STB_IMAGE_IMPLEMENTATION
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "global.h"
#include "scene.h"
#include "sampler.h"
#include "ui.h"
#include "camera.h"
#include "material.h"
#include "triangle.h"
#include "light.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "renderer.cuh"

int main(int argc, char** argv)
{
    // parse scene file
    if(argc < 2){
        std::cout << "Please provide a scene file path!" << std::endl;
        return -1;
    }
    std::string sceneFilePath = argv[1];
    UI *ui = nullptr;
    Scene *scene = nullptr;
    Renderer *renderer = nullptr;
    try{
        ui = new UI(sceneFilePath);
        scene = new Scene(sceneFilePath);
        renderer = new Renderer(ui);
        renderer -> RenderLoop();
        delete renderer;
        delete scene;
        delete ui;
        return 0;
    }catch(const std::exception& e){
        std::cout << "Fatal error: " << e.what() << std::endl;
        delete renderer;
        delete scene;
        delete ui;
        return -1;
    }
}
