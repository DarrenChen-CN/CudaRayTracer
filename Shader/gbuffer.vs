#version 330 core
layout(location = 0)in vec3 aPos;
layout(location = 1)in vec3 aNormal;
layout(location = 2)in vec2 aTexCoord;
layout(location = 3)in vec3 aBarycentric;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 prevView;
uniform mat4 prevProjection;
// uniform mat4 viewProjection;
// uniform mat4 prevViewProjection;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vBarycentric;
out vec4 vCurClipPos;
out vec4 vPrevClipPos;

void main(){
    vec4 worldPos = model * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;

    vNormal = normalize(mat3(transpose(inverse(model))) * aNormal);

    vBarycentric = aBarycentric;

    // vCurClipPos = viewProjection * worldPos;
    // vPrevClipPos = prevViewProjection * worldPos;
    vCurClipPos = projection * view * worldPos;
    vPrevClipPos = prevProjection * prevView * worldPos;

    gl_Position = vCurClipPos;
    // gl_Position = vec4(vWorldPos.xyz * 0.001, 1.0);
}

