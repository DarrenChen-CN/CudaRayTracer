#version 330 core 
layout(location = 0) out vec4 gPositionTriID; // rgb: position, a: triangleID 32F
layout(location = 1) out uvec4 gNormalMatID; // rgb: normal, a: materialID 16UI
layout(location = 2) out uvec4 gBaryMeshID; // rgb: barycentric coord, a: meshID 16UI
layout(location = 3) out vec4 gMotionDepth; // rg: motion vector, b: linear depth, a: dZ 32F

in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vBarycentric;
in vec4 vCurClipPos;
in vec4 vPrevClipPos;

uniform int uMaterialID;
uniform int uMeshID;
uniform float uNearPlane;
uniform float uFarPlane;

void main(){
    // position + triangleID
    gPositionTriID = vec4(vWorldPos, float(gl_PrimitiveID));

    // normal + materialID
    vec3 normalEncoded = (normalize(vNormal) * 0.5 + 0.5) * 65535.0;
    gNormalMatID = uvec4(uint(normalEncoded.r), uint(normalEncoded.g), uint(normalEncoded.b), uMaterialID);

    // barycentric + meshID
    vec3 baryEncoded = vBarycentric * 65535.0;
    gBaryMeshID = uvec4(uint(baryEncoded.r), uint(baryEncoded.g), uint(baryEncoded.b), uMeshID);
    // gBaryMeshID = uvec4(1, 1, 1, 1); // temp fix for gbuffer rendering issue);

    // motion vector + linear depth + dZ
    vec2 curNDC = vCurClipPos.xy / vCurClipPos.w;
    // curNDC = vec2(int(curNDC.x), int(curNDC.y)); // avoid sub-pixel motion
    vec2 prevNDC = vPrevClipPos.xy / vPrevClipPos.w;
    vec2 motionVector = curNDC - prevNDC;
    float zNDC = gl_FragCoord.z * 2.0 - 1.0;
    float linearDepth = (2.0 * uNearPlane * uFarPlane) / (uFarPlane + uNearPlane - zNDC * (uFarPlane - uNearPlane));
    float dZ = fwidth(linearDepth);
    gMotionDepth = vec4(motionVector, linearDepth, dZ);
    // gMotionDepth = vec4(motionVector, gl_FragCoord.z, dZ);
}