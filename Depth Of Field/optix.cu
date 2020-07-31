
#include <optix.h>
#include "random.h"
#include "LaunchParams7.h" // our launch params
#include <vec_math.h> // NVIDIAs math utils

#define 	M_PI_2f   1.57079632679489661923f
#define 	M_PI_4f   0.78539816339744830962f

extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}
//  a single ray type
enum { PHONG=0, SHADOW, RAY_TYPE_COUNT };

struct colorPRD{
    float3 color;
    unsigned int seed;
} ;

struct shadowPRD{
    float shadowAtt;
    unsigned int seed;
} ;

// -------------------------------------------------------
// closest hit computes color based lolely on the triangle normal
// -------------------------------------------------------
static __device__ __inline__ float2 concentric_sampling(const float2& uOffset)
{

	// Handle degeneracy at origin
	if (uOffset.x == 0 && uOffset.y == 0)
		return make_float2(0.f, 0.f);

	// Apply concentric mapping to point
	float theta;
	float r;
	if(abs(uOffset.x) > abs(uOffset.y))
	{
		r = uOffset.x;
		theta = M_PI_4f * (uOffset.y / uOffset.x);
	}
	else
	{
		r = uOffset.y;
		theta = M_PI_2f - M_PI_4f * (uOffset.x / uOffset.y);
	}


	return r * make_float2(cos(theta), sin(theta));
}

static __device__ __inline__ float2 disc_sampling(const float2& u,
	const float maxwidth,
	const float maxheight,
	const uchar2& index)
{

	float w = (-(1 / 2) + ((index.x + 0.5) / maxwidth));
		
	float h = ((1 / 2) - ((index.y + 0.5) / maxheight));

	// a) Map uniform random number to [-1,1]^2
	float2 uOffset = 2.f * u - make_float2(1.f, 1.f);

	float2 image_point = make_float2(w, h);

	image_point += uOffset;

	return concentric_sampling(image_point);
}

extern "C" __global__ void __closesthit__radiance() {

    float3 &prd = *(float3*)getPRD<float3>();

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];

    // intersection position
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();

    // direction towards light
    float3 lPos = make_float3(optixLaunchParams.global->lightPos);
    float lDirLength = length(lPos - pos) - 0.01f;
    float3 lDir = normalize(lPos - pos);
    float3 nn = normalize(make_float3(n));

    float intensity = max(dot(lDir, nn),0.0f);

    // ray payload
    float shadowAttPRD = 1.0f;
    uint32_t u0, u1;
    packPointer( &shadowAttPRD, u0, u1 );  
  
    // trace shadow ray
    optixTrace(optixLaunchParams.traversable,
        pos,
        lDir,
        0.001f,         // tmin
        lDirLength,     // tmax
        0.0f,           // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        SHADOW,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        SHADOW,             // missSBTIndex 
        u0, u1 );

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {  
        // get barycentric coordinates
        // compute pixel texture coordinate
        const float4 tc
          = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
          +         u * sbtData.vertexD.texCoord0[index.y]
          +         v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        prd = make_float3(fromTexture) * min(intensity * shadowAttPRD + 0.0, 1.0);
    }
    else
        prd = sbtData.color * min(intensity * shadowAttPRD + 0.0, 1.0);
}


// any hit to ignore intersections with back facing geometry
extern "C" __global__ void __anyhit__radiance() {

}

// miss sets the background color
extern "C" __global__ void __miss__radiance() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(0.0f, 0.0f, 1.0f);
}

// -------------
// Shadow rays
// -------------

extern "C" __global__ void __closesthit__shadow() {

    float &prd = *(float*)getPRD<float>();
    prd = 0.0f;
}

// any hit for shadows
extern "C" __global__ void __anyhit__shadow() {

}

// miss for shadows
extern "C" __global__ void __miss__shadow() {

    float &prd = *(float*)getPRD<float>();
    // set blue as background color
    prd = 1.0f;
}

// -----------------------------------------------
// Light material


extern "C" __global__ void __closesthit__light() {

    float3 &prd = *(float3*)getPRD<float3>();
    prd = make_float3(1.0f, 1.0f, 1.0f);
}


extern "C" __global__ void __anyhit__light() {
}


extern "C" __global__ void __miss__light() {
}


extern "C" __global__ void __closesthit__light_shadow() {

    float &prd = *(float*)getPRD<float>();
    prd = 1.0f;
}


// any hit to ignore intersections based on alpha transparency
extern "C" __global__ void __anyhit__light_shadow() {
}


// miss sets the background color
extern "C" __global__ void __miss__light_shadow() {
}




// -----------------------------------------------
// Metal Phong rays

extern "C" __global__ void __closesthit__phong_metal() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];
    // ray payload

    float3 normal = normalize(make_float3(n));

    // entering glass
    //if (dot(optixGetWorldRayDirection(), normal) < 0)

    float3 afterPRD = make_float3(1.0f);
    uint32_t u0, u1;
    packPointer( &afterPRD, u0, u1 );  

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    //(1.f-u-v) * A + u * B + v * C;
    
    float3 rayDir = reflect(optixGetWorldRayDirection(), normal);
    optixTrace(optixLaunchParams.traversable,
        pos,
        rayDir,
        0.04f,    // tmin is high to void self-intersection
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        PHONG,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        PHONG,             // missSBTIndex 
        u0, u1 );

    float3 &prd = *(float3*)getPRD<float3>();
    prd = make_float3(0.8,0.8,0.8) * afterPRD;
}





// -----------------------------------------------
// Glass Phong rays

SUTIL_INLINE SUTIL_HOSTDEVICE float3 refract(const float3& i, const float3& n, const float eta) {

    float k = 1.0 - eta * eta * (1.0 - dot(n, i) * dot(n, i));
    if (k < 0.0)
        return make_float3(0.0f);
    else
        return (eta * i - (eta * dot(n, i) + sqrt(k)) * n);
}


extern "C" __global__ void __closesthit__phong_glass() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];

    float3 normal = normalize(make_float3(n));
    const float3 normRayDir = optixGetWorldRayDirection();

    // new ray direction
    float3 rayDir;
    // entering glass
    float dotP;
    if (dot(normRayDir, normal) < 0) {
        dotP = dot(normRayDir, -normal);
        rayDir = refract(normRayDir, normal, 0.66);
    }
    // exiting glass
    else {
        dotP = 0;
        rayDir = refract(normRayDir, -normal, 1.5);
    }

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    
    float3 refractPRD = make_float3(0.0f);
    uint32_t u0, u1;
    packPointer( &refractPRD, u0, u1 );  
    
    if (length(rayDir) > 0)
        optixTrace(optixLaunchParams.traversable,
            pos,
            rayDir,
            0.00001f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
            PHONG,             // SBT offset
            RAY_TYPE_COUNT,     // SBT stride
            PHONG,             // missSBTIndex 
            u0, u1 );

    // ray payload 
    float3 &prd = *(float3*)getPRD<float3>();
 
    float3 reflectPRD = make_float3(0.0f);
    if (dotP > 0) {
        float3 reflectDir = reflect(normRayDir, normal);        
        packPointer( &reflectPRD, u0, u1 );  
        optixTrace(optixLaunchParams.traversable,
            pos,
            reflectDir,
            0.00001f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
            PHONG,             // SBT offset
            RAY_TYPE_COUNT,     // SBT stride
            PHONG,             // missSBTIndex 
            u0, u1 );
        float r0 = (1.5f - 1.0f)/(1.5f + 1.0f);
        r0 = r0*r0 + (1-r0*r0) * pow(1-dotP,5);
        prd =  refractPRD * (1-r0) + r0*reflectPRD;
    }
    else
        prd =  refractPRD ;
}



extern "C" __global__ void __anyhit__phong_glass() {

}


// miss sets the background color
extern "C" __global__ void __miss__phong_glass() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(0.0f, 0.0f, 1.0f);
}



// -----------------------------------------------
// Glass Shadow rays

extern "C" __global__ void __closesthit__shadow_glass() {

    // ray payload
    float afterPRD = 1.0f;
    uint32_t u0, u1;
    packPointer( &afterPRD, u0, u1 );  

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    
    // trace primary ray
    optixTrace(optixLaunchParams.traversable,
        pos,
        optixGetWorldRayDirection(),
        0.001f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        SHADOW,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        SHADOW,             // missSBTIndex 
        u0, u1 );

    float &prd = *(float*)getPRD<float>();
    prd = 0.95f * afterPRD;
}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow_glass() {

}


// miss for shadows
extern "C" __global__ void __miss__shadow_glass() {

    float &prd = *(float*)getPRD<float>();
    // set blue as background color
    prd = 1.0f;
}

// --------------
// Primary Rays
// --------------

extern "C" __global__ void __raygen__renderFrame() {

    //-- compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;  

    //-- ray payload// ray payload
    colorPRD pixelColorPRD;
    pixelColorPRD.color = make_float3(1.f);

    //-- ray's direction computation
    const float2 screen(make_float2(ix+.5f,iy+.5f)/ make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);

    float3 ray_dir = normalize(camera.direction
       + screen.x  * camera.horizontal
       + screen.y * camera.vertical);

    //-- N ray samples
    float raysPerPixel = float(optixLaunchParams.frame.raysPerPixel);
    
    //-- color values (RGB)
    float red = 0.0f, green = 0.0f, blue = 0.0f;
     
    //-- Lens values calculation 
    float aperture = optixLaunchParams.global->aperture;
    float focal_length = optixLaunchParams.global->focalDistance;

    //-- if 'aperture == 0' treat camera as a 'pinhole camera'
    if(aperture == 0)
	{   
        
        uint32_t u0, u1;
        packPointer( &pixelColorPRD, u0, u1 );

           // trace primary ray
        optixTrace(optixLaunchParams.traversable,
            camera.position + camera.direction * optixLaunchParams.global->lensDistance,
            ray_dir,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_NONE,//,OPTIX_RAY_FLAG_DISABLE_ANYHIT
            PHONG,             // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            PHONG,             // missSBTIndex 
            u0, u1 );

            red = pixelColorPRD.color.x ;
            green = pixelColorPRD.color.y ;
            blue = pixelColorPRD.color.z ;
    }
    else{
        //-- Step 1 -> Calculate distance to the 'Focus Plane'
		float ft = focal_length / dot(ray_dir,camera.direction);
        float3 focus_plane = camera.position + ray_dir * ft;
        
        for (int i = 0; i < raysPerPixel; ++i) {
            for (int j = 0; j < raysPerPixel; ++j) {
    
                uint32_t seed = tea<4>( ix * optixGetLaunchDimensions().x + iy, i*raysPerPixel + j );
                
                //-- Step 2 -> Calculate sample point on lens (using Concentric Disk Sampling)
                float2 p_sample = make_float2(rnd(seed), rnd(seed));
				float2 disc_sample = disc_sampling(p_sample,raysPerPixel,raysPerPixel,make_uchar2(i,j));
                float2 lens_point = aperture * disc_sample;
                
                //-- Step 3 -> Compute point 'P' on 'Focus Plane'
				float3 ray_origin = camera.position + lens_point.x * camera.horizontal + lens_point.y * camera.vertical;
                float3 ray_direction = normalize(focus_plane - ray_origin);
                
                pixelColorPRD.seed = seed;
                uint32_t u0, u1;
                packPointer( &pixelColorPRD, u0, u1 );
                
                //-- Step 4 -> Trace ray's emitted from point 'P'            
                optixTrace(optixLaunchParams.traversable,
                        ray_origin + camera.direction * optixLaunchParams.global->lensDistance,
                        ray_direction,
                        0.f,    // tmin
                        1e20f,  // tmax
                        0.0f,   // rayTime
                        OptixVisibilityMask( 255 ),
                        OPTIX_RAY_FLAG_NONE,//,OPTIX_RAY_FLAG_DISABLE_ANYHIT
                        PHONG,             // SBT offset
                        RAY_TYPE_COUNT,               // SBT stride
                        PHONG,             // missSBTIndex 
                        u0, u1 );
    
                red += pixelColorPRD.color.x / (raysPerPixel*raysPerPixel);
                green += pixelColorPRD.color.y / (raysPerPixel*raysPerPixel);
                blue += pixelColorPRD.color.z / (raysPerPixel*raysPerPixel);
            }
        }
    }

    //convert float (0-1) to int (0-255)
    const int r = int(255.0f*red);
    const int g = int(255.0f*green);
    const int b = int(255.0f*blue);
    // convert to 32-bit rgba value 
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);
    // compute index
    const uint32_t fbIndex = ix + iy*optixGetLaunchDimensions().x;
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
  

