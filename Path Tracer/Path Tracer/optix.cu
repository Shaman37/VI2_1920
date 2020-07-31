#include "optixParams.h" // our launch params


extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}

// ray types
enum { RAIDANCE=0, SHADOW, RAY_TYPE_COUNT };

struct RadiancePRD{
    float3   emitted;
    float3   radiance;
    float3   attenuation;
    float3   origin;
    float3   direction;
    bool done;
    uint32_t seed;
    int32_t  countEmitted;
} ;

struct shadowPRD{
    float shadowAtt;
    uint32_t seed;
} ;

// -------------------------------------------------------

extern "C" __global__ void __closesthit__radiance() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    RadiancePRD &prd = *(RadiancePRD *)getPRD<RadiancePRD>();

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

    const float3 nn = normalize(make_float3(n));

    // intersection position
    const float3 &rayDir =  optixGetWorldRayDirection();
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir ;


    if (prd.countEmitted && length(sbtData.emission) != 0) {
        prd.emitted = sbtData.emission ;
        return;
    }
    else
        prd.emitted = make_float3(0.0f);

    uint32_t seed = prd.seed;

    float3 color = sbtData.diffuse;
    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {  
        // get barycentric coordinates
        // compute pixel texture coordinate
        const float4 tc
          = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
          +         u * sbtData.vertexD.texCoord0[index.y]
          +         v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        color = make_float3(fromTexture);
    }

    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        cosine_sample_hemisphere( z1, z2, w_in );
        Onb onb( nn );
        onb.inverse_transform( w_in );
        prd.direction = w_in;
        prd.origin    = pos;
        prd.attenuation *= color ;
        prd.countEmitted = false;
    }
    
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd.seed = seed;

    const float3 lightV1 = make_float3(0.47f, 0.0f, 0.0f);
    const float3 lightV2 = make_float3(0.0f, 0.0f, 0.38f);
    const float3 light_pos = make_float3(optixLaunchParams.global->lightPos) + lightV1 * z1 + lightV2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - pos );
    const float3 L     = normalize(light_pos - pos );
    const float  nDl   = dot( nn, L );
    const float3 Ln    = normalize(cross(lightV1, lightV2));
    const float  LnDl  = -dot( Ln, L );

    float weight = 0.0f;
    if( nDl > 0.0f && LnDl > 0.0f )
    {
        uint32_t occluded = 0u;
        optixTrace(optixLaunchParams.traversable,
            pos,
            L,
            0.001f,         // tmin
            Ldist - 0.01f,  // tmax
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            SHADOW,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            SHADOW,      // missSBTIndex
            occluded);

        if( !occluded )
        {
            const float att = Ldist * Ldist;
            const float A = length(cross(lightV1, lightV2));
            weight = nDl * LnDl * A  / att;
        }
    }

    prd.radiance += make_float3(5.0f, 5.0f, 5.0f) * weight * optixLaunchParams.global->lightScale;
}


extern "C" __global__ void __anyhit__radiance() {

}


// miss sets the background color
extern "C" __global__ void __miss__radiance() {

    RadiancePRD &prd = *(RadiancePRD*)getPRD<RadiancePRD>();
    // set black as background color
    prd.radiance = make_float3(0.0f, 0.0f, 0.0f);
    prd.done = true;
}


// -----------------------------------------------
// Shadow rays

extern "C" __global__ void __closesthit__shadow() {

    optixSetPayload_0( static_cast<uint32_t>(true));
}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow() {

}


// miss for shadows
extern "C" __global__ void __miss__shadow() {

    optixSetPayload_0( static_cast<uint32_t>(false));
}

// -----------------------------------------------
// Metal Rays

extern "C" __global__ void __closesthit__metal() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    RadiancePRD &prd = *(RadiancePRD *)getPRD<RadiancePRD>();

    // Primitive ID and index
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // Barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // Normal computation
    const float4 n
        = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];

    const float3 normal = normalize(make_float3(n));

    // Intersection position
    const float3 &rayDir =  optixGetWorldRayDirection();
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir ;
    
    // Reflect Ray Direction
    float3 reflected = reflect(rayDir, normal);

    // PRD tidy up
    prd.origin = pos;
    prd.direction = reflected;
    prd.attenuation *= sbtData.diffuse;
}

// -----------------------------------------------
// Light material


extern "C" __global__ void __closesthit__light() {
    RadiancePRD &prd = *(RadiancePRD*)getPRD<RadiancePRD>();
    prd.radiance = make_float3(1.0f, 1.0f, 1.0f);
    prd.done = true;
}


extern "C" __global__ void __anyhit__light() {
}


extern "C" __global__ void __miss__light() {
}


extern "C" __global__ void __closesthit__light_shadow() {
    optixSetPayload_0( static_cast<uint32_t>(false));
}

// any hit to ignore intersections based on alpha transparency
extern "C" __global__ void __anyhit__light_shadow() {
}


// miss sets the background color
extern "C" __global__ void __miss__light_shadow() {
    optixSetPayload_0( static_cast<uint32_t>(true));
}


// -----------------------------------------------
// Glass Rays
extern "C" __global__ void __closesthit__glass() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    RadiancePRD &prd = *(RadiancePRD *)getPRD<RadiancePRD>();

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

    const float3 rayDir = optixGetWorldRayDirection();
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;
    
    
    const float3 nl = dot(normal, rayDir) < 0 ? normal : normal * -1; // Right facing Normal calculation

    bool into = dot(normal, nl) > 0; // is ray entering or leaving refractive material?

    float nc = 1.0f;  // Index of Refraction air
    float nt = 1.5f;  // Index of Refraction glass/water

    float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
    float ddn = dot(rayDir, nl);
    float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

    if (cos2t < 0.0f) // total internal reflection 
    {
        prd.direction = reflect(rayDir, normal); //d = rayDir - 2.0f * normal * dot(normal, rayDir);
        prd.origin = pos;
    }
    else // cos2t > 0
    {
        // compute direction of Refract ray
        float3 tdir = normalize(rayDir * nnt - normal * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))));

        float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc); 
        float c = 1.f - (into ? -ddn : dot(tdir, normal));
        float Re = R0 + (1.f - R0) * c * c * c * c * c;
        
        float Tr = 1 - Re; // Refract
        float P = .25f + .5f * Re;

        float RP = Re / P; // Reflection color mask
        float TP = Tr / (1.f - P); // Refraction color mask

        // randomly choose reflection or Refract ray
        if (rnd(prd.seed) < 0.15) // reflection ray
        {
            prd.attenuation *= RP;
            prd.direction = reflect(rayDir, normal);
            prd.origin = pos;
        }
        else // Refract ray
        {
            prd.attenuation *= TP;
            prd.direction = tdir;  
            prd.origin = pos;
        }
    }
}

// -----------------------------------------------
// Glass Shadow rays

extern "C" __global__ void __closesthit__shadow_glass() {
    optixSetPayload_0( static_cast<uint32_t>(true));
}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow_glass() {

}


// miss for shadows
extern "C" __global__ void __miss__shadow_glass() {
    optixSetPayload_0( static_cast<uint32_t>(false));
}


// -----------------------------------------------
// Primary Rays


extern "C" __global__ void __raygen__renderFrame() {

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;  

    const int &maxDepth = optixLaunchParams.frame.maxDepth;
 
    float squaredRaysPerPixel = float(optixLaunchParams.frame.raysPerPixel);
    float2 delta = make_float2(1.0f/squaredRaysPerPixel, 1.0f/squaredRaysPerPixel);

    float3 result = make_float3(0.0f);

    uint32_t seed = tea<4>( ix * optixGetLaunchDimensions().x + iy, optixLaunchParams.frame.frame );

    for (int i = 0; i < squaredRaysPerPixel; ++i) {
        for (int j = 0; j < squaredRaysPerPixel; ++j) {

            const float2 subpixel_jitter = make_float2( delta.x * (i + rnd(seed)), delta.y * (j + rnd( seed )));
            const float2 screen(make_float2(ix + subpixel_jitter.x, iy + subpixel_jitter.y)
                            / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);
        
            // note: nau already takes into account the field of view and ratio when computing 
            // camera horizontal and vertical
            float3 origin = camera.position;
            float3 rayDir = normalize(camera.direction
                                + (screen.x ) * camera.horizontal
                                + (screen.y ) * camera.vertical);

            RadiancePRD prd;
            prd.emitted      = make_float3(0.f);
            prd.radiance     = make_float3(0.f);
            prd.attenuation  = make_float3(1.f);
            prd.countEmitted = true;
            prd.done         = false;
            prd.seed         = seed;

            uint32_t u0, u1;
            packPointer( &prd, u0, u1 );             
            
            for (int k = 0; k < maxDepth && !prd.done; ++k) {
                
                optixTrace(optixLaunchParams.traversable,
                        origin,
                        rayDir,
                        0.001f,    // tmin
                        1.3e20f,  // tmax
                        0.0f, OptixVisibilityMask( 1 ),
                        OPTIX_RAY_FLAG_NONE, RAIDANCE, RAY_TYPE_COUNT, RAIDANCE, u0, u1 );

                result += prd.emitted;
                result += prd.radiance * prd.attenuation;

                origin = prd.origin;
                rayDir = prd.direction;

                const bool russian = true;
                
                // Russian Roulette
                if (russian) {
                    float prob = max_component(prd.attenuation);
                    if (maxDepth > 7) {
                        if (rnd(prd.seed) >= prob){
                            prd.radiance += prd.attenuation;
                            prd.done = true;
                        }
                        else
                            prd.attenuation *= 1.f / prob;
                    }
                }
            }
        }
    }

    result = result / (squaredRaysPerPixel*squaredRaysPerPixel);
    float gamma = optixLaunchParams.global->gamma;
    
    // compute index
    const uint32_t fbIndex = ix + iy*optixGetLaunchDimensions().x;

    optixLaunchParams.global->accumBuffer[fbIndex] = 
        (optixLaunchParams.global->accumBuffer[fbIndex] * optixLaunchParams.frame.subFrame +
        make_float4(result.x, result.y, result.z, 1)) /(optixLaunchParams.frame.subFrame+1);

    
    float4 rgbaf = optixLaunchParams.global->accumBuffer[fbIndex];
    //convert float (0-1) to int (0-255)
    const int r = int(255.0f*min(1.0f, pow(rgbaf.x, 1/gamma)));
    const int g = int(255.0f*min(1.0f, pow(rgbaf.y, 1/gamma)));
    const int b = int(255.0f*min(1.0f, pow(rgbaf.z, 1/gamma))) ;

    // convert to 32-bit rgba value 
    const uint32_t rgba = 0xff000000 | (r<<0) | (g<<8) | (b<<16);
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
  

