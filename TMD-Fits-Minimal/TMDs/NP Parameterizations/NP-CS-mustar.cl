#define bmax 1.1229189f
#define xh   0.1f
#define Q0   1.0f

#define expf(x) exp(x)
#define powf(x,y) pow(x,y)
#define sqrtf(x) sqrt(x)

inline float bstar_func(float b, float Q) {
    return b;
}

inline float mustar_func(float b, float Q) {
    float mu = bmax / b;
    return fmax(mu, 1.0f); 
}

typedef struct {
    float g2;
} Params_Struct;

inline float2 NP_f_func(float x, float b, __constant Params_Struct* params)
{
    float g2 = params->g2;

    float b2 = b*b;

    float gK = -0.5f * (g2*g2) * b2;

    float SNP_ze = 0.5f * gK;

    return (float2)(1.0f, SNP_ze);
}

__kernel void NP_f_vec(
    __global const float* x,
    __global const float* b,
    int N,
    __constant Params_Struct* P,
    __global float* out_mu,
    __global float* out_ze
){
    int i = get_global_id(0);
    if (i >= N) return;

    float2 r = NP_f_func(x[i], b[i], P);
    out_mu[i] = r.x;
    out_ze[i] = r.y;
}