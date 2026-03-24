#define bmax 1.1229189f
#define xh   0.1f
#define Q0   1.0f

#define expf(x) exp(x)
#define powf(x,y) pow(x,y)
#define sqrtf(x) sqrt(x)

inline float bstar_func(float b, float Q) {

    float t  = b / bmax;
    float t2 = t * t;
    float t4 = t2 * t2;

    float denom = sqrt(sqrt(1.0f + t4));

    return b / denom;
}

typedef struct {
    float g2, l, l2;
    float N1, N2, N3;
    float a1, a2, a3;
    float s1, s2, s3;
} Params_Struct;

inline float2 NP_f_func(float x, float b, __constant Params_Struct* params)
{
    float g2 = params->g2,  l = params->l,  l2 = params->l2;
    float N1 = params->N1, N2 = params->N2, N3 = params->N3;
    float a1 = params->a1, a2 = params->a2, a3 = params->a3;
    float s1 = params->s1, s2 = params->s2, s3 = params->s3;

    float b2 = b*b;

    float denom1 = 1.0f / (powf(xh, s1) * powf(1.0f - xh, a1*a1));
    float denom2 = 1.0f / (powf(xh, s2) * powf(1.0f - xh, a2*a2));
    float denom3 = 1.0f / (powf(xh, s3) * powf(1.0f - xh, a3*a3));

    float g1x = N1 * (powf(x, s1) * powf(1.0f - x, a1*a1)) * denom1;
    float g2x = N2 * (powf(x, s2) * powf(1.0f - x, a2*a2)) * denom2;
    float g3x = N3 * (powf(x, s3) * powf(1.0f - x, a3*a3)) * denom3;

    float e1 = expf(-0.25f * g1x * b2);
    float e2 = expf(-0.25f * g2x * b2);
    float e3 = expf(-0.25f * g3x * b2);

    float Sud_num   = g1x*e1 + (l*l)*g2x*g2x*(1.0f - 0.25f*g2x*b2)*e2 + (l2*l2)*g3x*e3;
    float Sud_denom = g1x + (l*l)*g2x*g2x + (l2*l2)*g3x;

    float t = b / bmax; float t2 = t*t; float t4 = t2*t2;
    float gK = -0.5f * (g2*g2) * b2;

    float SNP_mu = Sud_num / Sud_denom;
    float SNP_ze = 0.5f * gK;

    return (float2)(SNP_mu, SNP_ze);
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