#define bmax 1.1229189f

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
    float g2;
    float a_u, a_ub, a_d, a_db, a_sea;
} Params_Struct;

inline float8 NP_f_func(float x, float b, __constant Params_Struct* params)
{
    float g2 = params->g2;
    float a_u = params->a_u;
    float a_ub = params->a_ub;
    float a_d = params->a_d;
    float a_db = params->a_db;
    float a_sea = params->a_sea;

    float b2 = b*b;

    float shape_u   = a_u ;
    float shape_ub  = a_ub;
    float shape_d   = a_d;
    float shape_db  = a_db;
    float shape_sea = a_sea;
    float SNP_u   = expf(- shape_u   * shape_u   *b2);
    float SNP_ub  = expf(- shape_ub  * shape_ub  *b2);
    float SNP_d   = expf(- shape_d   * shape_d   *b2);
    float SNP_db  = expf(- shape_db  * shape_db  *b2);
    float SNP_sea = expf(- shape_sea * shape_sea *b2);

    float gK = -0.5f * (g2*g2) * b2;
    float SNP_ze = 0.5f * gK;

    return (float8)(SNP_u, SNP_ub, SNP_d, SNP_db, SNP_sea, SNP_ze, 0.0f, 0.0f);
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

    float8 r = NP_f_func(x[i], b[i], P);
    out_mu[i] = r.x;
    out_ze[i] = r.y;
}