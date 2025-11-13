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
    float b_u, b_ub, b_d, b_db, b_sea;
} Params_Struct;

//inline float8 NP_f_func(float x, float b, __constant Params_Struct* params)
//{
//    float g2 = params->g2;
//    float a_u = params->a_u, b_u = params->b_u;
//    float a_ub = params->a_ub, b_ub = params->b_ub;
//    float a_d = params->a_d, b_d = params->b_d;
//    float a_db = params->a_db, b_db = params->b_db;
//    float a_sea = params->a_sea, b_sea = params->b_sea;
//
//    float b2 = b*b;
//    float xb = 1.0f; //- x;
//
//    float shape_u   = (a_u   * xb + b_u   * x);
//    float shape_ub  = (a_ub  * xb + b_ub  * x);
//    float shape_d   = (a_d   * xb + b_d   * x);
//    float shape_db  = (a_db  * xb + b_db  * x);
//    float shape_sea = (a_sea * xb + b_sea * x);
//
//    float SNP_u   = 1/cosh(shape_u   * b);
//    float SNP_ub  = 1/cosh(shape_ub  * b);
//    float SNP_d   = 1/cosh(shape_d   * b);
//    float SNP_db  = 1/cosh(shape_db  * b);
//    float SNP_sea = 1/cosh(shape_sea * b);
//
//    float gK = -0.5f * (g2*g2) * b2;
//    float SNP_ze = 0.5f * gK;
//
//    return (float8)(SNP_u, SNP_ub, SNP_d, SNP_db, SNP_sea, SNP_ze, 0.0f, 0.0f);
//}

inline float8 NP_f_func(float x, float b, __constant Params_Struct* params)
{
    float g2 = params->g2;
    float a_u = params->a_u, b_u = params->b_u;
    float a_ub = params->a_ub, b_ub = params->b_ub;
    float a_d = params->a_d, b_d = params->b_d;
    float a_db = params->a_db, b_db = params->b_db;
    float a_sea = params->a_sea, b_sea = params->b_sea;

    float b2 = b*b;
    float xbar = (1.0f -x);

    float shape_u   = (a_u*xbar   + b_u   * x);
    float shape_ub  = (a_ub*xbar  + b_ub  * x);
    float shape_d   = (a_d*xbar   + b_d   * x);
    float shape_db  = (a_db*xbar  + b_db  * x);
    float shape_sea = (a_sea*xbar + b_sea * x);

    float SNP_u   = 1/cosh(shape_u *b);
    float SNP_ub  = 1/cosh(shape_ub *b);
    float SNP_d   = 1/cosh(shape_d *b);
    float SNP_db  = 1/cosh(shape_db *b);
    float SNP_sea = 1/cosh(shape_sea *b);

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