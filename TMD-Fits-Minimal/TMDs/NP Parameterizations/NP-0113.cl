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

inline float mustar_func(float b, float Q) {
    float bstar = bstar_func(b, Q0);
    float mu = bmax / bstar;
    return mu; 
}

typedef struct {
  float g2, bmax_CS, g3, m;
} Params_Struct;

inline float clampf(float x,float lo,float hi){ return fmin(fmax(x,lo),hi); }
inline float sechf(float t){ t=fabs(t); float u=exp(-2.f*t); return (2.f*exp(-t))/(1.f+u); }

inline float2 NP_f_func(float x, float b, __constant Params_Struct* p)
{
  const float g2       = p->g2;
  const float bmax_CS  = p->bmax_CS;
  const float g3       = p->g3;
  const float m = p->m;

  float u = b/bmax_CS;
  float um = pow(u,m);   
  float bstar_CS = b / pow(1 + u,1/m);

  float SNP_ze = -0.25f * (g2*g2 + g3*g3*log(bstar_CS/bmax_CS)) * (b*bstar_CS);

  return (float2)(1.0, SNP_ze);
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