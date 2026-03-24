#define bmax 1.1229189f
#define xh   0.1f
#define Q0   1.0f

#define expf(x) exp(x)
#define powf(x,y) pow(x,y)
#define sqrtf(x) sqrt(x)

inline float mustar_func_quartic(float b, float Q) {

    float t  = b / bmax;
    float t2 = t * t;
    float t4 = t2 * t2;
    float denom = sqrt(sqrt(1.0f + t4));
    //float denom = sqrt(1.0f + t2);
    float bstar = b / denom;

    float mu = bmax / bstar;
    return mu; 
}

inline float mustar_func(float b, float Q) {

    float mu = bmax / b;
    return max(mu, 1.0f); 
}

typedef struct {
  float g2, bmax_CS, power_CS;
  float a1,a2,a3,a4;     // xshape coeffs
  float b1,b2,b3;        // bshape exponent coeffs
  float a;           // alpha
} Params_Struct;

inline float clampf(float x,float lo,float hi){ return fmin(fmax(x,lo),hi); }
inline float sechf(float t){ t=fabs(t); float u=exp(-2.f*t); return (2.f*exp(-t))/(1.f+u); }

inline float2 NP_f_func(float x, float b, __constant Params_Struct* p)
{
  const float g2       = p->g2;
  const float bmax_CS  = p->bmax_CS;
  const float power_CS = p->power_CS;

  const float a1 = p->a1, a2 = p->a2, a3 = p->a3, a4 = p->a4;
  const float b1 = p->b1, b2 = p->b2, b3 = p->b3;
  const float alpha = p->a;

  x = clampf(x, 1e-7f, 1.f-1e-7f);
  float xbar = 1.f - x, xxbar = x*xbar;

  float xshape = a1*x + a2*xbar + a3*xxbar + a4*log(x);

  float expo = b1*x*x + b2*xbar*xbar + 2.f*b3*xxbar;
  expo = clampf(expo, -80.f, 80.f);
  float bshape = exp(expo);

  float t  = b/(bmax*bshape);
  float t2 = t*t;
  float t4 = t2*t2;                      
  float bstar = b * powr(1.f + t4, 0.25f*(alpha - 1.f));
  //float bstar = b * powr(1.f + t2, 0.5f*(alpha - 1.f));

  float u = b/bmax_CS; 
  float u2 = u*u; 
  float u4 = u2*u2;    
  float bstar_CS = b * powr(1.f + u4, 0.25f*(power_CS - 1.f));
  //float bstar_CS = b * powr(1.f + u2, 0.5f*(power_CS - 1.f));

  float SNP_mu = sechf(xshape * bstar);
  float SNP_ze = -0.25f * (g2*g2) * (bstar_CS*bstar_CS);

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