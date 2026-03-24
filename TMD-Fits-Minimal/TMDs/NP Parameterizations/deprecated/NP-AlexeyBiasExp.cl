#define bmax 1.1229189f
#define expf(x) exp(x)
#define powf(x,y) pow(x,y)
#define sqrtf(x) sqrt(x)

inline float mustar_func(float b, float Q) {
    float mu = bmax / b;
    return max(mu, 1.0f);
}

typedef struct {
  float lambda1, lambda2, l0;
  float BNP, c0, c1;
} Params_Struct;

inline float clampf(float x, float lo, float hi) { return fmin(fmax(x, lo), hi); }
inline float sechf(float t) { t = fabs(t); float u = exp(-2.f * t); return (2.f * exp(-t)) / (1.f + u); }

inline float2 NP_f_func(float x, float b, __constant Params_Struct* p)
{
  const float lambda1 = p->lambda1;
  const float lambda2 = p->lambda2;
  const float l0 = p->l0;
  const float BNP = p->BNP;
  const float c0 = p->c0;
  const float c1 = p->c1;

  x = clampf(x, 1e-7f, 1.f - 1e-7f);
  float xbar = 1.f - x;
  float b2 = b * b;
  float shape = lambda1 * xbar + lambda2 * x;
  float denom = sqrtf(1.f + l0 * x * x * b2);
  float SNP_mu = expf(-shape * b2 / fmax(denom, 1e-12f));

  float v = b / fmax(BNP, 1e-6f);
  float bstar = b / sqrtf(1.f + v * v);
  float log_ratio = log(fmax(bstar / fmax(BNP, 1e-6f), 1e-7f));
  float DNP = b * bstar * (c0 + c1 * log_ratio);
  float SNP_ze = -DNP;

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
