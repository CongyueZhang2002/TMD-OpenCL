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
    float g2;
} Params_Struct;

inline float8 NP_f_func(float x, float b, __constant Params_Struct* params)
{
    float g2 = params->g2;
    float b2 = b*b;

    float gK = -0.5f * (g2*g2) * b2;

    float SNP_ze = 0.5f * gK;

    return (float8)(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, SNP_ze, 0.0f, 0.0f);
}