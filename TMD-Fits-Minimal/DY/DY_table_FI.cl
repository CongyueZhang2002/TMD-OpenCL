#define b0 1.1229189f
#define db0 0.89053626f

inline float DY_NP_func(float b, float xp, float xN, float Q, __constant Params_Struct* params)
{
    float2 NP_p = NP_f_func(xp, b, params);
    float2 NP_n = NP_f_func(xN, b, params);

    float NP_Suda = NP_p.s0 * NP_n.s0;
    float NP_CS = NP_p.s1 + NP_n.s1;

    float log_Zeta = 2.0f * log(Q/mustar_func(b,Q)); 
    float NP_Zeta = exp(NP_CS * log_Zeta);

    return NP_Suda * NP_Zeta;
}

inline void Atomic_Add(__global float* p, float v){
    __global volatile unsigned int* u = (__global volatile unsigned int*)p;
    unsigned int old = *u, neu;
    for(;;){
        neu = as_uint(as_float(old) + v);
        unsigned int prev = atomic_cmpxchg(u, old, neu);
        if (prev == old) break;
        old = prev;
    }
}

__kernel void DY_xsec(
    __global const float* xp_vec,
    __global const float* xN_vec,
    __global const float* Q_vec,
    __global const float* b_vec,

    __global const float* pert_vec,

    __constant Params_Struct* params,
    int dim,
    __global float* xsec,
    __local  float* sum_data)
{
    int mult_id = get_global_id(0);
        int mult_size = get_global_size(0);

        int add_id = get_local_id(0);
            int add_size = get_local_size(0);

            float sum = 0.0f;
            for (int i = mult_id; i < dim; i += mult_size){
                float product = pert_vec[i] * DY_NP_func(b_vec[i], xp_vec[i], xN_vec[i], Q_vec[i], params);
                sum += product;
            }

            sum_data[add_id] = sum;  
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int off = add_size >> 1; off; off >>= 1){ 
                if (add_id < off) sum_data[add_id] += sum_data[add_id+off]; 
                barrier(CLK_LOCAL_MEM_FENCE); 
            }
            if (add_id == 0) Atomic_Add(xsec, sum_data[0]); 
}