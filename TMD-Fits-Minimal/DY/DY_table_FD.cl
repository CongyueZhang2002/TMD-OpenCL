#define b0 1.1229189f
#define db0 0.89053626f

inline float16 DY_NP_func(float b, float xp, float xN, float Q, float isoscalarity, __constant Params_Struct* params)
{
    float8 NP_p = NP_f_func(xp, b, params);
    float8 NP_n = NP_f_func(xN, b, params);

    float u_p = NP_p.s0, ub_p = NP_p.s1, d_p = NP_p.s2, db_p = NP_p.s3, sea_p = NP_p.s4, z_p = NP_p.s5;
    float u_n = NP_n.s0, ub_n = NP_n.s1, d_n = NP_n.s2, db_n = NP_n.s3, sea_n = NP_n.s4, z_n = NP_n.s5;

    float NP_CS = z_p + z_n;
    float log_Zeta = 2.0f * log(Q * bstar_func(b, Q) * db0); 
    float NP_Zeta = exp(NP_CS * log_Zeta);

    float16 NP_Suda;
    if (isoscalarity > 0.0f) {
        NP_Suda = (float16)(
            u_p*ub_n,  u_p*db_n,  d_p*ub_n,  d_p*db_n,
            ub_p*u_n,  ub_p*d_n,  db_p*u_n,  db_p*d_n,
            sea_p*sea_n,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        );
    } else {
        NP_Suda = (float16)(
            u_p*u_n,   u_p*d_n,   d_p*u_n,   d_p*d_n,
            ub_p*ub_n, ub_p*db_n, db_p*ub_n, db_p*db_n,
            sea_p*sea_n,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        );
    }
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

// up_uN, up_dN, dp_uN, dp_dN, ubp_ubN, ubp_dbN, dbp_ubN, dbp_dbN, sea

__kernel void DY_xsec(
    __global const float* xp_vec,
    __global const float* xN_vec,
    __global const float* Q_vec,
    __global const float* b_vec,

    __global const float* pert_up_uN_vec,
    __global const float* pert_up_dN_vec,
    __global const float* pert_dp_uN_vec,
    __global const float* pert_dp_dN_vec,
    __global const float* pert_ubp_ubN_vec,
    __global const float* pert_ubp_dbN_vec,
    __global const float* pert_dbp_ubN_vec,
    __global const float* pert_dbp_dbN_vec,
    __global const float* pert_sea_vec,

    const float isoscalarity,
    
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
                float16 NP_array = DY_NP_func(b_vec[i], xp_vec[i], xN_vec[i], Q_vec[i], isoscalarity, params);
                float product = (
                      pert_up_uN_vec[i] * NP_array.s0
                    + pert_up_dN_vec[i] * NP_array.s1
                    + pert_dp_uN_vec[i] * NP_array.s2
                    + pert_dp_dN_vec[i] * NP_array.s3
                    + pert_ubp_ubN_vec[i] * NP_array.s4
                    + pert_ubp_dbN_vec[i] * NP_array.s5
                    + pert_dbp_ubN_vec[i] * NP_array.s6
                    + pert_dbp_dbN_vec[i] * NP_array.s7
                    + pert_sea_vec[i] * NP_array.s8
                );
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