// ================================================
//            Grid generators
// ================================================

// axis_generator

__kernel void build_axis(
    const float a0,          
    const float an,         
    const int   N,          
    const float power,      
    __global float* out    
){
    int i = get_global_id(0);
    if (i >= N) return;

    float u      = (float)i / (float)(N - 1);      
    float up     = pow(u, power);
    out[i]       = a0 + (an - a0) * up;
}

__kernel void build_grid(
    __global const float* x1_array,  int N1,
    __global const float* x2_array,  int N2,
    __constant const Params* P,
    __global float2* grid         
){
    int i = get_global_id(0);
        int j = get_global_id(1);
            if (i >= N1 || j >= N2) return;

            int id = j * N1 + i;
            float x1 = x1_array[i];
            float x2 = x2_array[j];

    grid[id] = NP_f_func(x1, x2, P);
}

// ================================================
//            Grid Interpolators
// ================================================

inline float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

inline float search_id(float x, float a0, float inv_range, float inv_p, int N, __private int* i_out)
{

    float w = clamp01((x - a0) * inv_range);
    float u = native_powr(w, inv_p);
    float s = u * (float)(N - 1);

    int id = (int)floor(s);
    id = min(id, N - 2);

    *i_out = id;

    return s - (float)id;
}

__kernel void grid_bilinear_interp_aos_power(
    __global const float* x1_vec,
    __global const float* x2_vec,
    int n_points,

    // axis-1 (x1) params
    float a0_1,    float inv_range1,    float inv_p1,   int N1,
    // axis-2 (x2) params
    float a0_2,    float inv_range2,    float inv_p2,   int N2,

    __global const float2* grid_aos,     // length N1*N2 (row-major: x1 fastest)
    __global float2*       out           // length n_points
){
    int gid = get_global_id(0);
    int gsz = get_global_size(0);

    for (int p = gid; p < n_points; p += gsz) {
        float x1 = x1_vec[p];
        float x2 = x2_vec[p];

        // Invert axes -> get (i1,t1) and (i2,t2) without any searching
        int i1, i2;
        float t1 = search_id(x1, a0_1, inv_range1, inv_p1, N1, &i1);
        float t2 = search_id(x2, a0_2, inv_range2, inv_p2, N2, &i2);

        // Bilinear weights
        float omt1 = 1.0f - t1, omt2 = 1.0f - t2;
        float w00 = omt1 * omt2;
        float w10 = t1   * omt2;
        float w01 = omt1 * t2;
        float w11 = t1   * t2;

        // Corner indices (row-major: x1 is the fast index)
        int base = i2 * N1 + i1;
        float2 v00 = grid_aos[base];
        float2 v10 = grid_aos[base + 1];
        float2 v01 = grid_aos[base + N1];
        float2 v11 = grid_aos[base + N1 + 1];

        // Blend both channels at once (float2 math)
        float2 val = (float2)(0.0f, 0.0f);
        val += v00 * w00;
        val += v10 * w10;
        val += v01 * w01;
        val += v11 * w11;

        out[p] = val;  // (mu, ze)
    }
}