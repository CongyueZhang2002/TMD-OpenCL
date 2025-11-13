using Statistics, Random
using OpenCL
const cl = OpenCL.cl

# ---------------- settings ----------------
const grid_eps_x = 1e-5
const grid_eps_b = 1e-4
const grid_variable_ranges = [
    (Float32(grid_eps_x), Float32(1 - grid_eps_x)),  # x1 : (a0, an)
    (Float32(grid_eps_b), Float32(30.0)),            # x2 : (a0, an)
]
const grid_variable_settings = [
    (Int32(1000), Float32(2.5)),   # (N1, power1)
    (Int32(1000), Float32(2.5)),   # (N2, power2)
]

# ---- derived host-side constants ----
const a0_1, an_1 = grid_variable_ranges[1]
const a0_2, an_2 = grid_variable_ranges[2]
const N1,  power1 = grid_variable_settings[1]
const N2,  power2 = grid_variable_settings[2]
const N1_i = Int32(N1)
const N2_i = Int32(N2)

# inverse params for interpolator (power-mapped axes)
const inv_range1 = 1f0 / (an_1 - a0_1)
const inv_range2 = 1f0 / (an_2 - a0_2)
const inv_p1     = 1f0 / power1
const inv_p2     = 1f0 / power2

# ---------------- OpenCL build ----------------
# np.cl must define: build_axis, build_grid, grid_bilinear_interp_aos_power, NP_f_vec
src_np  = read(abspath(joinpath(@__DIR__, "..", "TMDs", "NP Parameterizations", "NP-Default.cl")), String)
src_itp = read(abspath(joinpath(@__DIR__, "interpolation.cl")), String)
prog    = cl.Program(; source = string(src_np, "\n\n", src_itp)) |> cl.build!

cl.build!(prog; options="-cl-fast-relaxed-math -cl-mad-enable")
k_axis   = cl.Kernel(prog, "build_axis")
k_build  = cl.Kernel(prog, "build_grid")
k_vec    = cl.Kernel(prog, "NP_f_vec")
k_interp = cl.Kernel(prog, "grid_bilinear_interp_aos_power")

# ---------------- Params (host mirror of OpenCL) ----------------
Base.@kwdef struct Params
    g2::Float32;  l::Float32;  l2::Float32
    N1::Float32;  N2::Float32; N3::Float32
    a1::Float32;  a2::Float32; a3::Float32
    s1::Float32;  s2::Float32; s3::Float32
end
initial_params = Float32.([0.232, 1.1, 0.0194, 0.171, 0.115, 0.00894, 1.83, 3.42, 2.34, 1.27, 0.0949, 7.82])
g2, λ, λ2, N1p, N2p, N3p, α1, α2, α3, σ1, σ2, σ3 = initial_params
P = Params(g2=g2, l=λ, l2=λ2, N1=N1p, N2=N2p, N3=N3p, a1=α1, a2=α2, a3=α3, s1=σ1, s2=σ2, s3=σ3)
dP = CLArray([P])

# ---------------- Device axis generation via kernel ----------------
# buffers
dx1 = CLArray{Float32}(undef, N1)
dx2 = CLArray{Float32}(undef, N2)

# build_axis signature: (float a0, float an, int N, float power, __global float* out)
axis_sig = Tuple{Float32, Float32, Int32, Float32, Ptr{Float32}}

# warm-up (tiny)
cl.clcall(k_axis, axis_sig, a0_1, an_1, N1_i, power1, dx1; global_size=(min(Int(N1),64),))
cl.clcall(k_axis, axis_sig, a0_2, an_2, N2_i, power2, dx2; global_size=(min(Int(N2),64),))
cl.finish(cl.queue())

# full axes
cl.clcall(k_axis, axis_sig, a0_1, an_1, N1_i, power1, dx1; global_size=(Int(N1),))
cl.clcall(k_axis, axis_sig, a0_2, an_2, N2_i, power2, dx2; global_size=(Int(N2),))
cl.finish(cl.queue())

# ---------------- Build NP grid (AoS float2) on device ----------------
# build_grid signature: (__global const float* x1, int N1, __global const float* x2, int N2, __constant Params*, __global float2* grid)
dgrid = CLArray{NTuple{2,Float32}}(undef, Int(N1)*Int(N2))
t_grid = @elapsed begin
    cl.clcall(k_build,
        Tuple{Ptr{Float32}, Int32, Ptr{Float32}, Int32, Ptr{Params}, Ptr{NTuple{2,Float32}} },
        dx1, N1_i, dx2, N2_i, dP, dgrid;
        global_size=(Int(N1), Int(N2))
    )
    cl.finish(cl.queue())
end
println("Grid built on GPU: N1=$(Int(N1)), N2=$(Int(N2))  (t = $(round(t_grid,digits=3)) s)")

# ---------------- Interpolation vs direct (GPU) ----------------
total_points = 100_000_000
# single batch to minimize overhead (ensure your VRAM can hold it)
batch_points = total_points

# prepare random queries on host & upload once
Random.seed!(123)
x1h = rand(Float32, total_points) .* (an_1 - a0_1) .+ a0_1
x2h = rand(Float32, total_points) .* (an_2 - a0_2) .+ a0_2
dx1q = CLArray(x1h)
dx2q = CLArray(x2h)

# Interp outputs: float2 per point
dout = CLArray{NTuple{2,Float32}}(undef, total_points)
# Direct outputs
dmu = CLArray{Float32}(undef, total_points)
dze = CLArray{Float32}(undef, total_points)

# kernel signatures
interp_sig = Tuple{
    Ptr{Float32}, Ptr{Float32}, Int32,
    Float32, Float32, Float32, Int32,   # a0_1, inv_range1, inv_p1, N1
    Float32, Float32, Float32, Int32,   # a0_2, inv_range2, inv_p2, N2
    Ptr{NTuple{2,Float32}},             # grid_aos
    Ptr{NTuple{2,Float32}}              # out (float2)
}
vec_sig = Tuple{
    Ptr{Float32}, Ptr{Float32}, Int32, Ptr{Params}, Ptr{Float32}, Ptr{Float32}
}

# warm-up (small)
warm_n = min(200_000, total_points)
cl.clcall(k_interp, interp_sig,
    dx1q, dx2q, Int32(warm_n),
    a0_1, inv_range1, inv_p1, N1_i,
    a0_2, inv_range2, inv_p2, N2_i,
    dgrid, dout; global_size=(warm_n,)
)
cl.clcall(k_vec, vec_sig,
    dx1q, dx2q, Int32(warm_n), dP, dmu, dze; global_size=(warm_n,)
)
cl.finish(cl.queue())

# timed: interpolation (all points)
t_interp = @elapsed begin
    cl.clcall(k_interp, interp_sig,
        dx1q, dx2q, Int32(total_points),
        a0_1, inv_range1, inv_p1, N1_i,
        a0_2, inv_range2, inv_p2, N2_i,
        dgrid, dout; global_size=(total_points,)
    )
    cl.finish(cl.queue())
end

# timed: direct (all points)
t_direct = @elapsed begin
    cl.clcall(k_vec, vec_sig,
        dx1q, dx2q, Int32(total_points), dP, dmu, dze; global_size=(total_points,)
    )
    cl.finish(cl.queue())
end

println("GPU Interp: 100M points in $(round(t_interp,digits=3)) s  (~$(round(total_points/t_interp/1e6,digits=2)) Mpts/s)")
println("GPU Direct: 100M points in $(round(t_direct,digits=3)) s (~$(round(total_points/t_direct/1e6,digits=2)) Mpts/s)")

# ---------------- Error stats: interpolation vs direct (GPU subset) ----------------
subset_n = min(1_000_000, total_points)
Random.seed!(42)
x1s = rand(Float32, subset_n) .* (an_1 - a0_1) .+ a0_1
x2s = rand(Float32, subset_n) .* (an_2 - a0_2) .+ a0_2
dx1s = CLArray(x1s); dx2s = CLArray(x2s)

douts = CLArray{NTuple{2,Float32}}(undef, subset_n)
dmu_s = CLArray{Float32}(undef, subset_n)
dze_s = CLArray{Float32}(undef, subset_n)

cl.clcall(k_interp, interp_sig,
    dx1s, dx2s, Int32(subset_n),
    a0_1, inv_range1, inv_p1, N1_i,
    a0_2, inv_range2, inv_p2, N2_i,
    dgrid, douts; global_size=(subset_n,)
)
cl.clcall(k_vec, vec_sig,
    dx1s, dx2s, Int32(subset_n), dP, dmu_s, dze_s; global_size=(subset_n,)
)
cl.finish(cl.queue())

outs   = Array(douts)                         # Vector{NTuple{2,Float32}}
mu_itp = Float32[ v[1] for v in outs ]
ze_itp = Float32[ v[2] for v in outs ]
mu_ref = Array(dmu_s)
ze_ref = Array(dze_s)

# thresholding
thr = 1e-5
mask_mu = abs.(mu_ref) .>= thr
mask_ze = abs.(ze_ref) .>= thr

rel_mu = abs.(mu_itp[mask_mu] .- mu_ref[mask_mu]) ./ (abs.(mu_ref[mask_mu]) .+ 1f-20)
rel_ze = abs.(ze_itp[mask_ze] .- ze_ref[mask_ze]) ./ (abs.(ze_ref[mask_ze]) .+ 1f-20)

stats = function (name, v, kept, total)
    if isempty(v)
        println("[$name] no points kept; kept=$kept / total=$total"); return
    end
    q50  = quantile(v, 0.50); q90 = quantile(v,0.90); q99 = quantile(v,0.99); q999 = quantile(v,0.999)
    println("[$name] kept=$kept / total=$total  max=$(round(maximum(v),sigdigits=3))  ",
            "p99.9=$(round(q999,sigdigits=3))  p99=$(round(q99,sigdigits=3))  ",
            "p90=$(round(q90,sigdigits=3))  median=$(round(q50,sigdigits=3))")
end

println("---- Interp vs Direct error (GPU subset: $(subset_n), threshold=$(thr)) ----")
stats("rel μ", rel_mu, count(mask_mu), subset_n)
stats("rel ζ", rel_ze, count(mask_ze), subset_n)
