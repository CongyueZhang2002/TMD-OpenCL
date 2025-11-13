using Statistics
using OpenCL
const cl = OpenCL.cl

# ---------------- Build program & kernel ----------------
src  = read(joinpath(@__DIR__, "NP-Default.cl"), String)
prog = cl.Program(; source = src)
cl.build!(prog)
kern = cl.Kernel(prog, "NP_f_vec")

# -------- Struct mirroring the OpenCL 'Params' (same order/types) -----------
Base.@kwdef struct Params
    g2::Float32;  l::Float32;  l2::Float32
    N1::Float32;  N2::Float32; N3::Float32
    a1::Float32;  a2::Float32; a3::Float32
    s1::Float32;  s2::Float32; s3::Float32
end

# --------------- GPU driver (struct as single kernel arg) -------------------
function np_vec(x::Vector{Float32}, b::Vector{Float32}; p::Params)
    @assert length(x) == length(b)
    N = Int32(length(x))

    dx  = CLArray(x)
    db  = CLArray(b)
    dmu = similar(dx)
    dze = similar(dx)

    # one-element constant buffer with params
    dP = CLArray([p])

    t_gpu = @elapsed begin
        cl.clcall(kern,
            Tuple{Ptr{Float32}, Ptr{Float32}, Int32,
                  Ptr{Params}, Ptr{Float32}, Ptr{Float32}},
            dx, db, N,
            dP, dmu, dze;
            global_size = (length(x),)
        )
        cl.finish(cl.queue())
    end

    return Array(dmu), Array(dze), t_gpu
end

# --------------------- CPU reference (matches kernel) -----------------------
const bmax = 1.1229189f0
const xhat = 0.1f0
@inline exp32(x::Float32) = exp(x)

function NP_f_func_cpu(x::Float32, b::Float32, P::Params)
    # extract once
    g2 = P.g2;  l = P.l;  l2 = P.l2
    N1 = P.N1;  N2 = P.N2; N3 = P.N3
    a1 = P.a1;  a2 = P.a2; a3 = P.a3
    s1 = P.s1;  s2 = P.s2; s3 = P.s3

    b2 = b*b

    denom1 = 1f0 / (xhat^s1 * (1f0 - xhat)^(a1*a1))
    denom2 = 1f0 / (xhat^s2 * (1f0 - xhat)^(a2*a2))
    denom3 = 1f0 / (xhat^s3 * (1f0 - xhat)^(a3*a3))

    g1x = N1 * (x^s1 * (1f0 - x)^(a1*a1)) * denom1
    g2x = N2 * (x^s2 * (1f0 - x)^(a2*a2)) * denom2
    g3x = N3 * (x^s3 * (1f0 - x)^(a3*a3)) * denom3

    e1 = exp32(-0.25f0 * g1x * b2)
    e2 = exp32(-0.25f0 * g2x * b2)
    e3 = exp32(-0.25f0 * g3x * b2)

    Sud_num   = g1x*e1 + (l*l)*g2x*g2x*(1f0 - 0.25f0*g2x*b2)*e2 + (l2*l2)*g3x*e3
    Sud_denom = g1x + (l*l)*g2x*g2x + (l2*l2)*g3x

    t  = b / bmax;  t2 = t*t;  t4 = t2*t2
    gK = -0.5f0 * (g2*g2) * b2 / sqrt(sqrt(1f0 + t4))

    SNP_mu = Sud_num / Sud_denom
    SNP_ze = 0.5f0 * gK
    return SNP_mu, SNP_ze
end

# ------------------------- Parameters & data -------------------------------
initial_params = [0.232, 1.1, 0.0194, 0.171, 0.115, 0.00894, 1.83, 3.42, 2.34, 1.27, 0.0949, 7.82]
g2, λ, λ2, N1p, N2p, N3p, α1, α2, α3, σ1, σ2, σ3 = Float32.(initial_params)
P = Params(g2=g2, l=λ, l2=λ2, N1=N1p, N2=N2p, N3=N3p, a1=α1, a2=α2, a3=α3, s1=σ1, s2=σ2, s3=σ3)

# Square grid (N×N queries) flattened into length N^2
Nside = 2000
x_axis = Float32.(range(1e-5, 1-1e-5; length=Nside))
b_axis = Float32.(range(1e-3, 30.0;   length=Nside))
x_vec  = Float32.(repeat(x_axis; inner=Nside))  # length N^2
b_vec  = Float32.(repeat(b_axis; outer=Nside))  # length N^2

# Warm-up (smaller)
Nw   = 100
xw   = Float32.(repeat(Float32.(range(1e-5, 1-1e-5; length=Nw)); inner=Nw))
bw   = Float32.(repeat(Float32.(range(1e-3, 30.0;   length=Nw)); outer=Nw))
_μw, _ζw, _ = np_vec(xw, bw; p = P)

# --------------------- Run GPU & CPU, compare ------------------------------
NP_μ_gpu, NP_ζ_gpu, t_gpu = np_vec(x_vec, b_vec; p = P)

NP_μ_cpu = similar(NP_μ_gpu)
NP_ζ_cpu = similar(NP_ζ_gpu)

t_cpu = @elapsed begin
    @inbounds for i in eachindex(x_vec)
        μ, ζ = NP_f_func_cpu(x_vec[i], b_vec[i], P)
        NP_μ_cpu[i] = μ
        NP_ζ_cpu[i] = ζ
    end
end

# Relative diffs (robust to zero): |Δ| / (|ref| + eps)
eps32 = 1f-20
diff_μ = abs.(NP_μ_gpu .- NP_μ_cpu) ./ (abs.(NP_μ_cpu) .+ eps32)
diff_ζ = abs.(NP_ζ_gpu .- NP_ζ_cpu) ./ (abs.(NP_ζ_cpu) .+ eps32)

println("Test for $(Nside) x $(Nside) NP grid")
println("CPU time: $(round(t_cpu, digits=5)) s")
println("GPU time: $(round(t_gpu, digits=5)) s")
println("speed gain: $(round(t_cpu / t_gpu, digits=1)) x")

stats = function (name, v)
    q50  = round(quantile(v, 0.50);  sigdigits=3)
    q99  = round(quantile(v, 0.99);  sigdigits=3)
    q999 = round(quantile(v, 0.999); sigdigits=3)
    vmax = round(maximum(v);         sigdigits=3)
    println("== $name ==")
    println("max    : ", vmax)
    println("p99.9  : ", q999, "   (99.9%)")
    println("p99    : ", q99,  "   (99%)")
    println("median : ", q50,  "   (50%)")
end

stats("rel μ", diff_μ)
stats("rel ζ", diff_ζ)