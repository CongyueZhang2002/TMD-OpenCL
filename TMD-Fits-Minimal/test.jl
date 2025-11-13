#################### Compute-bound (NO Threads; 16 workers + GPU) ####################
# Single-core CPU | Multi-core CPU via Distributed+SharedArrays (exactly 16 procs)
# GPU via OpenCL (Windows AMD driver; OpenCL-C kernel, no SPIR-V)

using Distributed
using Random
using Dates
using OpenCL
const CL = OpenCL.cl

# ---------------- knobs ----------------
const USE_GPU = true
const N       = 12_000_000      # quick to run; scale later
const ITERS   = 5
const REPEAT  = 128
const UNROLL  = 8              # 8 FMAs per inner step (compute heavy)

# ---------------- worker pool: exactly 16 ----------------
function ensure_workers(n::Int)
    cur = length(workers())
    if cur > n
        rmprocs(workers()[1:(cur-n)])
    elseif cur < n
        addprocs(n - cur)
    end
end
ensure_workers(16)             # << create/trim workers FIRST

# Only now load SharedArrays on ALL current workers
@everywhere using SharedArrays  # << this was the missing ordering

# ======== shared code on all procs ========
@everywhere @inline function fma8!(s::Float32, a::Float32, b::Float32)
    @inbounds begin
        s = muladd(s,a,b); s = muladd(s,a,b); s = muladd(s,a,b); s = muladd(s,a,b)
        s = muladd(s,a,b); s = muladd(s,a,b); s = muladd(s,a,b); s = muladd(s,a,b)
    end
    return s
end

@everywhere function compute_chunk_repeated!(z::AbstractVector{Float32},
                                            x::AbstractVector{Float32},
                                            r::UnitRange{Int},
                                            repeat::Int, iters::Int)
    a = 1.000001f0
    b = 1.0f-6
    @inbounds begin
        for _ in 1:iters
            for i in r
                s = x[i]
                for _ in 1:repeat
                    s = fma8!(s, a, b)   # 8 FMAs in registers (compute-bound)
                end
                z[i] = s
            end
        end
    end
    return nothing
end

function chunks_1d(N::Int, parts::Int)
    parts = max(parts, 1)
    base, rem = divrem(N, parts)
    ranges = UnitRange{Int}[]
    s = 1
    @inbounds for p in 1:parts
        len = base + (p <= rem ? 1 : 0)
        e = s + len - 1
        push!(ranges, s:e); s = e + 1
    end
    ranges
end

# ---------------- benches ----------------
function bench_cpu_single(xh::Vector{Float32}, iters::Int, repeat::Int)
    N = length(xh)
    z = similar(xh)
    a = 1.000001f0; b = 1.0f-6
    # tiny warm-up so JIT isn’t in timing
    @inbounds for i in 1:min(N, 1000)
        s = xh[i]; s = fma8!(s,a,b); z[i] = s
    end
    t = @elapsed begin
        for _ in 1:iters
            @inbounds for i in 1:N
                s = xh[i]
                for _ in 1:repeat; s = fma8!(s,a,b); end
                z[i] = s
            end
        end
    end
    (; t, per_iter=t/iters, z)
end

function bench_cpu_distributed(xh::Vector{Float32}, iters::Int, repeat::Int)
    N = length(xh)
    x = SharedArray{Float32}(N; pids=workers());  x[:] = xh
    z = SharedArray{Float32}(N; pids=workers())
    ranges = chunks_1d(N, nworkers())

    # warm-up: one quick local pass per worker (one task/worker)
    @sync for (pid, r) in zip(workers(), ranges)
        @spawnat pid compute_chunk_repeated!(z, x, r, 1, 1)
    end

    # timing: ONE task per worker; each loops locally 'iters' times
    t = @elapsed begin
        @sync for (pid, r) in zip(workers(), ranges)
            @spawnat pid compute_chunk_repeated!(z, x, r, repeat, iters)
        end
    end
    (; t, per_iter=t/iters, z)
end

const FMA_SRC = """
__kernel void fma_heavy(__global const float *x,
                        __global       float *z)
{
    int i = get_global_id(0);
    float s = x[i];
    const float a = 1.000001f;
    const float b = 1.0e-6f;
    for (int r = 0; r < REPEAT_PLACEHOLDER; ++r) {
        // 8 FMAs per loop (compute-bound, registers only)
        s = fma(s,a,b); s = fma(s,a,b); s = fma(s,a,b); s = fma(s,a,b);
        s = fma(s,a,b); s = fma(s,a,b); s = fma(s,a,b); s = fma(s,a,b);
    }
    z[i] = s;
}
""";

function bench_gpu_opencl(xh::Vector{Float32}, iters::Int, repeat::Int)
    len = length(xh)
    dx = CLArray(xh)
    dz = similar(dx)
    src  = replace(FMA_SRC, "REPEAT_PLACEHOLDER" => string(repeat))
    prog = CL.Program(; source=src) |> CL.build!
    kern = CL.Kernel(prog, "fma_heavy")

    # warm-up
    CL.clcall(kern, Tuple{Ptr{Float32}, Ptr{Float32}}, dx, dz; global_size=(len,))
    CL.finish(CL.queue())

    t = @elapsed begin
        for _ in 1:iters
            CL.clcall(kern, Tuple{Ptr{Float32}, Ptr{Float32}}, dx, dz; global_size=(len,))
        end
        CL.finish(CL.queue())
    end
    zgpu = Array(dz)
    (; t, per_iter=t/iters, z=zgpu)
end

# ---------------- orchestrator ----------------
function nowstr()  Dates.format(now(), DateFormat("HH:MM:SS"))  end

function main()
    println("Workers: ", workers())
    println("Config: N=$(N), iters=$(ITERS), repeat=$(REPEAT), unroll=$(UNROLL) (FMAs/elem = $(REPEAT*UNROLL))")
    println("[$(nowstr())] allocating & seeding…")
    Random.seed!(0)
    xh = rand(Float32, N)

    println("[$(nowstr())] single-core running…")
    sc = bench_cpu_single(xh, ITERS, REPEAT)
    println("  single-core per-iter = ", round(sc.per_iter, digits=6), " s")

    println("[$(nowstr())] distributed (16 workers) running…")
    dist = bench_cpu_distributed(xh, ITERS, REPEAT)
    println("  distributed per-iter = ", round(dist.per_iter, digits=6), " s")

    local gpu_ok = USE_GPU
    local gpu
    if USE_GPU
        println("[$(nowstr())] GPU (OpenCL) building/running…")
        try
            gpu = bench_gpu_opencl(xh, ITERS, REPEAT)
            println("  GPU per-iter = ", round(gpu.per_iter, digits=6), " s")
        catch e
            gpu_ok = false
            @warn "GPU run failed" e
        end
    else
        println("GPU step skipped (set USE_GPU = true to enable)")
    end

    # FLOPs accounting (each FMA = 2 FLOPs)
    flops_per_iter = 2.0 * REPEAT * UNROLL * N
    gflops_sc   = flops_per_iter / sc.per_iter   / 1e9
    gflops_dist = flops_per_iter / dist.per_iter / 1e9

    println("\n--- Compute-bound Summary (Float32) ---")
    println("Work per iter: ", round(flops_per_iter/1e9, digits=3), " GFLOP")
    println("Single-core      : ", round(sc.per_iter,   digits=6), " s  |  ", round(gflops_sc,   digits=1), " GFLOP/s")
    println("Distributed x16  : ", round(dist.per_iter, digits=6), " s  |  ", round(gflops_dist, digits=1), " GFLOP/s  |  speedup vs single ×", round(sc.per_iter/dist.per_iter, digits=2))

    if gpu_ok
        gflops_gpu = flops_per_iter / gpu.per_iter / 1e9
        maxerr = maximum(abs.(gpu.z .- dist.z))
        println("GPU (OpenCL)     : ", round(gpu.per_iter, digits=6), " s  |  ", round(gflops_gpu, digits=1), " GFLOP/s  |  speedup vs single ×", round(sc.per_iter/gpu.per_iter, digits=2))
        println("max |CPU(dist) - GPU| = ", maxerr)
    end
    nothing
end

main()
