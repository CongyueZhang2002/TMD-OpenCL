#!/usr/bin/env julia
# DY/test_allgpu_time.jl
# Time the all-GPU path (flatten -> dy_np_rows -> seg_reduce) over all DY tables.

using Printf, Statistics
include(joinpath(@__DIR__, "DY_table.jl"))
using .DY_Table

# ---------------- CLI ----------------
# Usage: julia test_allgpu_time.jl [fit_name] [n_iter]
const FIT_NAME = length(ARGS) >= 1 ? ARGS[1] : "Default"
const N_ITER   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5

# Example NP params (12 floats)
const PARAMS = DY_Table.Params(
    0.232f0, 1.1f0, 0.0194f0,
    0.171f0, 0.115f0, 0.00894f0,
    1.83f0,  3.42f0,  2.34f0,
    1.27f0,  0.0949f0, 7.82f0
)

function main()
    tables_root = joinpath(@__DIR__, "..", "Tables", FIT_NAME, "DY")
    cl_path     = joinpath(@__DIR__, "DY_table.cl")

    println("=== DY all-GPU timing ===")
    @printf("fit_name     : %s\n", FIT_NAME)
    @printf("tables_root  : %s\n", abspath(tables_root))
    @printf("cl source    : %s\n", abspath(cl_path))

    # one-time setup (discover + upload all .jls)
    DY_Table.DY_setup_from_root(tables_root; cl_source_path=cl_path)

    # warm-up
    @info "Warm-up evaluate…"
    preds, t0 = DY_Table.DY_predict_all(PARAMS)
    @printf("warm-up: %.6f s   n_tables=%d   sum(xsec)=%.6e\n",
            t0, length(preds), sum(values(preds)))

    # timed iterations
    times = Float64[]
    @info "Running $N_ITER evaluations…"
    for it in 1:N_ITER
        p, t = DY_Table.DY_predict_all(PARAMS)
        push!(times, t)
        @printf("iter %2d  time = %.6f s   sum(xsec)=%.6e\n",
                it, t, sum(values(p)))
    end

    @printf("\nSummary over %d runs:\n", N_ITER)
    @printf("  mean = %.6f s   min = %.6f s   max = %.6f s\n",
            mean(times), minimum(times), maximum(times))

    # show a few outputs
    println("\n--- sample outputs (first 5 tables) ---")
    i = 0
    for (k, v) in preds
        @printf("  %-70s  %.6e\n", k, v)
        i += 1
        i >= 5 && break
    end
end

main()
