const bmax = b0

function μstar_func(b)

    t = b / b0
    t4 = t^4
    denom = sqrt(sqrt(1 + t4))
    bstar = b / denom

    return max(b0 / bstar, 1.0)
end

@inline exp32(x::Float32) = exp(x)

function DNP_42best_func(b_64::Float64)

    BNP = Float32(NP_BNP)
    c0 = Float32(NP_c0)
    c1 = Float32(NP_c1)

    b = Float32(b_64)
    v = b / max(BNP, 1f-6)
    bstar = b / sqrt(1f0 + v^2)
    log_ratio = log(max(bstar / max(BNP, 1f-6), 1f-7))

    return b * bstar * (c0 + c1 * log_ratio)
end

function NP_f_func(x_64::Float64, b_64::Float64)

    lambda1 = Float32(NP_lambda1)
    lambda2 = Float32(NP_lambda2)
    lambda3 = Float32(NP_lambda3)
    lambda4 = Float32(NP_lambda4)
    alpha = Float32(NP_alpha)
    logx0 = Float32(NP_logx0)
    sigx = Float32(NP_sigx)
    amp = Float32(NP_amp)

    x = clamp(Float32(x_64), 1f-7, 1f0 - 1f-7)
    b = Float32(b_64)

    xbar = 1f0 - x
    xxbar = x * xbar
    base = lambda1 * xbar + lambda2 * x + lambda3 * xxbar + lambda4 * log(x)

    u = (log(x) - logx0) / max(sigx, 1f-6)
    bump = amp * exp32(-0.5f0 * u^2)
    shape = base + bump

    t = b / b0
    t4 = t^4
    bstar_μ = b * (1f0 + t4)^((alpha - 1f0) / 4f0)

    SNP_μ = Float32(sech(Float64(shape * bstar_μ)))
    SNP_ζ = -Float32(DNP_42best_func(b_64))

    return SNP_μ, SNP_ζ
end
