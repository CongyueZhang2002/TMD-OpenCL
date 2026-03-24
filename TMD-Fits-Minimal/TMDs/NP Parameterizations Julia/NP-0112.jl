const bmax = b0

function bstar_func(b)

    bstar = b/(1+(b/bmax)^4)^(1/4)
    #bstar = b

    return bstar
end

function μstar_func(b)

    bstar = bstar_func(b)

    return b0/bstar
    #return max(b0/bstar, 1.0)
end

function bstar_CS_func(b)

    bstar = b/(1+(b/bmax)^4)^(1/4)

    return bstar
end

function μstar_CS_func(b)

    bstar = bstar_CS_func(b)

    return b0/bstar
    #return max(b0/bstar, 1.0)
end

@inline exp32(x::Float32) = exp(x)

function NP_f_func(x_64::Float64, b_64::Float64)

    g2 = Float32(NP_g2)
    bmax_CS = Float32(NP_bmax_CS)
    power_CS = Float32(NP_power_CS)

    a1 = Float32(NP_a1)
    a2 = Float32(NP_a2)
    a3 = Float32(NP_a3)
    a4 = Float32(NP_a4)

    b1 = Float32(NP_b1)
    b2 = Float32(NP_b2)
    b3 = Float32(NP_b3)
    a = Float32(NP_a)

    x = clamp(Float32(x_64), 1f-7, 1f0-1f-7)
    b = Float32(b_64)

    xshape = a1*x + a2*(1-x) + a3*x*(1-x) + a4*log(x)
    bshape = exp(Float64(b1*x^2 + b2*(1-x)^2 + 2*b3*x*(1-x)))
    bstar = b*(1+(b/(b0*bshape))^4)^((a-1)/4)    

    Sudakov = sech(Float64(xshape*bstar))

    bstar_CS = b*(1f0 + (b/bmax_CS)^4)^((power_CS-1f0)/4f0)
    gK = -g2^2*bstar_CS*bstar_CS/2  

    SNP_μ = Float32(Sudakov)
    SNP_ζ = (gK/2)

    return SNP_μ, SNP_ζ
end

function CS_total_func(b, Q)

    g2 = Float32(NP_g2)

    integrand(μ) = Γ_func(μ=μ, order=3)/μ
    μi = b0/bstar_func(b=b, Q=Q)

    CS_evolution = quadgk(integrand, μi, Q, rtol=rtol)[1]
    CS_boundary = CS_func(μ=μi, order=3)
    gK = -g2^2*b^2/2
    CS = 2*(-CS_evolution - CS_boundary + gK/2)

    return CS
end
