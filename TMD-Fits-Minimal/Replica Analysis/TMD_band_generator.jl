function TMDPDF_kt_vec(; kt_vec::AbstractVector, x::Float64, Q::Float64)
    out = Matrix{Float64}(undef, 10, length(kt_vec))
    for (i, kt) in pairs(kt_vec)
        @inbounds out[:, i] .= TMDPDF_kt_func(kt=Float64(kt), x=x, Q=Q)
    end
    return out
end
