using QuadGK
using PolyLog
using SpecialFunctions
using StaticArrays
using .FastGK

function xTMDPDF_raw_func(x, b)
    return xTMDPDF_raw_grid(x,b)
end

function PertSudakov_func(b, Q)
    return PertSudakov_grid(b,Q)
end

function TMDPDF_pert_func(; b::Float64, x::Float64, Q::Float64)

    xTMDPDF_raw = xTMDPDF_raw_func(x, b)

    SP = PertSudakov_func(b, Q)

    return xTMDPDF_raw .* (SP/x)
end

function TMDPDF_func(; b::Float64, x::Float64, Q::Float64)

    TMDPDF_pert = TMDPDF_pert_func(b = b, x = x, Q = Q)
    SNP_μ, SNP_ζ = NP_f_func(x, b)

    log_ζ = 2 * log(Q / μstar_func(b))
    NP_factor = SNP_μ * exp(log_ζ * SNP_ζ)

    return TMDPDF_pert .* NP_factor
end

function TMD_per_nucleon_func(fu, fub, fd, fdb, isoscalarity)  

    if isoscalarity == 1.0
        return fu, fub, fd, fdb
    else
        ZdA = abs(isoscalarity)
        NdA = 1 - ZdA

        fu_mix = ZdA*fu+NdA*fd
        fub_mix = ZdA*fub+NdA*fdb
        fd_mix = ZdA*fd+NdA*fu
        fdb_mix = ZdA*fdb+NdA*fub               

        if isoscalarity > 0.0
            return fu_mix, fub_mix, fd_mix, fdb_mix
        elseif isoscalarity < 0.0
            return fub_mix, fu_mix, fdb_mix, fd_mix
        else
            return error("For neutron/anti-neutron pass a very small non-zero isoscalarity, like eps = 1e-6/-1e-6")
        end
    end 
end

function TMDPDF_kt_func(; kt::Float64, x::Float64, Q::Float64)
    integrand(b) = begin
        vals = TMDPDF_func(b=b, x=x, Q=Q)  
        s = b * besselj0(b*kt)/(2*π)
        s .* SVector{10,Float64}(vals)      
    end
    b_lo = isdefined(Main, :TMDPDF_bmin) ? max(1e-3, Float64(TMDPDF_bmin)) : 1e-3
    b_hi_tmd = isdefined(Main, :TMDPDF_bmax) ? Float64(TMDPDF_bmax) : 30.0
    b_hi_sp = isdefined(Main, :PertSudakov_bmax) ? Float64(PertSudakov_bmax) : b_hi_tmd
    b_hi = min(b_hi_tmd, b_hi_sp)
    out, _ = quadgk(integrand, b_lo, b_hi, rtol=1e-3)
    return out
end
