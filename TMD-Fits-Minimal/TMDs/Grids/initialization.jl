include("interpolation.jl")

if if_grid == true
    read_csv(path) = DataFrame(CSV.File(joinpath(@__DIR__, path)))
    df_TMDPDF      = read_csv("$table_name/TMDPDF.csv")
    df_PertSudakov = read_csv("$table_name/PertSudakov.csv")
    SNP_path       = joinpath(@__DIR__, "$table_name/SNP.csv")
    has_SNP_grid   = isfile(SNP_path)
    if has_SNP_grid
        df_SNP = read_csv("$table_name/SNP.csv")
    end

    global TMDPDF_bmin = Float64(minimum(df_TMDPDF[!, "b"]))
    global TMDPDF_bmax = Float64(maximum(df_TMDPDF[!, "b"]))
    global PertSudakov_bmin = Float64(minimum(df_PertSudakov[!, "b"]))
    global PertSudakov_bmax = Float64(maximum(df_PertSudakov[!, "b"]))

    # 1) build interpolators (populates `interpolators`)
    initialize_interpolator(
        df = df_TMDPDF,
        interpolator_name = "xTMDPDF_raw_grid",
        variable_names = ["x","b"],
        target_names = ["f_u","f_ub","f_d","f_db","f_s","f_sb","f_c","f_cb","f_b","f_bb"],
    )
    initialize_interpolator(
        df = df_PertSudakov,
        interpolator_name = "PertSudakov_grid",
        variable_names = ["b","Q"],
        target_names = ["SP"],
    )

    if has_SNP_grid && flavor_scheme == "FI"
        initialize_interpolator(
            df = df_SNP,
            interpolator_name = "NP_f_grid",
            variable_names = ["x","b"],
            target_names = ["SNP_μ", "SNP_ζ"],
        )
    elseif has_SNP_grid && flavor_scheme == "FD"
        initialize_interpolator(
            df = df_SNP,
            interpolator_name = "NP_f_grid",
            variable_names = ["x","b"],
            target_names = ["SNP_u","SNP_ub","SNP_d","SNP_db","SNP_sea","SNP_ζ"],
        )
    end

    # xTMDPDF_raw_grid
    let itp = interpolators[:xTMDPDF_raw_grid]
        global xTMDPDF_raw_grid
        @inline xTMDPDF_raw_grid(x::Real, b::Real) = itp(x, b)
    end

    # PertSudakov_grid
    let itp = interpolators[:PertSudakov_grid]
        global PertSudakov_grid
        @inline PertSudakov_grid(b::Real, Q::Real) = itp(b, Q)[1]
    end

    # NP_f_grid

    if has_SNP_grid
        let itp = interpolators[:NP_f_grid]
            global NP_f_grid
            @inline NP_f_grid(x::Real, b::Real) = itp(x, b)
        end
    end

end
