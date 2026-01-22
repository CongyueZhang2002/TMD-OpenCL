#----------------------------------------------------------------------------
# NP
#----------------------------------------------------------------------------

const flavor_scheme = "FI"
const NP_name = "NP-0113.cl"

struct Params_Struct
    g2::Float32
    bmax_CS::Float32
    g3::Float32
    m::Float32
end

# initial guesses (conservative, stable)
initial_params = [
    0.337,   # g2

    1.5,     # bmax_CS  (GeV^-1)
    0.0,     # g3
    2.0      # m
]

# bounds in the order above
bounds_raw = [
    (0.0, 1.0),      # g2

    (0.5, 5.0),      # bmax_CS
    (0.0, 1.0),      # g3
    (1.0, 6.0)       # m
]
#----------------------------------------------------------------------------
# PDF
#----------------------------------------------------------------------------

const table_name = "CT18ZNNLO-FI"
const pdf_name = "CT18ZNNLO"
const error_sets_name = "CT18ZNNLO-ES"

#----------------------------------------------------------------------------
# Data Set
#----------------------------------------------------------------------------

const data_name = "Default"