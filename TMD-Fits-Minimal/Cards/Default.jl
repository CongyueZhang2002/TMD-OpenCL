#----------------------------------------------------------------------------
# NP
#----------------------------------------------------------------------------

const flavor_scheme = "FI"
const NP_name = "NP-0112.cl"

struct Params_Struct
    g2::Float32
    bmax_CS::Float32
    power_CS::Float32
    a1::Float32; a2::Float32; a3::Float32; a4::Float32
    b1::Float32; b2::Float32; b3::Float32
    a::Float32
end

# initial guesses (conservative, stable)
initial_params = [
    0.5,   # g2

    1.123,     # bmax_CS  (GeV^-1)
    1.0,     # power_CS 

    0.0,     # a1 (GeV)
    0.0,     # a2 (GeV)
    0.0,     # a3 (GeV)
    0.0,     # a4 (GeV)   (log(x) term; keep small initially)

    0.0,     # b1 (dimensionless)
    0.0,     # b2 (dimensionless)
    0.0,     # b3 (dimensionless)

    0.0      # a  (alpha; a=1 => bstar=b)
]
initial_params = [0.511, 1.123, 0.908, 2.97, -0.0553, 0.0, 0.0, -3.55, 0.168, 0.0, 0.0]
#initial_params = [0.5, 1.123, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# bounds in the order above
bounds_raw = [
    (0.0, 1.0),      # g2

    (0.5, 5.0),      # bmax_CS
    (0.0, 2.0),      # power_CS

    (-10.0, 10.0),     # a1
    (-10.0, 10.0),     # a2
    (-10.0, 10.0),     # a3
    (-0.5, 0.5),     # a4   (tighter because log(x) can be large)

    (-10.0, 10.0),     # b1
    (-10.0, 10.0),     # b2
    (-10.0, 10.0),     # b3

    (0.0, 2.0)       # a
]

#initial_params = [0.477, 1.48, 0.401, -1.16, -0.644, 0.308, -0.11, 2.08, -2.63, 6.61, 1.46]

frozen_indices = [5,6,9,10]#[3,4,5,6,7,8,9,10]
#----------------------------------------------------------------------------
# PDF
#----------------------------------------------------------------------------

#const table_name = "HERA20-CS"
#const pdf_name = "approximate"
#const error_sets_name = "HERA20-ES"

const table_name = "MSHT20N3LO-MC-0-2"
const pdf_name = "approximate"
const error_sets_name = "MSHT20N3LO-MC"

#----------------------------------------------------------------------------
# Data Set
#----------------------------------------------------------------------------

const data_name = "Default"