using Serialization, OpenCL, ProgressBars
const cl = OpenCL.cl

#---------------------------------------------------------------------------------------------------------------------
#   Parameters
#---------------------------------------------------------------------------------------------------------------------

make_params(v) = Params_Struct(Float32.(v)...)
Params = make_params(initial_params)

#---------------------------------------------------------------------------------------------------------------------
#   Build OpenCL program
#---------------------------------------------------------------------------------------------------------------------

source_NP = read(abspath(joinpath(@__DIR__, "..", "TMDs", "NP Parameterizations", "$NP_name")), String)
source_DY = read(abspath(joinpath(@__DIR__, "DY_table_FD.cl")), String)
programs = cl.Program(; source = string(source_NP, "\n\n", source_DY)) |>
        p -> cl.build!(p; options="-cl-fast-relaxed-math -cl-mad-enable -cl-std=CL2.0")

#---------------------------------------------------------------------------------------------------------------------
#   Load all tables 
#---------------------------------------------------------------------------------------------------------------------

# Tuple{Vector{Tuple{Tuple{xp,xN,Q},Matrix{b,pert}}},isoscalarity}
const table_type = Tuple{Vector{Tuple{Tuple{Float32,Float32,Float32},Matrix{Float32}}},Float64}
const tables = Dict{String, table_type}()
function load_table(rel_path::AbstractString)
    tables[rel_path] = open(isabspath(rel_path) ? rel_path : joinpath(@__DIR__, rel_path), "r") do io
        deserialize(io)::table_type
    end
    nothing
end

root = abspath(joinpath(@__DIR__, "../Tables/$table_name/DY"))
all_paths = sort([joinpath(d, f) for (d,_,fs) in walkdir(root) for f in fs if endswith(f, ".jls")])
rel_paths = [relpath(p, @__DIR__) for p in all_paths]
foreach(load_table, rel_paths)

#---------------------------------------------------------------------------------------------------------------------
#   Tables to vectors
#---------------------------------------------------------------------------------------------------------------------

table_to_vectors(blocks) = begin
    total = sum(size(matrix,1) for (_,matrix) in blocks)

    xp_vec = Vector{Float32}(undef, total)
    xN_vec = similar(xp_vec) 
    Q_vec = similar(xp_vec)
    b_vec = similar(xp_vec)

    pert_up_uN_vec = similar(xp_vec)
    pert_up_dN_vec = similar(xp_vec)
    pert_dp_uN_vec = similar(xp_vec)
    pert_dp_dN_vec = similar(xp_vec)
    pert_ubp_ubN_vec = similar(xp_vec)
    pert_ubp_dbN_vec = similar(xp_vec)
    pert_dbp_ubN_vec = similar(xp_vec)
    pert_dbp_dbN_vec = similar(xp_vec)
    pert_sea_vec = similar(xp_vec)

    id = 1
    @inbounds for ((xp,xN,Q), matrix) in blocks
        n = size(matrix,1)
        @views begin
            xp_vec[id:id+n-1] .= xp   
            xN_vec[id:id+n-1] .= xN
            Q_vec[id:id+n-1] .= Q
            b_vec[id:id+n-1] .= matrix[:,end]

            pert_up_uN_vec[id:id+n-1] .= matrix[:,1]
            pert_up_dN_vec[id:id+n-1] .= matrix[:,2]
            pert_dp_uN_vec[id:id+n-1] .= matrix[:,3]
            pert_dp_dN_vec[id:id+n-1] .= matrix[:,4]
            pert_ubp_ubN_vec[id:id+n-1] .= matrix[:,5]
            pert_ubp_dbN_vec[id:id+n-1] .= matrix[:,6]
            pert_dbp_ubN_vec[id:id+n-1] .= matrix[:,7]
            pert_dbp_dbN_vec[id:id+n-1] .= matrix[:,8]
            pert_sea_vec[id:id+n-1] .= matrix[:,9]
        end
        id += n
    end
    return (
        xp_vec, xN_vec, Q_vec, b_vec, 

        pert_up_uN_vec, 
        pert_up_dN_vec, 
        pert_dp_uN_vec, 
        pert_dp_dN_vec, 
        pert_ubp_ubN_vec, 
        pert_ubp_dbN_vec, 
        pert_dbp_ubN_vec, 
        pert_dbp_dbN_vec, 
        pert_sea_vec
    )
end

#---------------------------------------------------------------------------------------------------------------------
#   Load vectors to VRAM
#---------------------------------------------------------------------------------------------------------------------

const VRAM_Struct = NamedTuple{(
    :xp_vec, :xN_vec, :Q_vec, :b_vec, 

    :pert_up_uN_vec, 
    :pert_up_dN_vec, 
    :pert_dp_uN_vec, 
    :pert_dp_dN_vec, 
    :pert_ubp_ubN_vec, 
    :pert_ubp_dbN_vec, 
    :pert_dbp_ubN_vec, 
    :pert_dbp_dbN_vec, 
    :pert_sea_vec,

    :isoscalarity,
    
    :xsec, :params, :dim
),
    Tuple{
        CLArray{Float32,1},
        CLArray{Float32,1},
        CLArray{Float32,1},
        CLArray{Float32,1},

        CLArray{Float32,1},
        CLArray{Float32,1},
        CLArray{Float32,1},
        CLArray{Float32,1},
        CLArray{Float32,1},
        CLArray{Float32,1},
        CLArray{Float32,1},
        CLArray{Float32,1},
        CLArray{Float32,1},

        Float32,

        CLArray{Float32,1},
        CLArray{Params_Struct,1},
        Int32
    }
}

function Make_VRAM(path, params::Params_Struct)::VRAM_Struct

    blocks = tables[path][1]

    (
        xp_vec, xN_vec, Q_vec, b_vec,

        pert_up_uN_vec, 
        pert_up_dN_vec, 
        pert_dp_uN_vec, 
        pert_dp_dN_vec, 
        pert_ubp_ubN_vec, 
        pert_ubp_dbN_vec, 
        pert_dbp_ubN_vec, 
        pert_dbp_dbN_vec, 
        pert_sea_vec
    ) = table_to_vectors(blocks)

    n = length(b_vec); 
    dim = Int32(n)

    (  
        xp_vec = CLArray(xp_vec),   
        xN_vec = CLArray(xN_vec), 
        Q_vec = CLArray(Q_vec),
        b_vec = CLArray(b_vec),

        pert_up_uN_vec = CLArray(pert_up_uN_vec),
        pert_up_dN_vec = CLArray(pert_up_dN_vec),
        pert_dp_uN_vec = CLArray(pert_dp_uN_vec),
        pert_dp_dN_vec = CLArray(pert_dp_dN_vec),
        pert_ubp_ubN_vec = CLArray(pert_ubp_ubN_vec),
        pert_ubp_dbN_vec = CLArray(pert_ubp_dbN_vec),
        pert_dbp_ubN_vec = CLArray(pert_dbp_ubN_vec),
        pert_dbp_dbN_vec = CLArray(pert_dbp_dbN_vec),
        pert_sea_vec = CLArray(pert_sea_vec),

        isoscalarity = Float32(tables[path][2]),

        xsec = CLArray([0f0]),
        params = CLArray([params]),                    
        dim = dim 
    )
end

VRAM = [Make_VRAM(path, Params) for path in ProgressBar(rel_paths)]

#---------------------------------------------------------------------------------------------------------------------
#   Update parameters
#---------------------------------------------------------------------------------------------------------------------

function set_params(VRAM::Vector{VRAM_Struct}, params_new::Params_Struct)
    @inbounds for buffer in VRAM
        copyto!(buffer.params, (params_new,))                
    end
    nothing
end

set_params(VRAM, Params) 

#---------------------------------------------------------------------------------------------------------------------
#   Xsec
#---------------------------------------------------------------------------------------------------------------------

function xsec_dict(rel_paths::Vector{String}, VRAM::Vector{VRAM_Struct};
                   local_size::Int=256, n_groups::Int=64)
                   
    @assert length(rel_paths) == length(VRAM)
    DY_xsec = cl.Kernel(programs, "DY_xsec")
    lmem = cl.LocalMem(Float32, local_size)

    t_compute = @elapsed begin

        @inbounds for buffer in VRAM
            fill!(buffer.xsec, 0f0)
            n  = Int(buffer.dim)
            ng = min(n_groups, cld(n, local_size))

            cl.clcall(DY_xsec,
                Tuple{
                    CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32},

                    CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32},
                    CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32}, CLPtr{Float32},
                    CLPtr{Float32}, 

                    Float32,

                    CLPtr{Params_Struct},Int32,CLPtr{Float32},typeof(lmem)
                    },
                buffer.xp_vec, buffer.xN_vec, buffer.Q_vec, buffer.b_vec,

                buffer.pert_up_uN_vec, 
                buffer.pert_up_dN_vec,
                buffer.pert_dp_uN_vec,
                buffer.pert_dp_dN_vec,
                buffer.pert_ubp_ubN_vec,
                buffer.pert_ubp_dbN_vec,
                buffer.pert_dbp_ubN_vec,
                buffer.pert_dbp_dbN_vec,
                buffer.pert_sea_vec,
                
                buffer.isoscalarity,

                buffer.params, buffer.dim, buffer.xsec, lmem;
                global_size=(ng*local_size,), local_size=(local_size,)
            )

        end
        cl.finish(cl.queue())
    end

    xsecs = map(buffer -> Array(buffer.xsec)[1], VRAM)
    return Dict(rel_paths[i] => xsecs[i] for i in eachindex(rel_paths)), t_compute
end