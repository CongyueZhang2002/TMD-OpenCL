using Serialization, OpenCL
const cl = OpenCL.cl
println("OpenCL device: ", cl.device())
const CTX = cl.context(); const Q = cl.queue()

# --- discover all .jls files ---
root = abspath(joinpath(@__DIR__, "../Tables/Default/DY"))
paths = String[]
for (d,_,fs) in walkdir(root); for f in fs; endswith(f,".jls") && push!(paths, joinpath(d,f)); end; end
sort!(paths)

# --- tiny helpers (classic buffers + raw map/unmap copy) ---
d_alloc(nbytes::Int) = begin
    err = Ref{Int32}(0)
    ptr = cl.clCreateBuffer(CTX, UInt64(cl.CL_MEM_READ_WRITE), nbytes, C_NULL, err)
    err[] == cl.CL_SUCCESS || error("clCreateBuffer err=$(err[])")
    cl.Buffer(ptr, nothing, nbytes, CTX)
end

function h2d!(buf::cl.Buffer, h::Vector{Float32})
    nbytes = length(h)*sizeof(Float32); err = Ref{Int32}(0)
    p = cl.clEnqueueMapBuffer(Q, buf, cl.CL_TRUE, UInt64(cl.CL_MAP_WRITE), 0, nbytes, 0, C_NULL, C_NULL, err)
    err[] == cl.CL_SUCCESS || error("map err=$(err[])")
    GC.@preserve h unsafe_copyto!(Ptr{UInt8}(p), Base.unsafe_convert(Ptr{UInt8}, pointer(h)), nbytes)
    _ = cl.clEnqueueUnmapMemObject(Q, buf, p, 0, C_NULL, C_NULL); cl.finish(Q)
end

# --- flatten + upload each table; force residency with device copies ---
DEV = Any[]  # just to keep buffers alive & compute a size estimate
for p in paths
    blocks = open(p, "r") do io
        (deserialize(io)::Tuple{Vector{Tuple{Tuple{Float32,Float32,Float32},Matrix{Float32}}},Float64})[1]
    end
    total = sum(size(m,1) for (_,m) in blocks)
    xp = Vector{Float32}(undef,total); xN=similar(xp); Qv=similar(xp); pert=similar(xp); b=similar(xp)
    idx = 1
    @inbounds for ((xp0,xN0,Q0), m) in blocks
        n = size(m,1); @views begin
            xp[idx:idx+n-1].=xp0; xN[idx:idx+n-1].=xN0; Qv[idx:idx+n-1].=Q0
            pert[idx:idx+n-1].=m[:,1]; b[idx:idx+n-1].=m[:,2]
        end; idx += n
    end

    bytes = total*sizeof(Float32)
    d_pert = d_alloc(bytes); h2d!(d_pert, pert)
    d_b    = d_alloc(bytes); h2d!(d_b,    b)
    d_xp   = d_alloc(bytes); h2d!(d_xp,   xp)
    d_xN   = d_alloc(bytes); h2d!(d_xN,   xN)
    d_Q    = d_alloc(bytes); h2d!(d_Q,    Qv)
    d_out  = d_alloc(bytes)
    d_xsec = d_alloc(sizeof(Float32)); h2d!(d_xsec, Float32[0])

    # force GPU activity so driver commits to VRAM
    for s in (d_pert,d_b,d_xp,d_xN,d_Q); _ = cl.clEnqueueCopyBuffer(Q, s, d_out, 0, 0, bytes, 0, C_NULL, C_NULL); end
    cl.finish(Q)

    push!(DEV, (; d_pert,d_b,d_xp,d_xN,d_Q,d_out,d_xsec, dim=Int32(total)))
end

# --- report and keep allocations alive for overlay ---
GC.@preserve DEV sleep(10)