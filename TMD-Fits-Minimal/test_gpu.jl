using OpenCL

# Grab whatever default queue you're already using
q = OpenCL.queue()

# Get its device (fallback through context->devices if needed)
dev = try
    OpenCL.device(q)
catch
    OpenCL.devices(OpenCL.context(q))[1]
end

println("Device version   : ", OpenCL.device_info(dev, :VERSION))          # e.g. "OpenCL 3.0 ..."
println("OpenCL C version : ", OpenCL.device_info(dev, :OPENCL_C_VERSION)) # e.g. "OpenCL C 1.2"
exts = String(OpenCL.device_info(dev, :EXTENSIONS))
println("Has subgroups?   : ", occursin("cl_khr_subgroups", exts))
println("Float atomics?   : ",
        occursin("cl_khr_global_float_atomics", exts) ||
        occursin("cl_khr_local_float_atomics",  exts) ||
        occursin("cl_ext_float_atomics",        exts))
