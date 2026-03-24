using CSV, DataFrames, Interpolations, StaticArrays

if !isdefined(Main, :interpolators)
    const interpolators = Dict{Symbol,Any}()
else
    empty!(interpolators)
end

function initialize_interpolator(; df, interpolator_name, variable_names, target_names)
    # axes & values in Float32 → less memory traffic at query time
    variable_axes = map(v -> Float32.(sort(unique(df[!, v]))), variable_names)
    shape = map(length, variable_axes)
    N_targets = length(target_names)

    target_vec = Array{SVector{N_targets,Float32}}(undef, shape...)
    for row in eachrow(df)
        grid_index = ntuple(d -> searchsortedfirst(variable_axes[d], Float32(row[variable_names[d]])),
                            length(variable_names))
        target_vec[grid_index...] = SVector{N_targets,Float32}((Float32(row[t]) for t in target_names)...)
    end

    vector_interpolant = interpolate(Tuple(variable_axes), target_vec, Gridded(Linear()))
    interpolators[Symbol(interpolator_name)] = vector_interpolant
    return nothing
end


# Example
# evaluate_interpolator(name::Symbol, coords...) = interpolators[name](coords...)
