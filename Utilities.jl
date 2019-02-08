# Various utility functions used elsewhere in module

"""
Generate a normalized pdf from continuous data
"""
function genpdf(data::Array{Float64,1}, nbin=:auto::Union{Int,Symbol})

    filter!(x -> !isnan(x), data)    # Remove nans from data

    edges = binedges(DiscretizeUniformWidth(nbin), data)
    disc = LinearDiscretizer(edges)
    counts = get_discretization_counts(disc, data)

    x = edges[1:end-1]+0.5*diff(edges)
    y = (counts./(diff(edges)*float(size(data)[1]))) # Normalise
    return (x,y)

end


"""
1D integer version of genpdf, intended for use when the underlying data is discrete.
"""
function genpdf(data::Array{Int64,1}, nbin=:auto::Union{Int,Symbol})

    # Set bins to all the integers between lo and hi
    lo, hi = extrema(data)
    edges = collect(lo-1:hi) .+ 0.5

    disc = LinearDiscretizer(edges)
    counts = get_discretization_counts(disc, data)

    x = edges[1:end-1]+0.5*diff(edges)
    y = (counts./(diff(edges)*float(size(data)[1]))) # Normalise
    return (x,y)

end


"""
Function to compute the hypergeometric function using a Maclaurin series.
Adapted from https://github.com/JuliaApproximation/SingularIntegralEquations.jl/src/HypergeometricFunctions/specialfunctions.jl
Unused, as generally inferior to the GSL implemented version.
"""
function hypergeom1F1(a::Float64,b::Float64,z::Float64)
    
    S₀,S₁,err,j = 1.0,1.0+a*z/b,1.0,1
    while err > 1eps(Float64)
        rⱼ = inv(j+1.0)
        rⱼ *= a+j
        rⱼ /= b+j
        S₀,S₁ = S₁,S₁+(S₁-S₀)*rⱼ*z
        err = abs((S₁-S₀)/S₀)
        j+=1
    end
    return S₁
end