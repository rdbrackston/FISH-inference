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
Generate a normalized pdf from continuous data.
"""
function genkde(smpls)

    data = deepcopy(smpls)
    filter!(x -> !isnan(x), data)    # Remove nans from data

    kdeObj = KDE.kde(data)
    x = collect(range(minimum(data),stop=maximum(data),length=100))
    y = map(z->KDE.pdf(kdeObj,z),x)

    return (x,y)

end


"""
Generate a normalized pdf using a transform-retransform approach.
"""
function genkde_trans(smpls, Tfunc=:log::Symbol)

    if isequal(Tfunc,:log)
        T = z->log(z+1.)
        dT = z->1/(z+1.)
    elseif isequal(Tfunc, :arctan)
        T = z->2*atan(z)/pi
        dT = z->2/(pi*(z^2+1.))
    end

    X = deepcopy(smpls)
    filter!(z -> !isnan(z), X)    # Remove nans from data
    Y = T.(X)
    # Y = log.(X .+ 1.)

    # Perform KDE on log-transformed data
    kdeObj = KDE.kde(Y)
    x = collect(range(0.0, stop=1.05*maximum(X), length=1000))
    y = T.(x)
    # y = log.(x .+ 1.)
    g = map(z->KDE.pdf(kdeObj,z),y)

    # Retransform
    f = g.*dT.(x)
    # f = g./(1. .+ x)

    return (x,f)

end


"""
Wrapper around the KDE function to ensure that all calls to KDE use the same settings
"""
function kde_wrpr(data)
    KDE.kde(data, boundary=(0.0,2*max(data...)))
end


"""
Fraction of rising factorials, as used in a number of other functions.
"""
function fracrise(x, y, r)
	Q = 1
	for m=0:r-1
		Q *= (x + m) / (y + m)
	end
	return Q
end
