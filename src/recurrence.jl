# Functions for evaluating the leaky gene distribution using Lucy's recurrence relations

"""
Compute leaky gene distribution via the recurrence method.
Compute up to copy number N with M terms in the recursion
"""
function solvemaster_rec(prms, N, M)

	G = genfunc_twostate(prms, M)
	return [invgenfunc(G, n) for n=0:(N-1)]

end

"""
Evaluate the generating function G(z)
"""
function genfunc_twostate(prms, M)

	# Set up parameters
	λ = BigFloat(prms[3])
	ν = BigFloat(prms[4])
	K₀ = BigFloat(prms[1])
	K₁ = BigFloat(prms[2])
    # δ = BigFloat(1.0)

    function gg0(n, x, y) #calculate interates for g_0
	    a = BigFloat((((n + ν) * (n + λ)) / ν) - λ)
	    b = BigFloat((n + ν) / ν)
	    return BigFloat((1 / a) * ((b * K₀ * x) + (K₁ * y)))
	end
	function gg1(n, x, y) #calculate interates for g_1
	    c = BigFloat((((n + ν) * (n + λ)) / λ) - ν)
	    d = BigFloat((n + λ) / λ)
	    return BigFloat((1 / c) * ((d * K₁ * y) + (K₀ * x)))
	end

	LL = Array{BigFloat}(undef, M)
	KK = Array{BigFloat}(undef, M)
	LL[1] = BigFloat(ν / (λ + ν)) #Initial condition for g_0
	KK[1] = BigFloat(λ / (λ + ν)) #Initial condition for g_1
	for n in 2:M
	    LL[n] = gg0(n-1, LL[n-1], KK[n-1]) #calculates iterates for g0
	    KK[n] = gg1(n-1, LL[n-1], KK[n-1]) #calculates iterates for g1
	end
	G = LL .+ KK

	return G

end


"""
Back-calculate the distribution from the generating function
"""
function invgenfunc(G, n)

    M = length(G)
    s = BigFloat(0)
    fac = factorial(big(n))
    ds = fac*G[n+1]
    s += ds

    for k in 1:M-n-1
        i = n + k
        ds = G[i + 1] * (-1)^(k)
        fac *= big(k+n)/big(k)
        ds *= fac
        abs(ds/s)<0.001 && break
        s += ds
    end
    return Float64(s/factorial(big(n)))

end


"""
Evaluate the generating function for a compound distribution
"""
function genfunc_compound(parameters::AbstractArray, hyperParameters::AbstractArray,
                       distFunc::Symbol, M::Integer, parIndex=1; lTheta::Integer=200,
                       cdfMax::AbstractFloat=0.999)

	# Choose mixing distribution. The mean and variance are identical from case to case.
    m = parameters[parIndex[1]]
    v = hyperParameters[1]^2
    if isequal(distFunc,:LogNormal)
        parDistribution = LogNormal(log(m/sqrt(1+v/m^2)),sqrt(log(1+v/m^2)))

    elseif isequal(distFunc,:Gamma)
        θ = v/m
        k = m^2/v
        parDistribution = Gamma(k,θ)

    elseif isequal(distFunc,:Normal)
        lossFunc = prms->trunc_norm_loss(prms,m,v)
        res = optimize(lossFunc, [m,sqrt(v)]).minimizer
        parDistribution = TruncatedNormal(res[1],res[2], 0.0,Inf)

    else
        error("Mixing distribution not recognised.
               Current options are LogNormal, Gamma and (Truncated) Normal")
    end

    # Find the value of theta at which the CDF reaches cdfMax
    thetMax = invlogcdf(parDistribution, log(cdfMax))
    thetVec = collect(range(0.0,stop=thetMax,length=lTheta))
    dThet = thetVec[2]-thetVec[1]
    thetVec = thetVec[2:end]
    GVec = zeros(BigFloat,M)
    parMod = deepcopy(parameters)

    # Evaluate generating function for each theta and add contribution
    for (ii,thet) in enumerate(thetVec)
    	parMod[parIndex] = thet
    	GVec .+= genfunc(parMod,M) .* Distributions.pdf(parDistribution,thet)
    end
    GVec .*= dThet

    return GVec

end
