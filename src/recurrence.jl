# Functions for evaluating the leaky gene distribution using Lucy's recurrence relations

"""
Compute leaky gene distribution via the recurrence method.
Compute up to copy number N with M terms in the recursion
"""
function rec_twostate(prms, N, M)

	G = genfunc_twostate(prms, M)
	return [invgenfunc(G, n) for n=0:(N-1)]

end

"""
Compute leaky gene distribution via the recurrence method.
Compute up to copy number N with M terms in the recursion
"""
function rec_threestate(prms, N, M)

    G = genfunc_threestate(prms, M)
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
Evaluate the generating function G(z)
"""
function genfunc_threestate(prms, M)

    # Set up parameters
    v12 = BigFloat(prms[1])
    v21 = BigFloat(prms[2])
    v13 = BigFloat(prms[3])
    v31 = BigFloat(prms[4])
    v23 = BigFloat(prms[5])
    v32 = BigFloat(prms[6])
    k1 = BigFloat(prms[7])
    k2 = BigFloat(prms[8])
    k3 = BigFloat(prms[9])

    # Determinent like function
    function d(n)
        return (n^BigFloat(2) + n*(v12 + v13 + v21 + v23 + v31 + v32) + 
        v12*(v23 + v31 + v32) + v13*(v21 + v23 + v32) + v21*(v31 + v32) + v23*v31)
    end

    # Recursive updater
    function update(nn, ss)
        n = BigFloat(nn)
        up = Array{BigFloat}(undef, 3)
        for i in 1:3
            aa = [1,2,3]
            filter!(x->!isequal(x,i),aa)
            j = aa[1]
            k = aa[2]

            up[i] = kvec[j]*ss[j]*(BigFloat(n)*vvec[j, i] + vvec[j, k]*vvec[k, i] + vvec[j, i]*(vvec[k, i] + vvec[k, j]))
                + kvec[k]*ss[k]*(BigFloat(n)*vvec[k, i] + vvec[j, k]*vvec[k, i] + vvec[j, i]*(vvec[k, i] + vvec[k, j]))
                + kvec[i]*ss[i]*(BigFloat(n^2) + vvec[j, i]*vvec[k, i] + vvec[j, k]*vvec[k, i] + vvec[j, i]*vvec[k, j]
                + BigFloat(n)*(vvec[j, i] + vvec[j, k] + vvec[k, i] + vvec[k, j]))

            up[i] *= BigFloat(1)/(n*d(n))
        end
        return up
    end

    # Useful arrays
    vvec = [0 v12 v13; v21 0 v23; v31 v32 0]'
    kvec = [k1, k2, k3]

    # Initial condition
    ini = [(vvec[2, 1]*vvec[3, 1] + vvec[2, 3]*vvec[3, 1] + vvec[2, 1]*vvec[3, 2])/d(0),
            (vvec[1, 2]*vvec[3, 1] + vvec[1, 2]*vvec[3, 2] + vvec[1, 3]*vvec[3, 2])/d(0),
            (vvec[1, 3]*vvec[2, 1] + vvec[1, 2]*vvec[2, 3] + vvec[1, 3]*vvec[2, 3])/d(0)]

    # Evaluate the recursion
    SS = Array{BigFloat,2}(undef,3,M)
    SS[:,1] = ini
    for n in 2:M
        SS[:,n] = update(n-1, SS[:,n-1])
    end
    G = sum(SS,dims=1)

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
