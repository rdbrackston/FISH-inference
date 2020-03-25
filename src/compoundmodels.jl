"""
Function to draw many samples from one of the compound distributions.
Valid if the underlying gene expression is either constitutive (poisson),
extremely bursty (NegativeBinomial) or Telegraph.
"""
function samplecompound(parameters::AbstractArray, hyperParameters::AbstractArray,
						distFunc::Symbol, mixDist::Symbol, Nsamp::Int=1000; parIdx::Int=1)

	smpls = zeros(Nsamp)

	K = parameters[1]
    λ = parameters[2]
    ν = parameters[3]
    m = parameters[parIdx]
    v = hyperParameters[1]^2
    if isequal(mixDist,:LogNormal)
        parDistribution = LogNormal(log(m/sqrt(1+v/m^2)),sqrt(log(1+v/m^2)))

    elseif isequal(mixDist,:Gamma)
        θ = v/m
        k = m^2/v
        parDistribution = Gamma(k,θ)

    elseif isequal(mixDist,:Normal)
        lossFunc = prms->trunc_norm_loss(prms,m,v)
        res = optimize(lossFunc, [m,sqrt(v)]).minimizer
        parDistribution = TruncatedNormal(res[1],res[2], 0.0,Inf)

    elseif isequal(mixDist,:Gumbel)
        β = sqrt(6*v)/π
        μ = m - β*γ
        parDistribution = Truncated(Gumbel(μ,β),0.0,Inf)

    else
        error("Mixing distribution not recognised.
               Current options are LogNormal, Gamma and (Truncated) Normal")
    end

    if isequal(distFunc,:NegativeBinomial)
		for ii=1:Nsamp
			# Draw parameter from mixDist
			K = Distributions.rand(parDistribution)
			p = ν/(K+ν)

			# Sample from main distribution, with unique parametrization
			d = NegativeBinomial(λ,p)
			smpls[ii] = Distributions.rand(d)
		end

	elseif isequal(distFunc,:Poisson)
		for ii=1:Nsamp
			# Draw parameter from mixDist
			K = Distributions.rand(parDistribution)

			# Sample from main distribution, with unique parametrization
			d = Poisson(K)
			smpls[ii] = Distributions.rand(d)
		end

	elseif isequal(distFunc,:Telegraph)
		for ii=1:Nsamp
			# Draw parameter from mixDist
			θ = Distributions.rand(parDistribution)
            prms = deepcopy(parameters)
            prms[parIdx] = θ

			# Sample from main distribution, with unique parametrization
			d = TelegraphDist(prms...)
			smpls[ii] = rand(d)
		end

	else
		error("Main distribution not recognised.
               Current options are NegativeBinomial, Poisson or Telegraph")
	end

	return Integer.(smpls)

end


"""
Function to obtain the steady state distribution when one or more parameters are
drawn from a distribution. Recursively marginalises over each parameter in turn.
"""
function solvecompound(parameters::AbstractArray, hyperParameters::AbstractArray,
                       distFunc::Symbol, parIndex=[1]; lTheta::Integer=200,
                       cdfMax::AbstractFloat=0.999, N::Union{Symbol,Integer}=:auto,
                       verbose=false, method=:fast)

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
    thetVec = thetVec[2:end]
    PVec = Array{Array{Float64},1}(undef,length(thetVec))
    parMod = deepcopy(parameters)

    # Recursively apply solvecompound to marginalise over each parameter in turn
    L = 0
    for (ii,thet) in enumerate(thetVec)
    	parMod[parIndex[1]] = thet
    	if length(parIndex) > 1
    		PVec[ii] = solvecompound(parMod, hyperParameters[2:end], distFunc,
    			parIndex[2:end], lTheta=lTheta, cdfMax=cdfMax, N=N, method=method)
    	else
    		if isequal(method,:fast)
        		PVec[ii] = solvemaster(parMod, N, verbose)
            elseif isequal(method,:FSP)
                PVec[ii] = solvemaster_fsp(parMod, N, verbose)
        	else
        		PVec[ii] = solvemaster2(parMod, N, verbose)
        	end
        end
        L = max(L,length(PVec[ii])-1)
    end

    # Pad all Ps with zeros to make of equal length (N+1)
    for ii=1:length(PVec)
        l = length(PVec[ii])
        if l <= L
            PVec[ii] = [PVec[ii];eps(Float64)*ones(Float64,L+1-l)]
        end
    end

    Q = zeros(Float64, L+1)
    # Loop over all the values of n
    for ii=1:length(Q)

        # Perform the integration using the trapezium rule
        Q[ii] += PVec[1][ii]*Distributions.pdf(parDistribution,thetVec[1])
        Q[ii] += PVec[end][ii]*Distributions.pdf(parDistribution,thetVec[end])
        for jj = 2:length(thetVec)-1
            Q[ii] += 2*PVec[jj][ii]*Distributions.pdf(parDistribution,thetVec[jj])
        end

    end
    Q = Q./sum(Q)    # Normalize

end


"""
Loss function for the parameters of the truncated normal distribution.
Distribution is truncated at zero, parameterized by μ,σ and has moments m,v.
"""
function trunc_norm_loss(prms, m, v)

    μ = prms[1]
    σ = prms[2]
    mAct = μ + 2*exp(-μ^2/(2*σ^2))*σ/(sqrt(2*pi)*(1-SpecialFunctions.erf(-μ/sqrt(2)*σ)))
    vAct = σ^2*(1 - 2*μ*exp(-μ^2/(2*σ^2))/(σ*sqrt(2*pi)*(1-SpecialFunctions.erf(-μ/sqrt(2)*σ)))
           - (2*exp(-μ^2/(2*σ^2))/(sqrt(2*pi)*(1-SpecialFunctions.erf(-μ/sqrt(2)*σ))))^2)
    return (m-mAct)^2 + (v-vAct)^2

end
