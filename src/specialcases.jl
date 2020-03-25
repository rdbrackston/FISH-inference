# Functions related to fitting special cases of the model

"""
Function to evaluate the log-likelihood of the data, given the standard model
with a particular set of parameters.
"""
function log_likelihood_ZILP(parameters, data)

    w = parameters[3]
    Nmax = Integer(round(Base.maximum(data)))    # Maximum value in the data
    P = log_poisson(parameters[1:2],Nmax+1)
    P = (1-w).*P
    P[1] += w # Zero-inflation

    N = length(P)    # P runs from zero to N-1
    countVec = collect(0:max(N,Nmax))
    Pfull = zeros(Float64, size(countVec))
    Pfull[1:N] = P

    idx = Integer.(round.(data)) .+ 1
    filter!(x -> x>0, idx)
    lVec = Pfull[idx]

    return sum(log.(lVec))

end


"""
Function to evaluate the log-likelihood of the data, given the standard model
with a particular set of parameters.
"""
function log_likelihood_compNB(parameters, data)

    Nmax = Integer(round(Base.maximum(data)))    # Maximum value in the data
    P = comp_negbinom(parameters,Nmax+1)

    N = length(P)    # P runs from zero to N-1
    countVec = collect(0:max(N,Nmax))
    Pfull = zeros(Float64, size(countVec))
    Pfull[1:N] = P

    idx = Integer.(round.(data)) .+ 1
    filter!(x -> x>0, idx)
    lVec = Pfull[idx]

    return sum(log.(lVec))

end



"""
Functionto the return a LogNormal-Poisson mixture distribution
"""
function log_poisson(parameters::AbstractArray, N::Integer; lTheta::Integer=200,
                       cdfMax::AbstractFloat=0.999, verbose=false)

    # Parameters and distribution
    m = parameters[1]
    v = parameters[2]^2
    # w = parameters[3]
    parDist = LogNormal(log(m/sqrt(1+v/m^2)),sqrt(log(1+v/m^2)))

    # Initialize arrays
    thetMax = invlogcdf(parDist, log(cdfMax))
    thetMax = max(thetMax,Float64(N))
    thetVec = collect(range(0.0,stop=thetMax,length=lTheta))
    thetVec = thetVec[2:end]
    PVec = Array{Array{Float64},1}(undef,length(thetVec))

    # Loop over the hyperparameter vector
    for (ii,thet) in enumerate(thetVec)

        d = Poisson(thet)
        PVec[ii] = [Distributions.pdf(d,x) for x=0:N-1]

    end

    Q = zeros(Float64, N)
    # Loop over all the values of n
    for ii=1:N

        # Perform the integration using the trapezium rule
        Q[ii] += PVec[1][ii]*Distributions.pdf(parDist,thetVec[1])
        Q[ii] += PVec[end][ii]*Distributions.pdf(parDist,thetVec[end])
        for jj = 2:length(thetVec)-1
            Q[ii] += 2*PVec[jj][ii]*Distributions.pdf(parDist,thetVec[jj])
        end

    end
    Q = Q./sum(Q)    # Normalize

end


"""
Function to the return a Compound negative bimomial distribution
"""
function comp_negbinom(parameters::AbstractArray, N::Integer; lTheta::Integer=200,
                       distFunc::Symbol=:LogNormal, cdfMax::AbstractFloat=0.999)

    # Parameters and distribution
    m = parameters[1]
    p = parameters[2]
    v = parameters[3]^2

    if isequal(distFunc,:LogNormal)
        parDist = LogNormal(log(m/sqrt(1+v/m^2)),sqrt(log(1+v/m^2)))

    elseif isequal(distFunc,:Gamma)
        θ = v/m
        k = m^2/v
        if k<1 # Parameters entering the negative binomial must be positive
            println("Warning, Gamma distribution has probability at zero. Aborting.")
            return zeros(N)
        end
        parDist = Gamma(k,θ)

    elseif isequal(distFunc,:Normal)
        lossFunc = prms->trunc_norm_loss(prms,m,v)
        res = optimize(lossFunc, [m,sqrt(v)]).minimizer
        parDist = TruncatedNormal(res[1],res[2], 0.0,Inf)

    else
        error("Mixing distribution not recognised.
               Current options are LogNormal, Gamma and (Truncated) Normal")
    end

    # Initialize arrays
    thetMax = invlogcdf(parDist, log(cdfMax))
    thetVec = collect(range(0.0,stop=thetMax,length=lTheta))
    thetVec = thetVec[2:end] # Remove first element which is zero
    PVec = Array{Array{Float64},1}(undef,length(thetVec))

    # Loop over the hyperparameter vector
    for (ii,thet) in enumerate(thetVec)

        if !(thet>0)
            println(thet)
            println(m)
            println(v)
        end

        d = NegativeBinomial(thet,p)
        PVec[ii] = [Distributions.pdf(d,x) for x=0:N-1]

    end

    Q = zeros(Float64, N)
    # Loop over all the values of n
    for ii=1:N

        # Perform the integration using the trapezium rule
        Q[ii] += PVec[1][ii]*Distributions.pdf(parDist,thetVec[1])
        Q[ii] += PVec[end][ii]*Distributions.pdf(parDist,thetVec[end])
        for jj = 2:length(thetVec)-1
            Q[ii] += 2*PVec[jj][ii]*Distributions.pdf(parDist,thetVec[jj])
        end

    end
    Q = Q./sum(Q)    # Normalize

end


"""
Calculate the contributions to the variance of the intrinsic and extrinsic noise.
Currently implemented only for the compound negative binomial model.
"""
function contributions(parameters::AbstractArray, N::Integer;
                        lTheta::Integer=200, cdfMax::AbstractFloat=0.999)

    # Parameters and distribution
    m = parameters[1]
    p = parameters[2]
    v = parameters[3]^2

    # Evaluate mean and variance for compound distribution
    d = TxModels.comp_negbinom(parameters,N, distFunc=:LogNormal,lTheta=lTheta,cdfMax=cdfMax)
    E = LinAlg.dot(collect(0:length(d)-1),d)
    V = LinAlg.dot((collect(0:length(d)-1).-E).^2,d)

    # Set up extrinsic distribution
    dE = LogNormal(log(m/sqrt(1+v/m^2)),sqrt(log(1+v/m^2)))
    thetMax = invlogcdf(dE, log(cdfMax))
    thetVec = collect(range(0.0,stop=thetMax,length=lTheta))

    # Evaluate intrinsic contribution: E[Var[x|e]]
    Vi = 0.0
    for thet in thetVec[2:end]
        tmp = NegativeBinomial(thet,p)
        Vi += var(tmp)*Distributions.pdf(dE,thet)
    end
    Vi *= diff(thetVec)[1]

    # Evaluate extrinsic contribution: Var[E[x|e]]
    Ve = 0.0
    for thet in thetVec[2:end]
        tmp = NegativeBinomial(thet,p)
        Ve += (mean(tmp)-E)^2*Distributions.pdf(dE,thet)
    end
    Ve *= diff(thetVec)[1]
    
    return Vi,Ve

end
