# Main module file for work on modelling bursty transcription and extrinsic noise

module TxModels

using Distributions, CSV, DataFrames, Plots, Optim
import LinearAlgebra, GSL, Printf, Base.Threads, Random, Future, SparseArrays, SpecialFunctions

include("PlotUtils.jl")
include("ModelInference.jl")
include("Utilities.jl")

"""
Function to obtain the steady state distribution when one or more parameters are
drawn from a distribution.
"""
function solvecompound(parameters::AbstractArray, hyperParameters::AbstractArray,
                       distFunc::Symbol, parIndex=[1]; lTheta::Integer=100,
                       cdfMax::AbstractFloat=0.98, N::Union{Symbol,Integer}=:auto,
                       verbose=false)
    
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
    			parIndex[2:end], lTheta=lTheta, cdfMax=cdfMax, N=N)
    	else
        	PVec[ii] = solvemaster(parMod, N, verbose)
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
        Q[ii] += PVec[1][ii]*pdf(parDistribution,thetVec[1])
        Q[ii] += PVec[end][ii]*pdf(parDistribution,thetVec[end])
        for jj = 2:length(thetVec)-1
            Q[ii] += 2*PVec[jj][ii]*pdf(parDistribution,thetVec[jj])
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


"""
Function to solve the master equation at steady state, returning the probability
mass function. Evaluation uses the analytical expression for p(n) where possible
before solving the matrix equation for the remaining higher values of n.
"""
function solvemaster(parameters, N=:auto::Union{Symbol,Int64}, verbose=false)

	if verbose
    	Printf.@printf("Solving master equation via the combined method.\n")
    end
    
    λ = parameters[2]
    ν = parameters[3]
    K = parameters[1]
    δ = 1.0
    E = λ*K/(λ+ν)/δ    # Mean
    V = λ*K/(λ+ν)/δ + λ*ν*K^2/((λ+ν)^2)/(λ+ν+δ)/δ    # Variance
    
    # Rule of thumb for maximum number of mRNA with non-negligible probability
    Nguess = Int(round(E+5*sqrt(V)))
    if N==:auto
        N = Nguess
    else
    	N = min(N,Nguess)
    end
    
    a = λ/δ
    b = (λ+ν)/δ
    c = K/δ

    # Evaluate using anaytical expression as far as possible
	global T = N
    global failed = false
    P = zeros(N)
    for n=0:N-1
    	try # Hypergeom1F1 fails for some argument values
    		P[n+1] = c^n * GSL.hypergeom(a+n,b+n,-c)/factorial(big(n))
	    	if Base.isinf(P[n+1])    # Use Stirling's approximation for n!
	    		P[n+1] = GSL.hypergeom(a+n,b+n,-c) * (c*ℯ/n)^n / sqrt(2*n*pi)
	    	end
	        for m=0:n-1
	            P[n+1] *= (a+m)/(b+m)
	        end
	    catch
	    	P[n+1] = 0.0
	    	T = min(T,n+1)
            failed = true
    	end
    end

    # If required, evaluate remainder of P using the matrix expression
    if failed
        if verbose
    	   Printf.@printf("Hypergeom1f1 failed, solving matrix equation for N > %i.\n",T)
        end
        T-=1
        A = A_matrix(λ,ν,K,δ, N, false)
        A = [A[T:N,:]; A[N+T:end,:]]
        Pfull = [P*ν/(λ+ν); P*λ/(λ+ν)]

        idx1 = [collect(1:T-1);collect(N+1:N+T-1)]
        idx2 = [collect(T:N);collect(N+T:2N)]
        Pr = -A[:,idx2]\A[:,idx1]*Pfull[idx1]

        # Reassemble reduced form of P
        l = Integer(length(Pr)/2)
        Pr = abs.(Pr[1:l] + Pr[l+1:end])
        Pr = Pr .* P[T]/Pr[1]    # Rescale Pr
        P[T:end] = Pr
    end

    P = P./sum(P)
    
end


"""
Function to solve the master equation at steady state, returning the probability mass function.
Uses exclusively the nullspace method.
"""
function solvemaster_matrix(parameters, N=:auto::Union{Symbol,Int64}, verbose=false)
    
    if verbose
    	Printf.@printf("Solving master equation via the matrix method.\n")
    end

    λ = parameters[2]
    ν = parameters[3]
    K = parameters[1]
    δ = 1.0
    E = λ*K/(λ+ν)/δ    # Mean
    V = λ*K/(λ+ν)/δ + λ*ν*K^2/((λ+ν)^2)/(λ+ν+δ)/δ    # Variance
    
    if N==:auto
        # Rule of thumb for maximum number of mRNA with non-negligible probability
        N = Int(round(E+5*sqrt(V)))
    end
    
    A = A_matrix(λ,ν,K,δ, N)

    P = LinearAlgebra.nullspace(A)
    P = abs.(P[1:N] + P[N+1:end])
    P = P./sum(P)
    
end

"""
Function to assemble the A matrix of the master equation.
This matrix is valid for the bursty gene expresssion model with no leaky transcription.
"""
function A_matrix(λ,ν,K,δ, N, sparse=false)

	if sparse
		A = SparseArrays.spzeros(Float64, 2*N, 2*N)
	else
		A = zeros(Float64, 2*N, 2*N)
	end

    # Assemble A row by row
    A[1,1] = -λ
    A[1,2] = δ
    A[1,1+N] = ν

    A[1+N,1] = λ
    A[1+N,1+N] = -(ν+K)
    A[1+N,2+N] = δ

    for ii = 2:N-1
        A[ii,ii] = -(λ+δ*(ii-1))
        A[ii,ii+1] = δ*ii
        A[ii,ii+N] = ν

        A[ii+N,ii] = λ
        A[ii+N,ii+N] = -(ν+K+δ*(ii-1))
        A[ii+N,ii+N+1] = δ*ii
        A[ii+N,ii+N-1] = K
    end

    A[N,N] = -(λ+δ*(N-1))
    A[N,2*N] = ν

    A[2*N,N] = λ
    A[2*N,2*N] = -(ν+δ*(N-1))
    A[2*N,2*N-1] = K

    return A

end


end
