module ModelInference

using Distributions, CSV, DataFrames, Plots
import LinearAlgebra, GSL, Printf, Base.Threads, Random, Future, SparseArrays
import SignalUtils

include("PlotUtils.jl")

export solvemaster_matrix, solvemaster, solvecompound, mcmc_metropolis, log_likelihood


"""
Function to load in the experimental data, filtering out missings/NaNs and applying upper cut-off
"""
function load_data(File::String, Folder::String, cutOff::Number=5.0)

	rnaData = CSV.read(Folder*File*".csv", datarow=1)[1]
	rnaData = collect(Missings.replace(rnaData, NaN))
	filter!(x -> !isnan(x), rnaData)

	nMax = maximum(round.(rnaData))
	for ii=1:nMax
		global fltData = filter(x -> x<nMax-ii, rnaData)
		# if maximum(fltData)<cutOff*mean(fltData)
		if maximum(fltData)<mean(fltData)+cutOff*std(fltData)
			break
		end
	end

	return fltData

end


"""
Function to evaluate the log-likelihood of the data, given a particular set of parameters.
"""
function log_likelihood(parameters, data)
    
    Nmax = Integer(round(maximum(data)))    # Maximum value in the data
    # P = solvemaster(parameters,Nmax+1)
    P = solvemaster(parameters)
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
Function to evaluate the log-likelihood of the data, given a particular set of parameters.
"""
function log_likelihood_compound(baseParams, distParams, distFunc, idx, data; lTheta::Integer=100, cdfMax::AbstractFloat=0.98)
    
    Nmax = Integer(round(maximum(data)))    # Maximum value in the data
    P = solvecompound(baseParams, distParams, distFunc, idx; N=Nmax)
    L = length(P)
    N = max(L,Nmax)    # P runs from zero to N-1
    countVec = collect(0:N)
    Pfull = [P...;eps(Float64)*ones(Float64,N-L+1)]
    
    indcs = Integer.(round.(data)) .+ 1
    filter!(x -> x>0, indcs)
    lVec = Pfull[indcs]
    
    return sum(log.(lVec))
    
end


"""
Function to perform the MCMC metropolis algorithm.
"""
function mcmc_metropolis(x0::AbstractArray, logPfunc::Function, Lchain::Integer; propVar::AbstractFloat=0.1,
                         burn::Integer=500, step::Integer=500, printFreq::Integer=10000, prior=:none)
    
	if length(size(x0)) > 1 # restart from old chain
        println("Restarting from old chain")
		xOld = x0[end,:]
		n = length(xOld)
	else
	    xOld = x0
	    n = length(x0)
	end
    chain = zeros(Float64, Lchain,n)
    chain[1,:] = xOld
    acc = 0
    
    if prior == :none
    	logpOld = logPfunc(xOld)
    else
        logpOld = logPfunc(xOld)
        for  (ip,prr) in enumerate(prior)
        	logpOld += log(pdf(prr,xOld[ip]))
        end
    end
    
    for ii=2:Lchain
        proposal = MvNormal(propVar.*sqrt.(xOld))
        xNew = xOld + rand(proposal)
        if prior == :none
	    	logpNew = logPfunc(xNew)
	    else
	        logpNew = 0.0
	        for  (ip,prr) in enumerate(prior)
	        	logpNew += log(pdf(prr,xNew[ip]))
	        end
	        if !isinf(logpNew); logpNew += logPfunc(xNew); end
	    end
        a = exp(logpNew - logpOld)

        if rand(1)[1] < a
            xOld = xNew
            logpOld = logpNew
            acc += 1
        end
        chain[ii,:] = xOld
        
        if ii % printFreq == 0
            Printf.@printf("Completed iteration %i out of %i. \n", ii,Lchain)
        end
    end
    Printf.@printf("Acceptance ratio of %.2f (%i out of %i).\n", acc/Lchain,acc,Lchain)

	if length(size(x0)) > 1
		chainRed = [x0; chain[step:step:end,:]] # Append new chain to old
	else
		chainRed = chain[burn:step:end,:]
	end
	return chainRed
    
end

"""
Function to perform the MCMC metropolis algorithm in parallel using multithreading.
"""
function mcmc_metropolis_par(x0::AbstractArray, logPfunc::Function, Lchain::Integer;
                         prior=:none, propVar::AbstractFloat=0.1)
    
    nChains = Threads.nthreads()
    
    r = let m = Random.MersenneTwister(1)
        [m; accumulate(Future.randjump, fill(big(10)^20, nChains-1), init=m)]
    end

	chains = Array{Array{Float64,2},1}(undef,nChains)
	xstarts = Array{Array{Float64,1},1}(undef,nChains)
    if length(size(x0[1])) > 1   # Restart from old chain
        println("Restarting from old chain")
    	for ii=1:nChains
    		xstarts[ii] = x0[ii][end,:]
    	end
    	n = length(x0[1][end,:])
    else
    	for ii=1:nChains
    		xstarts[ii] = x0
    	end
    	n = length(x0)
    end
    println(size(xstarts[1]))

    acc = Threads.Atomic{Int64}(0)

    Threads.@threads for ii=1:nChains
	    chains[ii] = zeros(Float64, Lchain,n)
	    chains[ii][1,:] = xstarts[ii]
	    xOld = xstarts[ii]
	    
	    if prior == :none
			logpOld = logPfunc(xOld)
		else
		    logpOld = logPfunc(xOld)
		    for  (ip,prr) in enumerate(prior)
		    	logpOld += log(pdf(prr,xOld[ip]))
		    end
		end
	    
	    for jj=2:Lchain
            proposal = MvNormal(propVar.*sqrt.(xOld))
	        xNew = xOld + rand(r[Threads.threadid()], proposal)
	        if prior == :none
		    	logpNew = logPfunc(xNew)
		    else
		        logpNew = 0.0
		        for  (ip,prr) in enumerate(prior)
		        	logpNew += log(pdf(prr,xNew[ip]))
		        end
                # Don't evaluate if prior is zero
		        if !isinf(logpNew); logpNew += logPfunc(xNew); end
		    end
	        a = exp(logpNew - logpOld)

	        if rand(r[Threads.threadid()],1)[1] < a
	            xOld = xNew
	            logpOld = logpNew
	            Threads.atomic_add!(acc, 1)
	        end
	        chains[ii][jj,:] = xOld
	    end
	end

    Printf.@printf("Acceptance ratio of %.2f (%i out of %i).\n", acc[]/(Lchain*nChains),acc[],Lchain*nChains)
    
    if length(size(x0[1])) > 1
    	for ii=1:nChains
    		chains[ii] = [x0[ii];chains[ii]] # Append new chain to old
    	end
    end
    return chains
    
end

function chain_reduce(chains; burn::Integer=500, step::Integer=500)

    tmp = Array{Any,1}(undef,4)
	for ii=1:length(chains)
    	tmp[ii] = chains[ii][burn:step:end,:]
    end

    vcat(tmp...)

end

"""
Function to compute the hypergeometric function using a Maclaurin series. Adapted from 
https://github.com/JuliaApproximation/SingularIntegralEquations.jl/src/HypergeometricFunctions/specialfunctions.jl
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


"""
Function to obtain the steady state distribution when one or more parameters are drawn from a distribution.
"""
function solvecompound(parameters::AbstractArray, hyperParameters::AbstractArray, distFunc::Function,
		 parIndex=[1]; lTheta::Integer=100, cdfMax::AbstractFloat=0.98, N::Union{Symbol,Integer}=:auto)
    

    parDistribution = distFunc(log(parameters[parIndex[1]])-hyperParameters[1]^2/2,hyperParameters[1])

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
        	PVec[ii] = solvemaster2(parMod, N)
            # PVec[ii] = solvemaster(parMod, N)
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
Function to solve the master equation analytically
"""
function solvemaster(parameters, N=:auto::Union{Symbol,Int64})

    λ = parameters[2]
    ν = parameters[3]
    K = parameters[1]
    δ = 1.0
    E = λ*K/(λ+ν)/δ    # Mean
    V = λ*K/(λ+ν)/δ + λ*ν*K^2/((λ+ν)^2)/(λ+ν+δ)/δ    # Variance
    
    if N==:auto
        M = Int(ceil(E+5*sqrt(V)))    # Rule of thumb for maximum number of mRNA with non-negligible probability
    else
    	M = min(N,Int(ceil(E+5*sqrt(V))))
    end
    P = zeros(M)
    
    a = λ/δ
    b = (λ+ν)/δ
    c = K/δ
    
    try # Hypergeom1F1 fails for some argument values
    	for n=0:M-1
    		P[n+1] = c^n * GSL.hypergeom(a+n,b+n,-c)/factorial(big(n))
	    	if Base.isinf(P[n+1])    # Use Stirling's approximation for n!
	    		P[n+1] = GSL.hypergeom(a+n,b+n,-c) * (c*ℯ/n)^n / sqrt(2*n*pi)
	    	end
	        for m=0:n-1
	            P[n+1] *= (a+m)/(b+m)
	        end
    	end
    catch
    	P = solvemaster_matrix(parameters,N,true)
    end

    # P = P./sum(P)
    return P
    
end


# import Optim, JuMP, Clp
"""
Function to solve the master equation at steady state, returning the probability mass function.
"""
function solvemaster2(parameters, N=:auto::Union{Symbol,Int64}, verbose=false)

	if verbose
    	Printf.@printf("Solving master equation via the combined method.\n")
    end
    
    λ = parameters[2]
    ν = parameters[3]
    K = parameters[1]
    δ = 1.0
    E = λ*K/(λ+ν)/δ    # Mean
    V = λ*K/(λ+ν)/δ + λ*ν*K^2/((λ+ν)^2)/(λ+ν+δ)/δ    # Variance
    
    Nguess = Int(round(E+5*sqrt(V)))    # Rule of thumb for maximum number of mRNA with non-negligible probability
    if N==:auto
        N = Nguess
    else
    	N = min(N,Nguess)
    end

    P = zeros(N)
    
    a = λ/δ
    b = (λ+ν)/δ
    c = K/δ

	global T = N
    global failed = false
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

    if false
        if verbose
    	   Printf.@printf("Hypergeom1f1 failed, solving matrix equation for N > %i.\n",T)
        end
    	# T -= 1
        T = 10
	    A = A_matrix(λ,ν,K,δ, N, true)

	    # Solve AX=b problem for remaining values in P
	    Ar = [A[T:N,T:N] SparseArrays.spzeros(N-T+1,N-T+1);
	    	  SparseArrays.spzeros(N-T+1,N-T+1) A[N+T:end,N+T:end]]
	    b = zeros(2*(N-T+1),1)
	    b[N-T+2] = 1.0
	    Pr = Ar\b

	    # Reassemble reduced form of P
	    Pr = abs.(Pr[1:N-T+1] + Pr[N-T+2:end])
	    Pr = Pr .* P[T]/Pr[1]    # Rescale Pr
	    P[T:end] = Pr
	end

    if failed
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

    # Use optim to converge to the nullspace
    # f(x) = sum(abs.(A*x))
	# res = Optim.optimize(f,P)
	# P = res.minimizer

	# m = JuMP.Model(solver=Clp.ClpSolver())
	# JuMP.@variable(m, x[1:2*N])
	# JuMP.@variable(m, y[1:2*N])
	# for ii=1:2N
	# 	JuMP.@constraint(m, 0 <= x[ii])
	# 	JuMP.@constraint(m, y[ii] >= A[ii,:]'*x)
	#	JuMP.@constraint(m, y[ii] >= -A[ii,:]'*x)
	# end
	# JuMP.@objective(m, Min, sum(y))
	# JuMP.solve(m)
	# P = JuMP.getvalue(x)

    P = P./sum(P)
    
end


"""
Function to solve the master equation at steady state, returning the probability mass function.
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
        N = Int(round(E+5*sqrt(V)))    # Rule of thumb for maximum number of mRNA with non-negligible probability
    end
    
    A = A_matrix(λ,ν,K,δ, N)

    P = LinearAlgebra.nullspace(A)
    P = abs.(P[1:N] + P[N+1:end])
    P = P./sum(P)
    
end

"""
Function to assemble the A matrix of the master equation.
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