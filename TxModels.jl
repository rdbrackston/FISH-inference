# Main module file for work on modelling bursty transcription and extrinsic noise

module TxModels

using Distributions, CSV, DataFrames, Plots, Optim, DifferentialEquations
import LinearAlgebra, GSL, Printf, Base.Threads, Random, Future, SparseArrays, SpecialFunctions, DelimitedFiles
export TelegraphDist

include("ModelInference.jl")
include("PlotUtils.jl")
include("Distribution.jl")
include("Utilities.jl")


"""
Function to draw many samples from one of the special case compound distributions.
Valid if the underlying gene expression is either constitutive (poisson) or 
extremely bursty (NegataveBinomial).
"""
function samplecompound(parameters::AbstractArray, hyperParameters::AbstractArray,
						distFunc::Symbol, mixDist::Symbol, Nsamp::Int=1000)

	smpls = zeros(Nsamp)

	m = parameters[1]
    λ = parameters[2]
    ν = parameters[3]
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
			K = Distributions.rand(parDistribution)

			# Sample from main distribution, with unique parametrization
			d = TelegraphDist(K,λ,ν)
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
Function to set up and run a simulation of the system, returning a solution
"""
function runsim(parameters)

	# Set up full length parameter vector
	if length(parameters)==3
		prms = [0.0; parameters[:]]
	else
		prms = parameters
	end

	# Setup the problem
	x₀ = [Integer(round(prms[2])),1,0]
	tspan = (0.0, 200000.0)
	prob = DiscreteProblem(x₀,tspan,prms)

	# RNA synthesis at rate K₀
	jumprate1(u,p,t) = u[3]*p[1]
	affect1!(integrator) = integrator.u[1] += 1.
	jump1 = ConstantRateJump(jumprate1,affect1!)

	# RNA synthesis at rate K₁
	jumprate2(u,p,t) = u[2]*p[2]
	affect2!(integrator) = integrator.u[1] += 1.
	jump2 = ConstantRateJump(jumprate2,affect2!)

	# RNA degradation, rate equal to one
	jumprate3(u,p,t) = u[1]
	affect3!(integrator) = integrator.u[1] -= 1.
	jump3 = ConstantRateJump(jumprate3,affect3!)

	# Gene activation
	jumprate4(u,p,t) = u[3]*p[3]
	affect4!(integrator) = (integrator.u[2] += 1.; integrator.u[3] -= 1.)
	jump4 = ConstantRateJump(jumprate4,affect4!)

	# Gene deactivation
	jumprate5(u,p,t) = u[2]*p[4]
	affect5!(integrator) = (integrator.u[2] -= 1.; integrator.u[3] += 1.)
	jump5 = ConstantRateJump(jumprate5,affect5!)

	# Run simulation
	jump_prob = JumpProblem(prob,Direct(),jump1,jump2,jump3,jump4,jump5)
    sol = solve(jump_prob, FunctionMap(), maxiters=10^8)

end


"""
Function to return samples from a simulation
"""
function samplemaster(parameters, N::Integer=1000)

	sol = runsim(parameters)
	tSamp = 100.0 .+ (sol.t[end]-100.0)*Base.rand(N)
    simData = [sol(tSamp[ii])[1] for ii=1:length(tSamp)]

end


"""
Function to evaluate a normalised probability distribution from a simulation
"""
function simmaster(parameters)

    simData = samplemaster(parameters, 10000)
    bin,P = genpdf(Integer.(round.(simData)))

end


"""
Function to generate a distribution from simulating samples from a compound distribution.
"""
function simcompound(parameters::AbstractArray, hyperParameters::AbstractArray,
                     mixDist::Symbol, parIndex=[1], Nsamp::Integer=1000)

	simData = zeros(Nsamp)

	m = parameters[1]
    λ = parameters[2]
    ν = parameters[3]
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

    else
        error("Mixing distribution not recognised.
               Current options are LogNormal, Gamma and (Truncated) Normal")
    end

	for ii=1:Nsamp
		parameters[parIndex...] = Distributions.rand(parDistribution)
		simData[ii] = samplemaster(parameters)[1]
	end

	return Integer.(round.(simData))

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

 	# Call the full function if four parameters are included
    if length(parameters)==4

    	λ = parameters[3]
		ν = parameters[4]
		K₀ = parameters[1]
		K₁ = parameters[2]
	    δ = 1.0
	    E = λ*K₁/(λ+ν)/δ    # Mean
	    V = λ*K₁/(λ+ν)/δ + λ*ν*K₁^2/((λ+ν)^2)/(λ+ν+δ)/δ    # Variance
	    
	    # Rule of thumb for maximum number of mRNA with non-negligible probability
	    Nguess = Int(round(E+5*sqrt(V)))
	    if N==:auto
	        N = Nguess
	    else
	    	# N = min(N,Nguess)
	    end

	    return solvemaster_full(parameters, N)

	else

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
	    	# N = min(N,Nguess)
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
	        A = A_matrix(0,K,λ,ν,δ, N, false)
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
	    return P
	end
    
end


"""
Function to evaluate the analytical distribution for the leaky gene expression model.
Called by solvemaster for cases in which parameters has length four.
!! Has numerical stability issues for large n, solution is work in progress !!
"""
function fracrise(x, y, r)
	Q = 1
		for m=0:r-1
			Q *= (x + m) / (y + m)
		end
	return Q
end

function solvemaster_full(parameters, N)

	# Assign the parameters
	λ = parameters[3]
	ν = parameters[4]
	K₀ = parameters[1]
	K₁ = parameters[2]
    δ = 1.0

	P = zeros(N)
	Max = 150
    for n=0:N-1
    	for r=0:n
    		if r>Max && (n-r)>Max
    			println(n)
    			P[n+1] +=  big(ℯ*K₁/(n-r))^(n-r) * big(ℯ*(K₀-K₁)/r)^r * GSL.hypergeom(ν+r,λ+ν+r,K₁-K₀) * 
    					   fracrise(ν,ν+λ,r) * ℯ^(-K₁)  / (2π * sqrt(r*(n-r)))
    		else
    			P[n+1] += big(K₁)^(n-r) * big(K₀-K₁)^r * GSL.hypergeom(ν+r,λ+ν+r,K₁-K₀) * 
    					  fracrise(ν,ν+λ,r) * ℯ^(-K₁) / (factorial(big(n-r)) * factorial(big(r)))	
    		end
    	end
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

    if length(parameters)==3

	    λ = parameters[2]
	    ν = parameters[3]
	    K₀ = 0.0
	    K₁ = parameters[1]

	else

    	λ = parameters[3]
	    ν = parameters[4]
	    K₀ = parameters[1]
	    K₁ = parameters[2]

	end

	δ = 1.0
    E = λ*K₁/(λ+ν)/δ    # Mean
    V = λ*K₁/(λ+ν)/δ + λ*ν*K₁^2/((λ+ν)^2)/(λ+ν+δ)/δ    # Variance
    
    if N==:auto
        # Rule of thumb for maximum number of mRNA with non-negligible probability
        N = Int(round(E+5*sqrt(V)))
    end

    A = A_matrix(K₀,K₁,λ,ν,δ, N)
    
    P = LinearAlgebra.nullspace(A)
    P = abs.(P[1:N] + P[N+1:end])
    P = P./sum(P)
    
end


"""
Function to assemble the A matrix of the master equation.
This matrix is valid for the bursty gene expresssion model with no leaky transcription.
"""
function A_matrix(K₀,K₁,λ,ν,δ, N, sparse=false)

	if sparse
		A = SparseArrays.spzeros(Float64, 2*N, 2*N)
	else
		A = zeros(Float64, 2*N, 2*N)
	end

    # Assemble A row by row
    A[1,1] = -(λ+K₀)
    A[1,2] = δ
    A[1,1+N] = ν

    A[1+N,1] = λ
    A[1+N,1+N] = -(ν+K₁)
    A[1+N,2+N] = δ

    for ii = 2:N-1
        A[ii,ii] = -(λ+K₀+δ*(ii-1))
        A[ii,ii+1] = δ*ii
        A[ii,ii+N] = ν
        A[ii,ii-1] = K₀

        A[ii+N,ii] = λ
        A[ii+N,ii+N] = -(ν+K₁+δ*(ii-1))
        A[ii+N,ii+N+1] = δ*ii
        A[ii+N,ii+N-1] = K₁
    end

    A[N,N] = -(λ+δ*(N-1))
    A[N,2*N] = ν
    A[N,N-1] = K₀

    A[2*N,N] = λ
    A[2*N,2*N] = -(ν+δ*(N-1))
    A[2*N,2*N-1] = K₁

    return A

end


end
