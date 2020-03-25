# Main module file for work on modelling bursty transcription and extrinsic noise

module TxModels

using Distributions, CSV, DataFrames, Plots, Optim, DifferentialEquations
import LinearAlgebra, GSL, Base.Threads, Random, Future, SparseArrays, SpecialFunctions, DelimitedFiles
import KernelDensity; const KDE = KernelDensity
import LinearAlgebra; const LinAlg = LinearAlgebra
import Combinatorics: stirlings1
import Printf: @sprintf, @printf
using .MathConstants: γ
import Base: rand
import Random: AbstractRNG

export TelegraphDist,
    solvemaster,
    maxentropyestimation

include("ModelInference.jl")
include("PlotUtils.jl")
include("Distribution.jl")
include("Utilities.jl")
include("SpecialCases.jl")
include("Recurrence.jl")
include("extrinsicinference.jl")


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


end
