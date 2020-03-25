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
function samplesim(parameters, N::Integer=1000)

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
