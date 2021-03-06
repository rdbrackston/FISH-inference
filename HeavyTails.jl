module HeavyTails

using Plots, Statistics, Random, Distributions
gr()


"""
Mean excess plot generated from sampled data
"""
function meanexcess(X; L=1000)
	
	samps = deepcopy(X)
	sort!(samps)
	# strt = min(0.0, minimum(X)-eps())
	# uVec = collect(range(strt, samps[end-1], length=L))
	uVec = samps[1:end-1]
	eVec = Array{Float64,1}(undef,length(uVec))

	for (ii,u) in enumerate(uVec)
		filter!(x->x>u, samps)
		eVec[ii] = mean(samps.-u)
	end

	plt = plot(uVec,eVec, xlabel="Threshold, u",ylabel="Mean excess, e(u)", label="", line=0,marker=(2,))
	# plot!([mean(X),mean(X)], [min(eVec...),max(eVec...)], line=:black)

	m,b = ls_fit(uVec[uVec.>mean(X)],eVec[uVec.>mean(X)])
	plot!(uVec[uVec.>mean(X)],z->m*z+b, line=:black, label="Least squares fit")

	return uVec,eVec,m,plt

end
function ls_fit(x,y)
	N = length(x)
	sx = sum(x)
	sy = sum(y)
	m = (N*sum(x.*y)-sx*sy)/(N*sum(x.^2)-sx^2)
	b = (sy-m*sx)/N
	return m,b
end

"""
Mean excess plot generated by numerically integrating a distribution
"""
function meanexcess(d::ContinuousDistribution; L=1000)
	
	quantl = 0.99999

	strt = 0.0
	stop = invlogcdf(d, log(quantl))
	uVec = collect(range(strt, stop, length=L))
	eVec = Array{Float64,1}(undef,L)

	for (ii,u) in enumerate(uVec)
		tmp = Truncated(d,u,Inf)
		int = 0.0
		xgrid = range(u, invlogcdf(tmp,log(quantl)), length=1000)
		dx = diff(xgrid)[1]
		for x in xgrid[2:end]
			int += dx*(x-u)*pdf(tmp,x)
		end
		eVec[ii] = int
	end

	# plot(uVec,eVec, xlabel="Threshold, u",ylabel="Mean excess, e(u)")
	return uVec,eVec

end


"""
Mean excess plot generated by numerically integrating a distribution
"""
function meanexcess(d::DiscreteDistribution; L=1000)
	
	quantl = 0.99999

	strt = 0.0
	stop = invlogcdf(d, log(quantl))
	uVec = collect(range(strt, stop, length=L))
	eVec = Array{Float64,1}(undef,L)

	for (ii,u) in enumerate(uVec)
		tmp = Truncated(d,u,Inf)
		int = 0.0
		for x in Int(ceil(u)):invlogcdf(tmp,log(quantl))
			int += (x-u)*pdf(tmp,x)
		end
		eVec[ii] = int
	end

	# plot(uVec,eVec, xlabel="Threshold, u",ylabel="Mean excess, e(u)")
	return uVec,eVec

end


"""
Moment convergence plot 
"""
function Convergence(X, func::Function)
	
end


"""
Maximum sum ratio.
Indicate if moment of order p is finite.
"""
function maxsumratio(X, p::Int)

	N = length(X)
	Xp = (abs.(X)).^p
	idx = collect(1:N)

	Rn = Array{Float64,1}(undef,N-1)
	for n=2:N
		shuffle!(idx)
		samps = Xp[idx[1:n]]
		Rn[n-1] = maximum(samps)/sum(samps)
	end

	return Rn
	
end

end # module