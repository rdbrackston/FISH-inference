# Collection of functions used for parameter inference

"""
Function to load in the experimental data, filtering out missings/NaNs and
applying upper cut-off in terms of standard deviations from the mean.
"""
function load_data(File::String, Folder::String, cutOff::Number=Inf, format=:Int)

    if occursin(".csv",File)
        rnaData = CSV.read(Folder*File, datarow=1)[1]
    else
	    rnaData = CSV.read(Folder*File*".csv", datarow=1)[1]
    end

    try
	    rnaData = collect(Missings.replace(rnaData, NaN))
    catch
        rnaData = collect(Missings.replace(rnaData, 0))
    end
	filter!(x -> !isnan(x), rnaData)

    fltData = filter(x->x<cutOff, rnaData)

    if format==:Int
	    return Integer.(round.(fltData))
    else
        return fltData
    end

end


"""
Function to load in two sets of experimental data, filtering out missings/NaNs 
from both sets together to maintain equal length.
"""
function load_data(File1::String, File2::String, Folder::String, cutOff1::Float64=Inf, cutOff2::Float64=Inf)

    # Load in the two sets of data, robust to inclusion/exclusion of ".csv"
    if occursin(".csv",File1)
        data1 = DelimitedFiles.readdlm(Folder*File1)[:]
    else
        data1 = DelimitedFiles.readdlm(Folder*File1*".csv")[:]
    end
    if occursin(".csv",File2)
        data2 = DelimitedFiles.readdlm(Folder*File2)[:]
    else
        data2 = DelimitedFiles.readdlm(Folder*File2*".csv")[:]
    end

    if !isequal(length(data1),length(data2))
        println("Two data files of unequal length")
        return [],[]
    end

    L = length(data1)
    data1Filt = [data1[ii] for ii=1:L if !isnan(data1[ii]) && !isnan(data2[ii]) && data1[ii]<cutOff1 && data2[ii]<cutOff2]
    data2Filt = [data2[ii] for ii=1:L if !isnan(data1[ii]) && !isnan(data2[ii]) && data1[ii]<cutOff1 && data2[ii]<cutOff2]

    return Int.(round.(data1Filt)), Int.(round.(data2Filt))

end


"""
Function to evaluate the log-likelihood of the data, given the standard model
with a particular set of parameters.
"""
function log_likelihood(parameters, data)
    
    Nmax = Integer(round(Base.maximum(data)))    # Maximum value in the data
    P = solvemaster(parameters,Nmax+1)
    # P = solvemaster(parameters)
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
Function to evaluate the log-likelihood of the data, given the compound model
with a particular set of parameters.
"""
function log_likelihood_compound(baseParams, distParams, distFunc, idx, data; lTheta::Integer=250, cdfMax::AbstractFloat=0.999)
    
    Nmax = Integer(round(Base.maximum(data)))+1    # Maximum value in the data
    P = solvecompound(baseParams, distParams, distFunc, idx; N=Nmax, lTheta=lTheta, cdfMax=cdfMax)
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
Function to perform the optimization maximising the logPfunc.
"""
function maximumlikelihood(x0, logPfunc, upper=:Auto, verbose=false)

    func(x) = 1/logPfunc(x)
    nx = length(x0)
    if upper==:Auto
        upper = Inf.*ones(nx)
    end
    res = optimize(func, zeros(nx), upper, x0)
    if  false #Optim.f_calls(res) == 1
        res = optimize(func, x0)
    end

    optPrms = Optim.minimizer(res)

    if verbose
        println(Printf.@sprintf("AIC value of %.2f",-2*logPfunc(optPrms)+2*4))
    end
    return optPrms

end


"""
Function to perform the MCMC metropolis algorithm.
"""
function mcmc_metropolis(x0::AbstractArray, logPfunc::Function, Lchain::Integer;
                         propStd::Union{AbstractFloat,AbstractArray}=0.1,
                         step::Integer=500, burn::Integer=500, printFreq::Integer=10000,
                         prior=:none, verbose::Bool=true, scaleProp::Bool=true)
    
	if length(size(x0)) > 1 # restart from old chain
        if verbose; println("Restarting from old chain"); end
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
        	logpOld += log(Distributions.pdf(prr,xOld[ip]))
        end
    end
    
    for ii=2:Lchain
        if scaleProp
            proposal = MvNormal(propStd.*xOld)
        else
            if isa(propStd, Array)
                proposal = MvNormal(propStd)
            else
                proposal = MvNormal(propStd.*ones(size(xOld)))
            end
        end
        xNew = xOld + Distributions.rand(proposal)

        # Reject if any x are negative
        if any(x->x<0, xNew); continue; end

        # Evaluate new log-likelihood
        if prior == :none
	    	logpNew = logPfunc(xNew)
	    else
	        logpNew = 0.0
	        for  (ip,prr) in enumerate(prior)
	        	logpNew += log(Distributions.pdf(prr,xNew[ip]))
	        end
	        if !isinf(logpNew); logpNew += logPfunc(xNew); end
	    end
        a = exp(logpNew - logpOld)

        if Base.rand() < a
            xOld = xNew
            logpOld = logpNew
            acc += 1
        end
        chain[ii,:] = xOld
        
        if ii % printFreq == 0
            if verbose
                Printf.@printf("Completed iteration %i out of %i. \n", ii,Lchain)
            end
        end
    end
    
    if verbose
        Printf.@printf("Acceptance ratio of %.2f (%i out of %i).\n", acc/Lchain,acc,Lchain)
    end

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
            proposal = MvNormal(propVar.*xOld)
	        xNew = xOld + Distributions.rand(r[Threads.threadid()], proposal)

            # Reject if any x are negative
            if any(x->x<0, xNew); continue; end

            # Evaluate new log-likelihood
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

	        if Base.rand(r[Threads.threadid()]) < a
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


"""
Function to prune the MCMC chain, removing the burn-in samples and thinning the remainder.
"""
function chain_reduce(chains; burn::Integer=500, step::Integer=500)

    tmp = Array{Any,1}(undef,4)
	for ii=1:length(chains)
    	tmp[ii] = chains[ii][burn:step:end,:]
    end

    vcat(tmp...)

end


"""
Function to find the MAP from the posterior distribution.
Finds independently for each parameter.
"""
function find_MAP(chain; idx=1)

    if length(size(chain))>1
        data = chain[:,idx]
    else
        data = copy(chain)
    end

    # Perform the KDE
    kdeObj = kde_wrpr(data)

    # Find MAP via optimisation
    func = x->KDE.pdf(kdeObj,x[1])
    # s = std(data)
    # m = mean(data)
    # res = optimize(func, [m-s], [m+s], [m])
    # MAP = Optim.minimizer(res)

    x = collect(range(min(data...),stop=max(data...),length=10000))
    y = map(func,x)
    MAP = x[y.==maximum(y)][1]

    # Check for multi-modality, warning if true

    return MAP

end


"""
Function to find the MAP from the posterior distribution.
Finds independently for each parameter.
"""
function credibleintervals(chain; idx=1, spread=0.682)

    if length(size(chain))>1
        data = chain[:,idx]
    else
        data = copy(chain)
    end

    # Perform the KDE
    kdeObj = kde_wrpr(data)
    func = z->KDE.pdf(kdeObj,z)
    x = collect(range(min(data...),stop=max(data...),length=10000))
    dx = x[2]-x[1]
    y = map(func,x)

    # Move out from MAP in both directions
    # Stop once intervals contain "spread" of probability
    MAP = find_MAP(data)
    xMin = last(x[x.<MAP])
    xMax = first(x[x.>MAP])
    tot = 0.5*dx*(KDE.pdf(kdeObj,xMin)+KDE.pdf(kdeObj,xMax))
    flag = false
    while tot<spread
        xMinN = xMin-dx
        xMaxN = xMax+dx
        if xMinN<0.0
            flag = true
            break
        end
        if KDE.pdf(kdeObj,xMinN)>KDE.pdf(kdeObj,xMaxN)
            tot += 0.5*dx*(KDE.pdf(kdeObj,xMin)+KDE.pdf(kdeObj,xMinN))
            xMin = xMinN
        else
            tot += 0.5*dx*(KDE.pdf(kdeObj,xMax)+KDE.pdf(kdeObj,xMaxN))
            xMax = xMaxN
        end
    end

    if flag
        sort!(data)
        idx = Int(round(spread*length(data)))
        xMin = 0.0
        xMax = data[idx]
    end

    return xMin,xMax

end


"""
Function to perform optimum parameter fitting.
"""
function optfit(x0, lossFunc::Function)

    nx = length(x0)
    res = optimize(lossFunc, zeros(nx), Inf.*ones(nx), x0)

    optPrms = Optim.minimizer(res)
end


"""
Function to evaluate the difference between a histogram of the data and a given distribition.
"""
function dist_loss_func(data, distribution::Distribution, p::Integer=1)

    x,y = genpdf(Integer.(round.(data)))
    xmax = Integer(round(max(Base.maximum(x), invlogcdf(distribution, log(0.999)))))
    y = [zeros(Integer(Base.minimum(x)))..., y...]

    loss = 0.0
    for ix = 0:xmax
        try
            loss += abs(y[ix+1] - pdf(distribution,ix))^p
        catch
            loss += abs(pdf(distribution,ix))^p
        end
    end

    return loss

end


"""
Function to evaluate the KUllback-Leibler divergence between histogrammed data and a distribution
"""
function KL_loss_func(data, distribution::Distribution)

    x,y = genpdf(Integer.(round.(data)))
    y = [zeros(Integer(Base.minimum(x)))..., y...]

    loss = 0.0
    for ix = 0:Integer(Base.maximum(x))
        if !iszero(y[ix+1])
            loss += y[ix+1]*log2(y[ix+1]/pdf(distribution,ix))
        end
    end

    return loss

end