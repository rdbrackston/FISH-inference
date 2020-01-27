# Functions for evaluating the leaky gene distribution using Lucy's recurrence relations

"""
Compute leaky gene distribution via the recurrence method.
Compute up to copy number N with M terms in the recursion
"""
function recurrence(prms, N, M)

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
	function gg1(n, x, y) #calculate interates for g_0
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
	println("Calculated generating functions")

	function riser(x,n)  #nth rise of x divided by n!
	    p = BigFloat(1)
	    for i in 0:(n - 1)
	        # t = BigFloat(x + i)
	        p = p * BigFloat(x + i) / (i + 1)
	    end
	    return p
	end
	# Only obvious way to reduce cost is to stop iteration in k once s has converged
	function q(n) #function for computing the steady-state solution
	    s = BigFloat(0)
	    for k in 0:M-N-1
	        i = n + k
	        ds = (G[i + 1] * (-1)^(k) * riser(k+1,n))
	        # if isequal(n,40); println(ds/s); end
	        abs(ds/s)<0.01 && break
	        s += ds
	    end
	    return s
	end

	# Back-calculate distribution from generating functions
	P = [q(n) for n=0:(N-1)]
	return P

end

