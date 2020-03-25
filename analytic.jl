"""
Function to solve the master equation at steady state, returning the probability
mass function. Evaluation uses the recursive implementation of the hypergeometric
function. This is slower but more robust than the GSL version.
"""
function solvemaster2(parameters, N=:auto::Union{Symbol,Int64}, verbose=false)

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
    end

    a = λ/δ
    b = (λ+ν)/δ
    c = K/δ

    P = zeros(N)
    global F = GSL.hypergeom(a,b,-c)
    P[1] = F
    for n=1:N-1
        M = max(5,Int(round(100*n/c)))
        F = hypfun_rec_miller(a+n-1,b+n-1,-c,1,M,F)
        P[n+1] = c^n * F/factorial(big(n))
        # Use Stirling's approximation for n!, splitting power computation into 2
        if Base.isinf(P[n+1])
            n1 = floor(n/2)
            n2 = floor(n/2) + n%2
            x = c*ℯ/n
            P[n+1] = F * x^n1 * x^n2 / sqrt(2*n*pi)
            # Now split power computation into 3
            if Base.isinf(P[n+1])
                n1 = floor(n/3)
                n2 = floor(n/3)
                n3 = floor(n/3) + n%3
                x = c*ℯ/n
                P[n+1] = F * x^n1 * x^n2 * x^n3 / sqrt(2*n*pi)
            end
        end
        for m=0:n-1
            P[n+1] *= (a+m)/(b+m)
        end
    end

    return P./sum(P)

end


"""
Computes the hypergeometric function 1F1(a+k;b+k;z), for k a large
positive integer, using M(a;b;z) and Miller's algorithm, with
n a number much larger than k.

Based on code copyright John W. Pearson 2014
"""
function hypfun_rec_miller(a::AbstractFloat,b::AbstractFloat,z::AbstractFloat,k::Integer, n=:auto, sol=:none)

    if n==:auto
        n = Int(50*k)
    end

    # Initialise f and v in Miller's algorithm
    f = big.(zeros(k+1,1))
    v = big.(zeros(n+1,1))

    # Input minimal solution with k=0
    if sol==:none
        f1 = GSL.hypergeom(a,b,z)/SpecialFunctions.gamma(b)
    else
        f1 = sol/SpecialFunctions.gamma(big(b))
    end
    v[end] = 0
    v[end-1] = 1

    # Compute recurrence backwards
    for i2 = 2:n
        v[n+1-i2] = (v[n+3-i2] + (b+n+1-i2-z-1)*v[n+2-i2]/((a+n+1-i2)*z)) * ((a+n+1-i2)*z)
    end

    # Return solution as last line of Miller's algorithm
    return f1/v[1]*v[k+1]*SpecialFunctions.gamma(big(b+k))

end


"""
Function to solve the master equation at steady state, returning the probability
mass function. Evaluation uses the analytical expression for p(n) where possible
before solving the matrix equation for the remaining higher values of n.
"""
function solvemaster(parameters, N=:auto::Union{Symbol,Int64}, verbose=false)

	if verbose
    	@printf("Solving master equation via the combined method.\n")
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
	    	   @printf("Hypergeom1f1 failed, solving matrix equation for N > %i.\n",T)
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
function solvemaster_full(parameters, N)

	# Assign the parameters
	λ = big(parameters[3])
    ν = big(parameters[4])
    K₀ = big(parameters[1])
    K₁ = big(parameters[2])

	P = zeros(BigFloat,N)
	Max = 150
    for n=0:N-1
    	for r=0:n
    		if r>Max && (n-r)>Max
                # Evaluate using Stirling's approximation
    			P[n+1] +=  big(ℯ*K₁/(n-r))^(n-r) * big(ℯ*(K₀-K₁)/r)^r * GSL.hypergeom(ν+r,λ+ν+r,K₁-K₀) *
    					   fracrise(ν,ν+λ,r) * ℯ^(-K₁)  / (2π * sqrt(r*(n-r)))
    		else
    			P[n+1] += K₁^big(n-r) * (K₀-K₁)^big(r) * big(GSL.hypergeom(ν+r,λ+ν+r,K₁-K₀)) *
    					  fracrise(ν,ν+λ,r) * exp(-K₁) / (factorial(big(n-r)) * factorial(big(r)))
    		end
    	end
    end

    P = P./sum(P)

end
function solvemaster_full2(parameters, N)

    # Assign the parameters
    λ = big(parameters[3])
    ν = big(parameters[4])
    K₀ = big(parameters[1])
    K₁ = big(parameters[2])

    P = zeros(BigFloat,N)
    for n=0:N-1
        F = GSL.hypergeom(ν,λ+ν,K₁-K₀)
        F = big(F)
        P[n+1] += K₁^big(n) * F * exp(-K₁) / factorial(big(n))
        for r=1:n
            # M = max(5,Int(round(100*r)))
            M = 500
            F = hypfun_rec_miller(ν+r-1,λ+ν+r-1,K₁-K₀,1,M,F)
            P[n+1] += K₁^big(n-r) * (K₀-K₁)^r * F * fracrise(ν,ν+λ,r) *
                exp(-K₁) / (factorial(big(n-r)) * factorial(big(r)))
        end
    end

    P = P./sum(P)

end
