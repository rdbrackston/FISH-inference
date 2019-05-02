# Functions associated with defining a samplable probability distribution for the Telegraph model.


"""
Telegraph model distribution Type.
"""
struct TelegraphDist{T<:Real} <: DiscreteUnivariateDistribution
	K::T
	λ::T
	ν::T
end


"""
Definition of the rand function to draw samples from the model.
"""
function rand(d::TelegraphDist)

	r = Base.rand()
	s = quantile(d,r)

end
Distributions.rand(d::TelegraphDist) = rand(d::TelegraphDist)


""" Required by distributions. """
sampler(d::TelegraphDist) = d


"""
Probability mass function for the Telegraph model.
"""
function pdf(d::TelegraphDist, n::Int)

	a = d.λ
    b = d.λ+d.ν
    c = d.K

    # try # Hypergeom1F1 fails for some argument values
		P = c^n * GSL.hypergeom(a+n,b+n,-c)/factorial(big(n))
    	if Base.isinf(P)    # Use Stirling's approximation for n!
    		P = GSL.hypergeom(a+n,b+n,-c) * (c*ℯ/n)^n / sqrt(2*n*pi)
    	end
        for m=0:n-1
            P *= (a+m)/(b+m)
        end
    #catch
    #	P = 0.0
     #   failed = true
	#end

	return P

end

"""
Logarithm of the probability mass function.
"""
logpdf(d::TelegraphDist, x::Real) = log(pdf(d,x))


"""
Functio to evaluate the cumalitive distribution function for the telegraph model.
"""
function cdf(d::TelegraphDist, n::Int)

	P = 0
	for x=0:n
		P += pdf(d,x)
	end
	return P

end


"""
Function to evaluate the smallest copy number at which the cdf is greater than q.
"""
function quantile(d::TelegraphDist, q::Real)

	C = 0.0
	n = 0
	while C < q
		C += pdf(d,n)
		n += 1
	end

	return n-1

end

""" Required by distributions. """
minimum(d::TelegraphDist) = 0

""" Required by distributions. """
maximum(d::TelegraphDist) = Inf

""" Required by distributions. """
insupport(d::TelegraphDist, x::Real) = x>=0