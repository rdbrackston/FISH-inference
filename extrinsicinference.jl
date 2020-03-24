# Functions for inferring a compounding distribution via the moments


"""
Run the full estimation procedure using the moments and maximum entropy.
"""
function maxentropyestimation(data, K::Int=5)

    N = Int(round(1.1*maximum(data)))
    μ = moments(data,K)
    Λ = maxentropy(μ,N,K)
    q = entropydist(Λ, N)

end

"""
Solves the dual function optimization to obtain the lagrange multipliers λᵢ
"""
function maxentropy(μData, N::Int, K::Int=5)

    # Define the dual function to be minimised
    function dual(λ)
        Z = 0.0
        for x in 0:N-1
            Z += exp(-sum([λ[k]*x^k for k in 1:K]))
        end
        return log(Z) + LinAlg.dot(λ,μData)
    end

    # Perform the optimisation over λᵢ
    result = optimize(dual, zeros(K))
    return result.minimizer

end

"""
Generates a discrete distribution from the lagrange multipliers λᵢ
"""
function entropydist(λ, N)

    # Calculate the distribution given these λᵢ
    K = length(λ)
    q = zeros(N)
    for n in 0:N-1
        nk = [n^k for k in 1:K]
        q[n+1] = exp(-1 - LinAlg.dot(λ,nk))
    end

    return q./sum(q)

end


"""
Return the first M non-central moments of the samples in data.
"""
function moments(data, M)

    L = length(data)
    return [sum(data.^k) for k in 1:M]./L

end

"""
Return the lower triangular matrix P of Stirling numbers associated with the
moments of a compound and extrinsic distribution.
"""
function poissonmatrix(n)
    P = zeros(n,n)
    for ii=1:n
        for jj=1:ii
            P[ii,jj] = stirlings1(ii,jj)
        end
    end
    return P
end

"""
Return the lower triangular matrix T of Stirling numbers and rising factorials
associated with the moments of a compound and extrinsic distribution.
"""
function telegraphmatrix(prms, n)
    P = poissonmatrix(n)
    ν0,ν1 = prms
    for ii=1:n
        for jj=1:ii
            P[ii,jj] *= fracrise(ν1,ν0+ν1,jj)
        end
    end
    return P
end
