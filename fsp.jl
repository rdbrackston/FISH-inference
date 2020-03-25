"""
Function to solve the master equation at steady state, returning the probability
mass function. Uses the steady state finite state projection algorithm.
"""
function solvemaster_fsp(parameters, N=:auto::Union{Symbol,Int64}, verbose=false)

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

    P = LinAlg.nullspace(A)
    P = abs.(P[1:N] + P[N+1:end])
    P = P./sum(P)

end


"""
Function to assemble the A matrix of the finite state space master equation.
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
