"""
Function to compute the hypergeometric function using a Maclaurin series.
Adapted from https://github.com/JuliaApproximation/SingularIntegralEquations.jl/src/HypergeometricFunctions/specialfunctions.jl
Unused, as generally inferior to the GSL implemented version.
"""
function hypergeom1F1(a::Float64,b::Float64,z::Float64)
    
    S₀,S₁,err,j = 1.0,1.0+a*z/b,1.0,1
    while err > 1eps(Float64)
        rⱼ = inv(j+1.0)
        rⱼ *= a+j
        rⱼ /= b+j
        S₀,S₁ = S₁,S₁+(S₁-S₀)*rⱼ*z
        err = abs((S₁-S₀)/S₀)
        j+=1
    end
    return S₁
end