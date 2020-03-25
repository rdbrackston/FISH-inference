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

include("inference.jl")
include("plotutils.jl")
include("telegraphdistribution.jl")
include("utilities.jl")
include("specialcases.jl")
include("recurrence.jl")
include("extrinsicinference.jl")
include("fsp.jl")
include("analytic.jl")
include("compoundmodels.jl")


end # module
