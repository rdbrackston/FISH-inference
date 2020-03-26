# TranscriptionModels.jl
Julia module to model bursty transcription with and without extrinsic noise. The
package consists of a set of methods providing the following features:

-- Evaluation of the discrete steady state probability distributions arising from bursty transcription.
-- Evaluation of these distributions in the presence of extrinsic (parametric)
noise [1].
-- Implementation of the above via analytic solutions, the steady state finite state projection algorithm and a recurrence method [2].
-- Bayesian parameter inference for such models using an MCMC sampling scheme.
-- Inference of extrinsic distributions via a moment-based method.

## Installation
To use the module, first clone this respository,
```bash
git clone https://github.com/rdbrackston/TranscriptionModels.git
```
Then add the path to the file located at `~/.julia/config/startup.jl` with the line:
```julia
push!(LOAD_PATH, <path-to-repository>)
```
The module can then be loaded in julia in the normal manner,
```bash
julia> using TranscriptionModels
```

## Related publications

[1]L. Ham, R. D. Brackston & M. P. H. Stumpf, Extrinsic Noise and Heavy-Tailed Laws in Gene Expression. (2020) *Physical Review Letters* [**124**, 108101](https://doi.org/10.1103/PhysRevLett.124.108101).

[2] L. Ham, D. Schnoerr, R. D. Brackston & M. P. H. Stumpf, Exactly solvable models of stochastic gene expression. (2020) *bioRxiv* [2020.01.05.895359](https://doi.org/10.1101/2020.01.05.895359).
