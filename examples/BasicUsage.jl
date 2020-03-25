# Basic usage examples for evaluating distributions and sampling

using TranscriptionModels,  Plots
gr()

## Solve the master equation for the Telegraph distribution
prms = [40,0.6,0.4]
P = solvemaster(prms,60)
plot(0:length(P)-1,P, xlabel="Copy no., n",ylabel="Probability, p(n)",
    label="Analytic solution")

# Sample from the distribution and compare the histogram
dist = TelegraphDist(prms...)
samps = rand(dist,800)
x,y = TranscriptionModels.genpdf(samps)
plot!(x,y, label="Sampled histogram", line=0,marker=3)


## Compare analytic and fsp solutions
