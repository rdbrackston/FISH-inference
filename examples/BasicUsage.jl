# Basic usage examples for evaluating distributions and sampling

using TranscriptionModels,  Plots
const TxMdls = TranscriptionModels
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


## Compare some analytic and fsp solutions
prms = [5.,40.,0.2,0.1]
P = solvemaster(prms,65) # Leaky Telegraphg model
plot(0:length(P)-1,P)
Q = solvemaster_fsp(prms,65)
plot!(0:length(Q)-1,Q, line=:dash)

prms = [40.,0.2,0.1]
P = solvemaster(prms,65) # Standard Telegraph model
plot(0:length(P)-1,P)
Q = solvemaster_fsp(prms,65)
plot!(0:length(Q)-1,Q, line=:dash)


## Test out the recurrence method
prms = [5.,40.,0.2,0.1]
P = solvemaster_rec(prms,65,300)
plot(0:length(P)-1,P, label="Recurrence")


## Compound models and heavy-tailed analysis
prms = [40.,0.5,0.1]
kStd = 20.0
Q = solvecompound(prms, [kStd], :LogNormal, [1], cdfMax=0.9999, N=200)
plot(0:length(Q)-1,Q, label="Compound model", yscale=:log10)

samps = samplecompound(prms, [kStd], :Telegraph, :LogNormal)
x,y = genpdf(samps)
scatter!(x[y.>0],y[y.>0], label="Sampled distribution")

x,y = genkde_trans(samps)
plot!(x,y, yscale=:log10, label="Kernel density (transformed)")
