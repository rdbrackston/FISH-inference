# Examples for inferring extrinsic distributions

using TranscriptionModels; const TxModels = TranscriptionModels
using Distributions, Plots
import Printf: @sprintf
gr()


## Run an example inferring a negative binomial distribution
r,p = 5.0,0.2
d = NegativeBinomial(r,p)
samps = rand(d,800)
plot(0:80, x->pdf(d,x), label="True distribution")
x,y = TxModels.genpdf(samps)
plot!(x,y, line=0,marker=3, label="Histogram")
q = maxentropyestimation(samps,6)
plot!(0:length(q)-1, q, label="Maximum entropy estimate")
plot!(xlabel="x",ylabel="f(x)")


## Attempt to infer gamma distributed extrinsic noise
r,p = 5.0,0.2
d = NegativeBinomial(r,p)
samps = rand(d,800)
n = 7 # Number of moments to consider
μd = TxModels.moments(samps,n)
P = TxModels.poissonmatrix(n)
μK = P\μd
N = 80
Λ = TxModels.maxentropy(μK,N,n)
q = TxModels.entropydist(Λ, N)
dG = Gamma(r,(1-p)/p)
plot(0:80,z->pdf(dG,z), label="Actual noise", line=2)
plot!(0:length(q)-1,q, label="Estimated noise", line=2)
plot!(xlabel="Transcription rate, K",ylabel="Probability, f(K)")


## Now attempt to infer bimodal noise
dK = MixtureModel(Gamma[Gamma(2.,1.),Gamma(5.,5.)], [0.2,0.8])
Kvec = rand(dK,800)
samps = zeros(800)
for ii in 1:length(samps)
    d = Poisson(Kvec[ii])
    samps[ii] = rand(d)
end
n = 8 # Number of moments to consider
μd = TxModels.moments(samps,n)
P = TxModels.poissonmatrix(n)
μK = P\μd
N = 80
Λ = TxModels.maxentropy(μK,N,n)
q = TxModels.entropydist(Λ, N)
plot(0:80,z->pdf(dK,z), label="Actual noise")
plot!(0:length(q)-1,q, label="Estimated noise")
plot!(xlabel="Transcription rate, K",ylabel="Probability, f(K)")


## Now no noise at all
K = 20
samps = rand(Poisson(K),800)
n = 2 # Number of moments to consider
μd = TxModels.moments(samps,n)
P = TxModels.poissonmatrix(n)
μK = P\μd
N = 80
Λ = TxModels.maxentropy(μK,N,n)
q = TxModels.entropydist(Λ, N)
plot([K, K],[0,1], label="Actual noise")
plot!(0:length(q)-1,q, label="Estimated noise")
plot!(xlabel="Transcription rate, K",ylabel="Probability, f(K)")


## Now try to recover the beta distributoin from telegraph data
dB = Beta(0.5,0.5)
plot(0:0.1:40,z->pdf(dB,z/40)/40, label="Actual noise", line=(:black,:dash,2))
dT = TelegraphDist([40.,0.5,0.5]...)
samps = rand(dT,800)

for n in 2:2:10
    μd = TxModels.moments(samps,n)
    P = TxModels.poissonmatrix(n)
    μK = P\μd
    N = 80
    Λ = TxModels.maxentropy(μK,N,n)
    q = TxModels.entropydist(Λ, N)
    lab = @sprintf("Estimated noise, n=%i", n)
    plot!(0:length(q)-1,q, label=lab, line=2)
end
plot!(xlabel="Transcription rate, K",ylabel="Probability, f(K)")
