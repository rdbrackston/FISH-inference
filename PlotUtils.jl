# Plotting utilities for use with the parameter inference

using RecipesBase, Reexport, Discretizers

@reexport using Plots
import Plots: _cycle
using Plots.PlotMeasures

@userplot CorrPlot


"""
Function to plot the full chains and posterior distributions.
"""
function plot_chain(chain, MAP=:none, Ints=:none)

	nVar = size(chain)[2]
	plots = Array{Any,1}(undef,nVar*2)

    # Set MAP to the mean if not provided
    if isequal(MAP,:none)
        MAP = mean(chain,dims=1)
        lbl = "Mean"
    else
        lbl = "MAP"
    end


	for ii=1:nVar
		plots[ii] = plot(chain[:,ii], legend=false, title=Printf.@sprintf("Parameter %i",ii));

		smpls = chain[:,ii]
		kdeObj = kde_wrpr(smpls)
		x = collect(range(Base.minimum(smpls),stop=Base.maximum(smpls),length=100))
		y = map(z->KDE.pdf(kdeObj,z),x)
		if ii ==1
			tmp = plot(x,y, label="Posterior", legend=:top);
			plot!(tmp, [MAP[ii],MAP[ii]],[0,Base.maximum(y)], label=lbl, line=(2,:black));
		else
			tmp = plot(x,y, legend=false);
			plot!(tmp, [MAP[ii],MAP[ii]],[0,Base.maximum(y)], line=(2,:black));
		end
        if !isequal(Ints,:none)
            plot!(tmp, [Ints[ii][1],Ints[ii][1]],[0,Base.maximum(y)], line=(2,:black,:dash), label="");
            plot!(tmp, [Ints[ii][2],Ints[ii][2]],[0,Base.maximum(y)], line=(2,:black,:dash), label="");
        end
		plots[ii+nVar] = tmp;

	end

	if nVar == 2
		plt = plot(plots[1],plots[2],plots[3],plots[4],
			       layout=(2,nVar), size=(800,800))
	elseif nVar == 3
		plt = plot(plots[1],plots[2],plots[3],plots[4],plots[5],plots[6],
			       layout=(2,nVar), size=(1200,800))
	elseif nVar == 4
		plt = plot(plots[1],plots[2],plots[3],plots[4],plots[5],plots[6],plots[7],plots[8],
			       layout=(2,nVar), size=(1600,800))
	elseif nVar == 5
		plt = plot(plots[1],plots[2],plots[3],plots[4],plots[5],plots[6],plots[7],plots[8],plots[9],plots[10],
			       layout=(2,nVar), size=(2000,800))
	elseif nVar == 6
		plt = plot(plots[1],plots[2],plots[3],plots[4],plots[5],plots[6],plots[7],plots[8],plots[9],plots[10],plots[11],plots[12],
			       layout=(2,nVar), size=(2400,800))
	else
		Printf.@printf("nVar of %i not currently catered for.", nVar)
		plt = plot()
	end

	return plt

end


"""
Function to plot a comparison of the distributions from the data and the parameter inference.
Chain can either be the full MCMC chain, or a vector of parameters.
"""
function plot_inference(chain, data, idx; guess=[],hyper=[])

    dataInt = Integer.(round.(data))
	(x,y) = genpdf(dataInt)
	# plt = plot(x,y, marker=(2,:cross),line=0, label="Data")
    plt = bar(x,y, label="Data", xlabel="mRNA copy no. (n)", ylabel="Probability, P(n)")

    if length(chain)>6
	   inferredParams = mean(chain,dims=1)[:]
    else
        inferredParams = chain
    end
	Q = ModelInference.solvecompound(inferredParams[1:3],inferredParams[4:3+length(idx)],(x,y)->LogNormal(x,y),idx)
	plot!(0:length(Q)-1, Q, label="MCMC inference")
	if length(guess)==3 & length(hyper)==length(idx)
		P = ModelInference.solvecompound(guess,hyper,(x,y)->LogNormal(x,y),idx)
		plot!(0:length(P)-1, P, label="Initial guess")
	end

	return plt

end


"""
Function to plot the posterior distributions along with the priors
"""
function plot_posteriors(chain, priors)

    nVar = size(chain)[2]
    plots = Array{Any,1}(undef,nVar)

    for ii=1:nVar

        smpls = chain[:,ii]
        kdeObj = KDE.kde(smpls)
        x = collect(range(Base.minimum(smpls),stop=Base.maximum(smpls),length=100))
        y = map(z->KDE.pdf(kdeObj,z),x)
        p = map(z->Distributions.pdf(priors[ii],z))

        if ii ==1
            tmp = plot(x,(y), label="Posterior", legend=:top);
            plot!(tmp, [mean(chain,dims=1)[ii],mean(chain,dims=1)[ii]],[0,Base.maximum(y)], label="Mean");
            plot!(x, (p), label="Prior");
        else
            tmp = plot(x,y, legend=false);
            plot!(tmp, [mean(chain,dims=1)[ii],mean(chain,dims=1)[ii]],[0,Base.maximum(y)]);
            plot!(x, (p));
        end
        plots[ii] = tmp;

    end

    plt = plot(plots[1], plots[2], plots[3], plots[4], layout=(1,nVar), size=(1600,400))

end


"""
Utility used as part of corrplot.
"""
function update_ticks_guides(d::KW, labs, i, j, n)
    # d[:title]  = (i==1 ? _cycle(labs,j) : "")
    # d[:xticks] = (i==n)
    d[:xguide] = (i==n ? _cycle(labs,j) : "")
    # d[:yticks] = (j==1)
    d[:yguide] = (j==1 ? _cycle(labs,i) : "")
end

"""
Correlation plot recipe
"""
@recipe function f(cp::CorrPlot)
    mat = cp.args[1]
    n = size(mat,2)
    C = cor(mat)
    labs = pop!(plotattributes, :label, [""])

    link := :x  # need custom linking for y
    layout := (n,n)
    size := (1200,1200)
    legend := false
    foreground_color_border := nothing
    margin := 1mm
    titlefont := font(11)
    fillcolor --> Plots.fg_color(plotattributes)
    linecolor --> Plots.fg_color(plotattributes)
    markeralpha := 0.4
    grad = cgrad(get(plotattributes, :markercolor, cgrad()))
    indices = reshape(1:n^2, n, n)'
    title = get(plotattributes,:title,"")
    title_location = get(plotattributes, :title_location, :center)
    title := ""

    # histograms on the diagonal
    for i=1:n
        @series begin
            if title != "" && title_location == :left && i == 1
                title := title
            end
            seriestype := :histogram
            subplot := indices[i,i]
            grid := false
            xformatter --> ((i == n) ? :auto : (x -> ""))
            yformatter --> ((i == 1) ? :auto : (y -> ""))
            update_ticks_guides(plotattributes, labs, i, i, n)
            view(mat,:,i)
        end
    end

    # scatters
    for i=1:n
        ylink := setdiff(vec(indices[i,:]), indices[i,i])
        vi = view(mat,:,i)
        for j = 1:n
            j==i && continue
            vj = view(mat,:,j)
            subplot := indices[i,j]
            update_ticks_guides(plotattributes, labs, i, j, n)
            if i > j
                #below diag... scatter
                @series begin
                    seriestype := :scatter
                    markercolor := grad[0.5 + 0.5C[i,j]]
                    smooth := true
                    markerstrokewidth --> 0
                    xformatter --> ((i == n) ? :auto : (x -> ""))
                    yformatter --> ((j == 1) ? :auto : (y -> ""))
                    vj, vi
                end
            else
                #above diag... hist2d
                @series begin
                    seriestype := get(plotattributes, :seriestype, :histogram2d)
                    if title != "" && i == 1 && ((title_location == :center && j == div(n,2)+1) || (title_location == :right && j == n))
                        if iseven(n)
                            title_location := :left
                        end
                        title := Printf.@sprintf("Parameter %i",i)
                    end
                    xformatter --> ((i == n) ? :auto : (x -> ""))
                    yformatter --> ((j == 1) ? :auto : (y -> ""))
                    vj, vi
                end
            end
        end
    end
end