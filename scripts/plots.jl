using CairoMakie
using LaTeXStrings
using Printf

function theme_article()
    merge(
        theme_latexfonts(),
        Theme(
            fontsize = 16,
            Axis = (
                xlabelsize = 18,
                ylabelsize = 18,
                titlesize = 18,
                xticklabelsize = 13,
                yticklabelsize = 13,
                spinewidth = 0.8,
                xtickwidth = 0.8,
                ytickwidth = 0.8,
                rightspinevisible = false,
                topspinevisible = false,
                xgridvisible = true,
                ygridvisible = true,
                xgridcolor = (:black, 0.08),
                ygridcolor = (:black, 0.08),
                titlegap = 8,
            ),
            Legend = (
                labelsize = 16,
                framevisible = false,
            ),
            Lines = (linewidth = 2.2,),
            Scatter = (markersize = 8,)
        )
    )
end

function article_plot(
    data;
    xlabel = "x",
    ylabel = "y",
    title = "",
    xscale = identity,
    yscale = identity,
    scatter = false,
    size = (650, 400),
    padding = (10, 15, 15, 15),  # left, right, bottom, top
    digits = 2,
    xticks = nothing,
    yticks = nothing,
    savepath = nothing,
)
    with_theme(theme_article()) do
        fig = Figure(size = size, figure_padding = padding)

        if xticks === nothing
            all_x = reduce(vcat, [collect(d[1]) for d in data])
            xticks = range(minimum(all_x), maximum(all_x), length = 6)
        end

        if yticks === nothing
            all_y = reduce(vcat, [collect(d[2]) for d in data])
            yticks = range(minimum(all_y), maximum(all_y), length = 6)
        end

        xtickformat = xscale == identity ?
            (xs -> [@sprintf("%.*f", digits, x) for x in xs]) :
            automatic

        ytickformat = yscale == identity ?
            (ys -> [@sprintf("%.*f", digits, y) for y in ys]) :
            automatic

        ax = Axis(
            fig[1, 1],
            xlabel = xlabel,
            ylabel = ylabel,
            title = title,
            xscale = xscale,
            yscale = yscale,
            xticks = xticks,
            yticks = yticks,
            xtickformat = xtickformat,
            ytickformat = ytickformat,
            yticklabelspace = 28.0,
        )

        styles = [:solid, :dash, :dot, :dashdot]

        for (i, d) in enumerate(data)
            if length(d) == 3
                x, y, label = d
                style = styles[mod1(i, length(styles))]
            else
                x, y, label, style = d
            end

            lines!(ax, x, y, linestyle = style, label = label)

            if scatter
                scatter!(ax, x, y)
            end
        end

        has_labels = any(d -> !isnothing(d[3]), data)

        if has_labels
            Legend(fig[1, 2], ax, valign = :center)
            colsize!(fig.layout, 2, Auto())
        end

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end

x = range(0, 10, length = 100)
y = sin.(x)
z = cos.(x)

data = [
    (x, y, "sin(x)", :solid),
    (x, z, "cos(x)", :dash)
]

fig = article_plot(
    data,
    xlabel = L"x",
    ylabel = L"\mathit{y}\,[\mathrm{m}]",
    title = "Example Plot",
    digits = 1,
    savepath = "example_plot.pdf"
)

display(fig)
