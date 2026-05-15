if !isdefined(Main, :Project)
    include("../src/Project.jl")
end

using .Project.Math

using SparseArrays
using LinearAlgebra
using ForwardDiff
using Statistics
using Printf
using CairoMakie
using LaTeXStrings

# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
ms = [100, 200, 400, 800, 1600]

x0 = 0.0
Lx = 10.0

masse(x) = 1 + x^2
u(x) = sin(pi * x / Lx)

# Reference operator:
# Lf(x) = d/dx ( (1/m(x)) * d/dx u(x) )
Lf(x) = ForwardDiff.derivative(
    t -> (1 / masse(t)) * ForwardDiff.derivative(u, t),
    x
)

# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------
hs = Float64[]
rel_errs = Float64[]
abs_rmses = Float64[]
max_errs = Float64[]
orders = Float64[]

x_plot = nothing
Lu_ref_plot = nothing
Lu_num_plot = nothing

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
for m in ms
    x_int, x_half, dx = make_grid(x0, Lx, m)

    A = spdiagm(0 => [1 / masse(xi) for xi in x_half])
    D = build_FD(m, dx)
    L = -D' * A * D

    u_num = u.(x_int)
    Lu_ref = Lf.(x_int)
    Lu_num = L * u_num

    err = Lu_num - Lu_ref
    abs_rmse_val = sqrt(mean(err.^2))
    rel_rmse_val = abs_rmse_val / sqrt(mean(Lu_ref.^2))
    maxerr = maximum(abs.(err))

    push!(hs, dx)
    push!(abs_rmses, abs_rmse_val)
    push!(rel_errs, rel_rmse_val)
    push!(max_errs, maxerr)

    if m == ms[end]
        global x_plot = x_int
        global Lu_ref_plot = Lu_ref
        global Lu_num_plot = Lu_num
    end
end

# ------------------------------------------------------------
# Convergence order from relative RMSE
# ------------------------------------------------------------
for i in 2:length(ms)
    p = log(rel_errs[i] / rel_errs[i - 1]) / log(hs[i] / hs[i - 1])
    push!(orders, p)
    @info "order between $(ms[i-1]) -> $(ms[i]): p ≈ $(p)"
end
pushfirst!(orders, NaN)

# ------------------------------------------------------------
# Print tables
# ------------------------------------------------------------
println("\n--- Convergence Table ---")
println(
    rpad("m", 8),
    rpad("h", 14),
    rpad("Abs. RMSE", 18),
    rpad("Rel. RMSE", 18),
    rpad("Max Abs Error", 18),
    "Order"
)

for i in eachindex(ms)
    @printf(
        "%-8d %-14.6e %-18.6e %-18.6e %-18.6e %-6.3f\n",
        ms[i], hs[i], abs_rmses[i], rel_errs[i], max_errs[i], orders[i]
    )
end

@assert x_plot !== nothing
@assert Lu_ref_plot !== nothing
@assert Lu_num_plot !== nothing

# ------------------------------------------------------------
# O(h²) guide using relative RMSE, plotted against m
# ------------------------------------------------------------
c2 = rel_errs[1] * ms[1]^2
order2_line = c2 ./ (ms .^ 2)

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
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
            Scatter = (markersize = 8,),
        )
    )
end

# ------------------------------------------------------------
# Figure 1: Numerical vs analytical
# ------------------------------------------------------------
with_theme(theme_article()) do
    fig1 = Figure(size = (650, 400), figure_padding = (10, 5, 15, 15))

    ax1 = Axis(
        fig1[1, 1],
        xlabel = L"x",
        ylabel = L"L[u]",
        title = "Numerical vs analytical"
    )

    Makie.lines!(
        ax1,
        x_plot,
        Lu_ref_plot,
        linestyle = :solid,
        label = "Analytical"
    )

    Makie.lines!(
        ax1,
        x_plot,
        Lu_num_plot,
        linestyle = :dash,
        label = "Numerical"
    )

    Legend(fig1[1, 2], ax1, valign = :center)
    colsize!(fig1.layout, 2, Auto())

    save("position_dependent_mass_1d_solution.pdf", fig1)
    display(fig1)
end

# ------------------------------------------------------------
# Figure 2: Error / convergence
# ------------------------------------------------------------
with_theme(theme_article()) do
    fig2 = Figure(size = (650, 400), figure_padding = (10, 5, 15, 15))

    ax2 = Axis(
        fig2[1, 1],
        xlabel = L"m",
        ylabel = "Relative Error",
        title = "Convergence behaviour",

        xticks = range(minimum(ms), maximum(ms), length = 6),
        yticks = range(0, maximum(rel_errs), length = 5),

        yticklabelspace = 70.0,

        ytickformat = ys -> [
            y == 0 ? "0" :
            latexstring(
                @sprintf("%.1f", y / 10.0^floor(log10(y))),
                " \\times 10^{",
                floor(Int, log10(y)),
                "}"
            )
            for y in ys
        ],

        xminorticksvisible = true,
        yminorticksvisible = true,
    )

    Makie.lines!(
        ax2,
        ms,
        rel_errs,
        linestyle = :solid,
        label = "Relative error"
    )

    Makie.scatter!(
        ax2,
        ms,
        rel_errs
    )

    Makie.lines!(
        ax2,
        ms,
        order2_line,
        linestyle = :dash,
        label = L"\mathcal{O}(h^{2})"
    )

    Legend(fig2[1, 2], ax2, valign = :center)
    colsize!(fig2.layout, 2, Auto())

    save("position_dependent_mass_1d_error.pdf", fig2)
    display(fig2)
end
