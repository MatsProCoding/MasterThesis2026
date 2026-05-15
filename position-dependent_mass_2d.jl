if !isdefined(Main, :Project)
    include("../src/Project.jl")
end

using .Project.Math
using .Project.Solver_2D
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

ms = [50, 100, 200, 400, 600]

x0, Lx = 0.0, 10.0
y0, Ly = 0.0, 10.0

u(x, y) = (2 / sqrt(Lx * Ly)) * sin(pi * (x) / (Lx)) * sin(pi * (y) / (Ly))
 
mx(x,y) =  0.15 *(1 + 0.1*sin(2pi*x/Lx)*cos(2pi*y/Ly))
my(x,y) =  0.067*(1 + 0.1*sin(2pi*x/Lx)*cos(2pi*y/Ly))


# Reference operator:
# Lf(x) = d/dx ( (1/m(x)) * d/dx u(x) ) + d/dy ( (1/m(y)) * d/dy u(y) )
function L_ref(x, y)
    term_x = ForwardDiff.derivative(
        xx -> (1.0 / mx(xx, y)) * ForwardDiff.derivative(t -> u(t, y), xx),
        x
    )

    term_y = ForwardDiff.derivative(
        yy -> (1.0 / my(x, yy)) * ForwardDiff.derivative(t -> u(x, t), yy),
        y
    )

    return term_x + term_y
end

# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

hs = Float64[]
rel_errs = Float64[]      
abs_rmses = Float64[]
max_errs = Float64[]
orders = Float64[]

x_plot = nothing
y_plot = nothing
Lref_plot = nothing
Lnum_plot = nothing

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

for m in ms
    n = m

    x_int, y_int, x_half, y_half, dx, dy = make_grid(x0, Lx, y0, Ly, m, n)

    D1_x = build_FD(m, dx)
    D1_y = build_FD(n, dy)

    D_x, D_y = make_2D(D1_x, D1_y, m, n)

    coord_x, coord_y = mass_coord_order(x_int, y_int, x_half, y_half)

    M_x, _ = build_mass(coord_x, mx)
    M_y, _ = build_mass(coord_y, my)

    L_x = -D_x' * M_x * D_x
    L_y = -D_y' * M_y * D_y
    L = L_x + L_y

    coord_int = coord_order(x_int, y_int)
    u_vec = Float64[u(x, y) for (x, y) in coord_int]

    Lu_num = L * u_vec
    Lu_ref_vec = Float64[L_ref(x, y) for (x, y) in coord_int]

    err = Lu_num - Lu_ref_vec
    abs_rmse_val = sqrt(mean(err.^2))
    rel_rmse_val = abs_rmse_val / sqrt(mean(Lu_ref_vec.^2))
    maxerr = maximum(abs.(err))

    push!(hs, dx)
    push!(abs_rmses, abs_rmse_val)
    push!(rel_errs, rel_rmse_val)
    push!(max_errs, maxerr)


    if m == 200
        global x_plot = x_int
        global y_plot = y_int


        global Lref_plot = reshape(Lu_ref_vec, n - 1, m - 1)
        global Lnum_plot = reshape(Lu_num, n - 1, m - 1)
    end
end

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
    rpad("N", 8),
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
@assert y_plot !== nothing
@assert Lref_plot !== nothing
@assert Lnum_plot !== nothing

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

            Axis3 = (
                xlabelsize = 18,
                ylabelsize = 18,
                zlabelsize = 18,
                titlesize = 18,
                xticklabelsize = 13,
                yticklabelsize = 13,
                zticklabelsize = 13,
                xgridcolor = (:black, 0.08),
                ygridcolor = (:black, 0.08),
                zgridcolor = (:black, 0.08),
                protrusions = (40, 40, 30, 15),
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

# --------------------------------------------------
# Figur 1: Error / konvergens
# --------------------------------------------------
with_theme(theme_article()) do
    fig1 = Figure(size = (750, 400), figure_padding = (10, 5, 15, 15))

    ax1 = Axis(
            fig1[1, 1],
            xlabel = L"m",
            ylabel = "Relative Error",
            title = "Convergence behaviour",

            xticks = range(minimum(ms), maximum(ms), length = 6),
            yticks = range(0, maximum(rel_errs), length = 5),

            yticklabelspace = 70.0,

            ytickformat = ys -> [
                y == 0 ? "0" :
                latexstring(@sprintf("%.1f", y / 10.0^floor(log10(y))), " \\times 10^{", floor(Int, log10(y)), "}")
                for y in ys
            ],

            xminorticksvisible = true,
            yminorticksvisible = true,
        )
    Makie.lines!(ax1, ms, rel_errs, linestyle = :solid, label = "Relative error")
    Makie.scatter!(ax1, ms, rel_errs)
    Makie.lines!(ax1, ms, order2_line, linestyle = :dash, label = L"\mathcal{O}(h^{2})")

    Legend(fig1[1, 2], ax1, valign = :center)
    colsize!(fig1.layout, 2, Auto())

    save("position_dependent_mass_2d_error.pdf", fig1)
    display(fig1)
end

# --------------------------------------------------
# Figur 2: 3D analytisk + numerisk
# --------------------------------------------------

# --------------------------------------------------
# Figur 2: Analytical vs Numerical contour plots
# --------------------------------------------------

with_theme(theme_article()) do
    fig2 = Figure(size = (1000, 420), figure_padding = (10, 15, 15, 15))

    cmin = min(minimum(Lref_plot), minimum(Lnum_plot))
    cmax = max(maximum(Lref_plot), maximum(Lnum_plot))
    levels = range(cmin, cmax, length = 12)

    # -------------------------
    # Venstre: Analytical
    # -------------------------
    ax1 = Axis(
        fig2[1, 1],
        xlabel = L"x",
        ylabel = L"y",
        title = "Analytical",
        aspect = DataAspect()
    )

    cf1 = CairoMakie.contourf!(
        ax1,
        x_plot, y_plot, Lref_plot;
        levels = levels,
        colormap = :inferno
    )

    CairoMakie.contour!(
        ax1,
        x_plot, y_plot, Lref_plot;
        levels = levels,
        color = :black,
        linewidth = 1.0
    )

    # -------------------------
    # Høyre: Numerical
    # -------------------------
    ax2 = Axis(
        fig2[1, 2],
        xlabel = L"x",
        ylabel = L"y",
        title = "Numerical",
        aspect = DataAspect()
    )

    cf2 = CairoMakie.contourf!(
        ax2,
        x_plot, y_plot, Lnum_plot;
        levels = levels,
        colormap = :inferno
    )

    CairoMakie.contour!(
        ax2,
        x_plot, y_plot, Lnum_plot;
        levels = levels,
        color = :black,
        linewidth = 1.0
    )

    # Felles colorbar
    Colorbar(
        fig2[1, 3],
        cf2,
        label = L"L[u]"
    )

    colgap!(fig2.layout, 20)

    save("position_dependent_mass_2d_contours.pdf", fig2)
    display(fig2)
end