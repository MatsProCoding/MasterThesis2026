module Eval

using ..Math

using LinearAlgebra
using Statistics
using Printf
using CairoMakie
using LaTeXStrings

export  sort_eigenpairs
export energy_report, wavefunction_report, convergence_report, wavefunction_report_2D, wave_plot_1D

# --------------------------------------------------
# Basic helpers
# --------------------------------------------------

    function theme_article()
        merge(
            CairoMakie.theme_latexfonts(),
            CairoMakie.Theme(
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
                    framevisible = true,
                ),
                Lines = (linewidth = 2.2,),
                Scatter = (markersize = 8, strokewidth = 0.8, strokecolor = :black)
            )
        )
    end


function sort_eigenpairs(E, Psi)
    perm = sortperm(real(E))
    return real(E[perm]), Psi[:, perm]
end


function degenerate!(E, psi, m, n, dA; tol=1e-9)
        for i in 1:length(E)-1
            if abs(E[i] - E[i+1]) < tol
                psi0 = reshape(psi[:,i], m-1, n-1)
                psi1 = reshape(psi[:,i+1], m-1, n-1)

                add = vec(psi0 + psi1')
                sub = vec(psi0 - psi1')

                psi[:,i]   = normalize_wavefunction(add, dA)
                psi[:,i+1] = normalize_wavefunction(sub, dA)
            end
        end
    end

function observed_order(ms, errs)
    mids = Float64[]
    p = Float64[]

    for i in 1:length(ms)-1
        m1 = ms[i] isa Tuple ? ms[i][1] : ms[i]
        m2 = ms[i+1] isa Tuple ? ms[i+1][1] : ms[i+1]

        push!(mids, sqrt(m1 * m2))
        push!(p, log(errs[i]/errs[i+1]) / log(m2/m1))
    end

    return mids, p
end

function lowest_states_2D(energy_analytical, nmax; nsearch=10)
    candidates = []
    for nx in 1:nsearch, ny in 1:nsearch
        push!(candidates, (energy_analytical(nx, ny), (nx, ny)))
    end
    sort!(candidates, by = first)
    return candidates[1:nmax]
end


# --------------------------------------------------
# Energy report
# --------------------------------------------------

function energy_report(E_numeric, energy_analytical; nmax=length(E_numeric), display_info=false)
    E_num = sort(real(E_numeric))
    nmax = min(nmax, length(E_num))

    if applicable(energy_analytical, 1)

        states = collect(1:nmax)
        E_an = [energy_analytical(n) for n in states]

    elseif applicable(energy_analytical, 1,1)

        candidates = lowest_states_2D(energy_analytical, nmax)
        E_an = first.(candidates)
        states = last.(candidates)

    else
        error("energy_analytical must take 1 or 2 arguments")
    end
    
    abs_errors = abs.(E_num[1:nmax] .- E_an)
    rel_errors = abs.(abs_errors) ./ abs.(E_an)

    rmse_abs = sqrt(mean(abs_errors .^ 2))
    rmse_rel = sqrt(mean(rel_errors .^ 2))

    if display_info
        println("--------------------------------------------------------------------------")
        println(" n    Numerical (eV)     Analytical (eV)     Abs. error       Rel. error")
        println("--------------------------------------------------------------------------")
        for i in 1:nmax
            @printf("%-s   %14.8f   %16.8f   %12.4e   %12.4e\n",
                string(states[i]), E_num[i], E_an[i], abs_errors[i], rel_errors[i])
        end
        println("--------------------------------------------------------------------------")
        @printf("Absolute RMSE = %.6e eV\n", rmse_abs)
        @printf("Relative RMSE = %.6e\n", rmse_rel)
        println("--------------------------------------------------------------------------")
    end

    return (
        n = collect(1:nmax),
        E_numeric = E_num[1:nmax],
        E_analytical = E_an,
        abs_error = abs_errors,
        rel_error = rel_errors,
        rmse_abs = rmse_abs,
        rmse_rel = rmse_rel,
        states = states
    )
end

# --------------------------------------------------
# Wavefunction report
# --------------------------------------------------

function wavefunction_report(
    x, V, E_numeric, Psi_numeric, psi_analytical;
    nmax=min(length(E_numeric), size(Psi_numeric, 2)),
    display_info=false,
    display_plot=true,
    x_window=nothing
)
    dx = x[2] - x[1]

    E_sorted, Psi_sorted = sort_eigenpairs(E_numeric, Psi_numeric)
    nmax = min(nmax, size(Psi_sorted, 2))

    overlaps = Float64[]
    l2_errors = Float64[]

    if x_window === nothing
        x_window = (minimum(x), maximum(x))
    end

    if display_info
        println("---------------------------------------------------------------")
        println(" n        overlap        L2 error")
        println("---------------------------------------------------------------")
    end

    E1 = E_sorted[1]

    p = nothing
    for n in 1:nmax
        psi_num = real(Psi_sorted[:, n])
        psi_num = normalize_wavefunction(psi_num, dx)

        psi_an = [psi_analytical(xi, n) for xi in x]
        psi_an = normalize_wavefunction(psi_an, dx)

        # Phase alignment
        if sum(psi_num .* psi_an) * dx < 0
            psi_num .*= -1
        end

        overlap = abs(sum(psi_num .* psi_an) * dx)
        l2_err = sqrt(sum(abs2, psi_num .- psi_an) * dx)

        push!(overlaps, overlap)
        push!(l2_errors, l2_err)

        if display_info
            @printf("%2d   %12.6f   %12.6e\n", n, overlap, l2_err)
        end

    end

    if display_plot
        p = CairoMakie.with_theme(theme_article()) do
            fig = CairoMakie.Figure(size = (900, 500))
            ax = CairoMakie.Axis(
                fig[1, 1],
                xlabel = "x",
                ylabel = "Energy shifted wavefunctions",
                title = "Wavefunction comparison (shifted by E/E1)"
            )

            for n in 1:nmax
                psi_num = real(Psi_sorted[:, n])
                psi_num = normalize_wavefunction(psi_num, dx)

                psi_an = [psi_analytical(xi, n) for xi in x]
                psi_an = normalize_wavefunction(psi_an, dx)

                if sum(psi_num .* psi_an) * dx < 0
                    psi_num .*= -1
                end

                shift = E_sorted[n] / E1
                CairoMakie.lines!(ax, x, abs2.(psi_num) .+ shift, label = "Num n=$n")
                CairoMakie.lines!(ax, x, abs2.(psi_an) .+ shift, linestyle = :dash, label = "Ana n=$n")
            end

            CairoMakie.axislegend(ax, position = :rt)
            display(fig)
            fig
        end
    end

    #=
    if display_plot
        plot!(p, x, V.(x),
            color=:black,
            linewidth=2,
            label="V(x)")
    end
    =#
    return (
        overlap = overlaps,
        l2_error = l2_errors,
        energies = E_sorted[1:nmax],
        plot = p
    )
    print("Oppdatert")
end


# 2D

function wavefunction_report_2D(
    x,
    y,
    m,
    n,
    shape,
    E_numeric,
    Psi_numeric,
    energy_analytical,
    psi_analytical;
    nmax=min(length(E_numeric), size(Psi_numeric, 2)),
    display_info=false,
    display_plot=true,
    plot_quantity=:probability,   # :probability eller :real
    tol_deg=1e-8
)

    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dA = dx * dy

    E_sorted, Psi_sorted = sort_eigenpairs(E_numeric, Psi_numeric)
    nmax = min(nmax, size(Psi_sorted, 2))

    overlaps = Float64[]
    l2_errors = Float64[]

    if display_info
        println("-------------------------------------------------------------------")
        println(" state        overlap              L2 error")
        println("-------------------------------------------------------------------")
    end

    # analytiske states i stigende energi
    cands = lowest_states_2D(energy_analytical, nmax; nsearch=10)

    p = nothing

    # Degeneracy

    degenerate!(E_sorted, Psi_sorted, m, n, dA)


    for i in 1:nmax
        state = cands[i][2]           # (nx, ny)
        nx, ny = state

        # numerisk state
        psi_num = real(Psi_sorted[:, i])
        psi_num = normalize_wavefunction(psi_num, dA)
        psi_num = reshape(psi_num, length(x), length(y))
        psi_num = psi_num  

        # analytisk state på hele 2D-gridet
        psi_an = psi_analytical.(x, y', nx, ny)
        psi_an = normalize_wavefunction(vec(psi_an), dA)
        psi_an = reshape(psi_an, length(x), length(y))'

        # fiks global fase / fortegn
        overlap = (dot(vec(conj.(psi_num)), vec(psi_an)) * dA)
        
        if abs(overlap) < 1e-3
            nx, ny = ny, nx
            psi_an = psi_analytical.(x, y', nx, ny)
            psi_an = normalize_wavefunction(vec(psi_an), dA)
            psi_an = reshape(psi_an, length(x), length(y))'
            overlap = (dot(vec((psi_num)), vec(psi_an)) * dx * dy)
        end
        
        if real(overlap) < 0
            psi_num .*= -1
            overlap = dot(vec(conj.(psi_num)), vec(psi_an)) * dA
        end

        l2_err = sqrt(sum(abs2, vec(psi_num) .- vec(psi_an)) * dA)

        push!(overlaps, overlap)
        push!(l2_errors, l2_err)

        if display_info
            @printf("%-10s   %14.8f     %14.8e\n", string(state), overlap, l2_err)
        end

    end

    if display_plot
        p = CairoMakie.with_theme(theme_article()) do
            fig = CairoMakie.Figure(size = (1500, 320 * nmax))

            for i in 1:nmax
                state = cands[i][2]
                nx, ny = state

                psi_num = real(Psi_sorted[:, i])
                psi_num = normalize_wavefunction(psi_num, dA)
                psi_num = reshape(psi_num, length(x), length(y))

                psi_an = psi_analytical.(x, y', nx, ny)
                psi_an = normalize_wavefunction(vec(psi_an), dA)
                psi_an = reshape(psi_an, length(x), length(y))'

                overlap = (dot(vec(conj.(psi_num)), vec(psi_an)) * dA)
                if abs(overlap) < 1e-3
                    nx, ny = ny, nx
                    psi_an = psi_analytical.(x, y', nx, ny)
                    psi_an = normalize_wavefunction(vec(psi_an), dA)
                    psi_an = reshape(psi_an, length(x), length(y))'
                    overlap = (dot(vec(psi_num), vec(psi_an)) * dA)
                end
                if real(overlap) < 0
                    psi_num .*= -1
                end

                Z_num = plot_quantity == :real ? psi_num : abs2.(psi_num)
                Z_an = plot_quantity == :real ? psi_an : abs2.(psi_an)

                title_num = "Numerical $(state), E = $(round(E_sorted[i], digits=6))"
                title_an = "Analytical $(state)"

                ax_num = CairoMakie.Axis(fig[i, 1], xlabel = "x", ylabel = "y", title = title_num)
                hm_num = CairoMakie.heatmap!(ax_num, x, y, Z_num, colormap=:inferno)
                CairoMakie.Colorbar(fig[i, 2], hm_num)

                ax_an = CairoMakie.Axis(fig[i, 3], xlabel = "x", ylabel = "y", title = title_an)
                hm_an = CairoMakie.heatmap!(ax_an, x, y, Z_an, colormap=:inferno)
                CairoMakie.Colorbar(fig[i, 4], hm_an)
            end

            display(fig)
            fig
        end
    end

    if display_info
        println("-------------------------------------------------------------------")
    end

    return (
        overlap = overlaps,
        l2_error = l2_errors,
        states = [c[2] for c in cands],
        energies = E_sorted[1:nmax]
    )
end

# --------------------------------------------------
# Convergence report
# builder(m) must return E_numeric
# --------------------------------------------------


function convergence_report(
    ms,
    builder,
    energy_analytical;
    nmax=3,
    display_info=true,
    display_plot=true,
    type="",
    save_plot=false,
)

    # Support both scalar m values and tuple-like entries such as (m, n)
    mvals = [m isa Tuple ? m[1] : m for m in ms]

    E_levels = [Float64[] for _ in 1:nmax]
    abs_errors_levels = [Float64[] for _ in 1:nmax]
    rel_errors_levels = [Float64[] for _ in 1:nmax]

    rmse_abs_vals = Float64[]
    rmse_rel_vals = Float64[]

    result = nothing
    analytics = nothing
    if display_info
        println("===============================================================")
        println("Convergence report")
        println("===============================================================")
        println("   m         abs RMSE           rel RMSE")
        println("---------------------------------------------------------------")
    end
    for m in ms
        res = builder(m)
        E_numeric = res.E
        psi_numeric = res.psi
        xint = res.x
        yint = hasproperty(res, :y) ? res.y : nothing


        result = res

        rep = energy_report(
            E_numeric,
            energy_analytical;
            nmax=nmax,
            display_info=false
        )
        analytics = [rep.E_analytical, rep.states]

        for n in 1:nmax
            push!(E_levels[n], rep.E_numeric[n])
            push!(abs_errors_levels[n], abs(rep.abs_error[n]))
            push!(rel_errors_levels[n], abs(rep.rel_error[n]))
        end

        push!(rmse_abs_vals, rep.rmse_abs)
        push!(rmse_rel_vals, rep.rmse_rel)
     
        if display_info
            @printf("%-6s    %12.6e    %12.6e\n", string(m), rep.rmse_abs, rep.rmse_rel)
        end
    end

    mids_rmse_abs, order_rmse_abs = observed_order(ms, rmse_abs_vals)
    mids_rmse_rel, order_rmse_rel = observed_order(ms, rmse_rel_vals)

    if display_info
        println("===============================================================")
        println()
        println("Observed order from absolute RMSE")
        println("---------------------------------------------------------------")
        println("  m_i -> m_{i+1}          p")
        println("---------------------------------------------------------------")
        for i in 1:length(order_rmse_abs)
            @printf("%-10s -> %-6s     %10.6f\n", string(ms[i]), string(ms[i+1]), order_rmse_abs[i])
        end
        println("---------------------------------------------------------------")
    
         println("===============================================================")
        println()
        println("Observed order from relative RMSE")
        println("---------------------------------------------------------------")
        println("  m_i -> m_{i+1}          p")
        println("---------------------------------------------------------------")
        for i in 1:length(order_rmse_rel)
            @printf("%-10s -> %-6s     %10.6f\n", string(ms[i]), string(ms[i+1]), order_rmse_rel[i])
        end
        println("---------------------------------------------------------------")

    end
    c2 = rmse_rel_vals[1] * mvals[1]^2
    order2_line = c2 ./ (mvals .^ 2)

    if display_plot
        CairoMakie.with_theme(theme_article()) do

            ### Convergence of energy levels ###
            fig_energy = CairoMakie.Figure(size = (650, 400), figurepadding = (10, 5, 15, 15))

            ax_energy = CairoMakie.Axis(fig_energy[1, 1], 
                xlabel = "N", 
                ylabel = "Energy (eV)", 
                title = "Grid convergence of numerical energy levels"
            )
          
            for n in 1:nmax
                CairoMakie.lines!(ax_energy, mvals, E_levels[n], label = "E$n numerical")
                CairoMakie.scatter!(ax_energy, mvals, E_levels[n])
                CairoMakie.hlines!(ax_energy, [analytics[1][n]], linestyle = :dash, label = "Level $(analytics[2][n]) analytical")
            end
            legend_elements = [
            CairoMakie.LineElement(color = :black, linestyle = :solid),
            CairoMakie.LineElement(color = :black, linestyle = :dash),
            ]
            CairoMakie.Legend(fig_energy[1, 2], legend_elements, ["Numerical", "Analytical"], framevisible = false)

            # spacing between axis and legend
            colgap!(fig_energy.layout, 10)
            
            ### ABS RMSE ###
            fig_rmse_abs = CairoMakie.Figure(size = (650, 400))
            ax_rmse_abs = CairoMakie.Axis(fig_rmse_abs[1, 1], xlabel = "N", ylabel = "Absolute RMSE (eV)", title = "Absolute energy convergence")
            CairoMakie.lines!(ax_rmse_abs, mvals, rmse_abs_vals, label = "abs RMSE")
            CairoMakie.scatter!(ax_rmse_abs, mvals, rmse_abs_vals)
            CairoMakie.axislegend(ax_rmse_abs, position = :rt)

            ### REL RMSE ###
            fig_rmse_rel = CairoMakie.Figure(size = (650, 400))
            ax_rmse_rel = CairoMakie.Axis(fig_rmse_rel[1, 1], 
                xlabel = "N", 
                ylabel = 
                "Relative RMSE", 
                title = "Relative energy convergence"
            )
            
            CairoMakie.lines!(ax_rmse_rel, mvals, rmse_rel_vals, label = "rel RMSE")
            CairoMakie.scatter!(ax_rmse_rel, mvals, rmse_rel_vals)
            leg = Legend(fig_rmse_rel[1, 2], ax_rmse_rel,
                framevisible = false,
                padding = (5, 5, 5, 5)
            )

            # spacing between axis and legend
            colgap!(fig_rmse_rel.layout, 10)

            
            ### Error per level ###
            fig_abs_levels = CairoMakie.Figure(size = (650, 400))
            ax_abs_levels = CairoMakie.Axis(
                fig_abs_levels[1, 1], 
                xlabel = "N",
                ylabel = L"|E_n - E_n^{{\mathrm{an}}}\,(\mathrm{eV})",
                ylabelsize = 20,
                title = "Absolute error per level")
            for n in 1:nmax
                CairoMakie.lines!(ax_abs_levels, mvals, abs_errors_levels[n], label = "Level n=$n")
                CairoMakie.scatter!(ax_abs_levels, mvals, abs_errors_levels[n])
            end
            CairoMakie.axislegend(ax_abs_levels, position = :rt)


            ### Relative error per level ###
            fig_rel_levels = CairoMakie.Figure(size = (650, 400))
            ax_rel_levels = CairoMakie.Axis(fig_rel_levels[1, 1], xlabel = "N", ylabel = "Relative error", title = "Relative error per level")
            for n in 1:nmax
                CairoMakie.lines!(ax_rel_levels, mvals, rel_errors_levels[n], label = "Level $(analytics[2][n])")
                CairoMakie.scatter!(ax_rel_levels, mvals, rel_errors_levels[n])
            end
            leg = Legend(fig_rel_levels[1, 2], ax_rel_levels,
                framevisible = false,
                padding = (5, 5, 5, 5)
            )

            colgap!(fig_rel_levels.layout, 10)

            ### Observed order ###
            fig_orders = CairoMakie.Figure(size = (1200, 450))
            ax_order_abs = CairoMakie.Axis(fig_orders[1, 1], 
                xlabel = "N", 
                ylabel = "Observed order p",
                title = "Observed absolute convergence order"
            )
            
            
            CairoMakie.lines!(ax_order_abs, mids_rmse_abs, order_rmse_abs, label = "p from abs RMSE")
            CairoMakie.scatter!(ax_order_abs, mids_rmse_abs, order_rmse_abs)
            CairoMakie.hlines!(ax_order_abs, [1.0], linestyle = :dash, label = "1st order")
            CairoMakie.hlines!(ax_order_abs, [2.0], linestyle = :dash, label = "2nd order")
            CairoMakie.axislegend(ax_order_abs, position = :rb)

            ax_order_rel = CairoMakie.Axis(fig_orders[1, 2], xlabel = "N", ylabel = "Observed order p", title = "Observed relative convergence order", yticks = [1.0, 2.0])
            CairoMakie.lines!(ax_order_rel, mids_rmse_rel, order_rmse_rel, label = "p from rel RMSE")
            CairoMakie.scatter!(ax_order_rel, mids_rmse_rel, order_rmse_rel)
            CairoMakie.hlines!(ax_order_rel, [1.0], linestyle = :dash, label = "1st order")
            CairoMakie.hlines!(ax_order_rel, [2.0], linestyle = :dash, label = "2nd order")
            CairoMakie.axislegend(ax_order_rel, position = :rb)

            fig_orders_rel = CairoMakie.Figure(size = (650, 400))
            ax_orders_rel = CairoMakie.Axis(fig_orders_rel[1, 1], xlabel = "N", ylabel = "Observed order p", title = "Observed relative convergence order")
            CairoMakie.lines!(ax_orders_rel, mids_rmse_rel, order_rmse_rel, label = "p from rel RMSE")
            CairoMakie.scatter!(ax_orders_rel, mids_rmse_rel, order_rmse_rel)
            CairoMakie.hlines!(ax_orders_rel, [1.0], linestyle = :dash, label = "1st order", color=:orange)
            CairoMakie.hlines!(ax_orders_rel, [2.0], linestyle = :dash, label = "2nd order")
            Legend(fig_orders_rel[1, 2], ax_orders_rel,
                framevisible = false
            )

            # --> HER
            all_y = order_rmse_rel
            ticks = sort(unique(vcat(0.0:0.2:2.5 |> collect, [1.0, 2.0])))
            ticks = filter(t -> t >= floor(minimum(all_y) - 0.1) && t <= ceil(maximum(all_y) + 0.1), ticks)
            ax_orders_rel.yticks = ticks
            CairoMakie.ylims!(ax_orders_rel, minimum(ticks) - 0.05, maximum(ticks)+0.05)



            ### Reference plot with O(m^-2) line ###
            fig_ref = CairoMakie.Figure(size = (650, 400))
            ax_ref = CairoMakie.Axis(
                fig_ref[1, 1],
                xlabel = "N",
                ylabel = "Error",
                title = "Convergence Behaviour",
                ytickformat = values -> [
                    iszero(y) ? rich("0") :
                    let
                        exp = floor(Int, log10(abs(y)))
                        mantissa = y / 10.0^exp
                        rich(@sprintf("%.1f × 10", mantissa), superscript(string(exp), fontsize=13))
                    end
                    for y in values
                ] 
            )
            l1 = CairoMakie.lines!(ax_ref, mvals, rmse_rel_vals, label = "Relative RMSE", linewidth = 2.2)
            CairoMakie.scatter!(ax_ref, mvals, rmse_rel_vals)
            l2 = CairoMakie.lines!(ax_ref, mvals, order2_line, linestyle = :dash, label = L"\mathcal{O}(h^{2})",)

            leg = Legend(fig_ref[1, 2], ax_ref,
                framevisible = false)
            
            


            display(fig_energy)
            display(fig_rmse_abs)
            display(fig_rmse_rel)
            display(fig_abs_levels)
            display(fig_rel_levels)
            display(fig_orders)
            display(fig_ref)
            display(fig_orders_rel)
            if save_plot
                save("$(type)_convergence_energy.pdf", fig_energy)
                save("$(type)_energy_conv_order.pdf", fig_ref)
                save("$(type)_relative_error_levels.pdf", fig_rel_levels)
                save("$(type)_rmse_rel.pdf", fig_rmse_rel)
                save("$(type)_orders_rel.pdf", fig_orders_rel)
            end
        end
    end

    return result
end


function wave_plot_1D(
    xint, 
    V,
    E_analytical, 
    E_numeric, 
    psi_numeric, 
    psi_analytical;
    nmax=min(length(E_numeric), size(psi_numeric, 2)),
    scaling = 5.0,
    name="",
    save_plot=false
    )

    dx = xint[2] - xint[1]
    # E_an = E_analytical.(1:nmax)

    CairoMakie.with_theme(theme_article()) do
        fig_wave = CairoMakie.Figure(size = (650, 400))
        ax_wave = CairoMakie.Axis(
            fig_wave[1, 1],
            xlabel = "x",
            ylabel = "Energy + psi(x)",
            title = "Wavefunctions and energy levels, ψ",
            limits = (nothing, (-0.5, maximum(E_numeric[1:nmax]) + 2))
        )
        CairoMakie.lines!(ax_wave, xint, V.(xint), color = :black, label = "V(x)")

        for n in 1:nmax
            psi_num = real(psi_numeric[:, n])
            psi_num = normalize_wavefunction(psi_num, dx)

            # psi_an = [psi_analytical(xi, n) for xi in xint]
            # psi_an = normalize_wavefunction(psi_an, dx)

            # if sum(psi_num .* psi_an) * dx < 0
            #     psi_num .*= -1
            # end

            CairoMakie.lines!(ax_wave, xint, E_numeric[n] .+ psi_num, label = "Numerical psi_$n")
            # CairoMakie.lines!(ax_wave, xint, E_an[n] .+ psi_an, linestyle = :dash, label = "Analytical psi_$n")
        end
        CairoMakie.axislegend(ax_wave, position = :rt)
        display(fig_wave)

        fig_prob = CairoMakie.Figure(size = (650, 400))
        ax_prob = CairoMakie.Axis(
            fig_prob[1, 1],
            xlabel = "x",
            ylabel = "Energy",
            title = rich("Wavefunctions and energy levels, |", 
                 rich("ψ", font = "Arial"), 
                 "|²"),
            limits = (nothing, (-minimum(E_numeric[1:nmax])*1.2, maximum(E_numeric[1:nmax])*1.2))
        )
        CairoMakie.lines!(ax_prob, xint, V.(xint), color = :black, label = "V(x)")

        for n in 1:nmax
            psi_num = real(psi_numeric[:, n])
            psi_num = normalize_wavefunction(psi_num, dx)

            # psi_an = [psi_analytical(xi, n) for xi in xint]
            # psi_an = normalize_wavefunction(psi_an, dx)

            println(sum(abs2, psi_num) * dx)

            CairoMakie.lines!(ax_prob, xint, E_numeric[n] .+ scaling * abs.(psi_num).^2, label = "Numerical psi_$n")
            # CairoMakie.lines!(ax_prob, xint, E_an[n] .+ scaling * abs.(psi_an).^2, linestyle = :dash, label = "Analytical psi_$n")
        end

        legend_elements = [
            CairoMakie.LineElement(color = :black, linestyle = :solid),
            # CairoMakie.LineElement(color = :black, linestyle = :dash),
        ]
        CairoMakie.Legend(fig_prob[1, 2], legend_elements, ["Numerical"], framevisible = false)
        
        display(fig_prob)
        if save_plot
            save("$(name)_wavefunctions.pdf", fig_wave)
            save("$(name)_probability.pdf", fig_prob)
        end
    end
end


end

