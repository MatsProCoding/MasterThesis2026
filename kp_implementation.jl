using Revise

includet("../src/project.jl")
includet("../src/math.jl")
includet("../src/solver_1D.jl")
includet("../src/effective_Masses.jl")
includet("../src/materials.jl")

using .Project.Constants
using .Project.Math
using .Project.Solver_1D
using .Project.Effective_Masses
using .Project.Materials

using Arpack
using CairoMakie
using LaTeXStrings
using LinearAlgebra
using SparseArrays
using Printf

plot_dir = "plots_1D_kp"
mkpath(plot_dir)


# ============================================================
# Helpers
# ============================================================

matval(mat, key) = mat isa NamedTuple ? getfield(mat, key) : mat[key]

function sort_eigenpairs(E, psi)
    perm = sortperm(real.(E))
    return real.(E[perm]), psi[:, perm]
end

arithmetic_mean(a, b) = 0.5 * (a + b)

function harmonic_mean(a, b)
    return 2 * a * b / (a + b)
end

function nearest_index(grid, x0)
    return argmin(abs.(grid .- x0))
end


# ============================================================
# Geometry
# ============================================================

function well_edges(Lx, Lwell)
    x_left = (Lx - Lwell) / 2
    x_right = (Lx + Lwell) / 2
    return x_left, x_right
end


# ============================================================
# Profiles on grid
# ============================================================

function make_piecewise_profile(
    grid,
    x_left,
    x_right,
    well_value,
    barrier_value;
    average=:arithmetic,
)
    values = similar(collect(grid), Float64)

    for i in eachindex(grid)
        x = grid[i]

        if x_left < x < x_right
            values[i] = well_value
        else
            values[i] = barrier_value
        end
    end

    interface_value = if average == :arithmetic
        arithmetic_mean(well_value, barrier_value)
    elseif average == :harmonic
        harmonic_mean(well_value, barrier_value)
    else
        error("Unknown average type: $average")
    end

    i_left = nearest_index(grid, x_left)
    i_right = nearest_index(grid, x_right)

    values[i_left] = interface_value
    values[i_right] = interface_value

    return values
end

function value_from_grid(x, grid, values)
    i = nearest_index(grid, x)
    return values[i]
end


# ============================================================
# Band alignment
# ============================================================

function band_edges_gap_split(well, barrier; Qc=0.65)
    Eg_well = matval(well, :Eg)
    Eg_barrier = matval(barrier, :Eg)

    gap_difference = Eg_barrier - Eg_well

    conduction_offset = Qc * gap_difference
    valence_offset = -(1 - Qc) * gap_difference

    Ec_well = 0.0
    Ev_well = -Eg_well

    Ec_barrier = Ec_well + conduction_offset
    Ev_barrier = Ev_well + valence_offset

    return (
        Ec_well=Ec_well,
        Ec_barrier=Ec_barrier,
        Ev_well=Ev_well,
        Ev_barrier=Ev_barrier,
        conduction_offset=conduction_offset,
        valence_offset=valence_offset,
        Eg_well=Eg_well,
        Eg_barrier=Eg_barrier,
        Qc=Qc,
    )
end


# ============================================================
# 2x2 diagnostics only
# ============================================================

function compute_kp_weights(E_full, psi_full, x, x_left, x_right)
    ngrid = length(x)
    nstates = length(E_full)

    conduction_weights = zeros(Float64, nstates)
    well_weights = zeros(Float64, nstates)

    inside = (x .> x_left) .& (x .< x_right)

    for i in 1:nstates
        psi_total = view(psi_full, :, i)
        psi_c = view(psi_full, 1:ngrid, i)

        conduction_weights[i] =
            sum(abs2, psi_c) / sum(abs2, psi_total)

        well_weights[i] =
            sum(abs2, psi_c[inside]) / sum(abs2, psi_c)
    end

    return (
        raw_energies=real.(E_full),
        conduction_weights=conduction_weights,
        well_weights=well_weights,
    )
end

# ============================================================
# Extraction of conduction bands
# ============================================================

function select_conduction_states(
    E_full,
    psi_full;
    threshold = 0.5,
)
    ngrid = Int(size(psi_full, 1) / 2)
    nstates = length(E_full)

    conduction_weights = zeros(Float64, nstates)

    for i in 1:nstates
        psi_total = view(psi_full, :, i)
        psi_c = view(psi_full, 1:ngrid, i)

        conduction_weights[i] =
            sum(abs2, psi_c) / sum(abs2, psi_total)
    end

    candidate_indices = [
        i for i in 1:nstates
        if conduction_weights[i] >= threshold
    ]

    candidate_indices = sort(candidate_indices, by = i -> real(E_full[i]))

    E_selected = real.(E_full[candidate_indices])

    psi_c_selected = psi_full[1:ngrid, candidate_indices]
    psi_v_selected = psi_full[ngrid+1:2ngrid, candidate_indices]

    diagnostics = (
        raw_energies = E_selected,
        selected_indices = candidate_indices,
        conduction_weights = conduction_weights[candidate_indices],
    )

    return E_selected, psi_c_selected, psi_v_selected, diagnostics
end

# ============================================================
# Main solver
# ============================================================

function perturbation_Hamiltonian(
    Lwell;
    Lx,
    well,
    barrier,
    system_name,
    m=1000,
    Nc=5,
    Qc=0.65,
)
    x0 = 0.0
    dk = 1e-6

    xint, xhalf, dx = make_grid(x0, Lx, m)
    ngrid = length(xint)

    x_left, x_right = well_edges(Lx, Lwell)

    edges = band_edges_gap_split(well, barrier; Qc=Qc)

    Ec_well = edges.Ec_well
    Ec_barrier = edges.Ec_barrier

    Ev_well = edges.Ev_well
    Ev_barrier = edges.Ev_barrier

    P_well = sqrt(Constants.hbar2_over_2me * matval(well, :Ep))
    P_barrier = sqrt(Constants.hbar2_over_2me * matval(barrier, :Ep))

    Ec_values = make_piecewise_profile(
        xint,
        x_left,
        x_right,
        Ec_well,
        Ec_barrier;
        average=:arithmetic,
    )

    Ev_values = make_piecewise_profile(
        xint,
        x_left,
        x_right,
        Ev_well,
        Ev_barrier;
        average=:arithmetic,
    )

    P_values = make_piecewise_profile(
        xint,
        x_left,
        x_right,
        P_well,
        P_barrier;
        average=:arithmetic,
    )

    Ec(x) = value_from_grid(x, xint, Ec_values)
    Ev(x) = value_from_grid(x, xint, Ev_values)
    P(x) = value_from_grid(x, xint, P_values)

    # --------------------------------------------------------
    # Bulk masses from local 2x2 Hamiltonians
    # --------------------------------------------------------

    function H_local_well(x, k)
        w11 = Ec_well + Constants.hbar2_over_2me * k^2
        w22 = Ev_well - Constants.hbar2_over_2me * k^2
        w12 = k * P_well
        w21 = conj(w12)

        return Hermitian([
            w11 w12
            w21 w22
        ])
    end

    function H_local_barrier(x, k)
        w11 = Ec_barrier + Constants.hbar2_over_2me * k^2
        w22 = Ev_barrier - Constants.hbar2_over_2me * k^2
        w12 = k * P_barrier
        w21 = conj(w12)

        return Hermitian([
            w11 w12
            w21 w22
        ])
    end

    mass_v_well_vec, mass_c_well_vec =
        effective_mass([0.0], dk, H_local_well)

    mass_v_barrier_vec, mass_c_barrier_vec =
        effective_mass([0.0], dk, H_local_barrier)

    mass_c_well = mass_c_well_vec[1]
    mass_c_barrier = mass_c_barrier_vec[1]

    mass_v_well = mass_v_well_vec[1]
    mass_v_barrier = mass_v_barrier_vec[1]

    mass_c = make_piecewise_profile(
        xhalf,
        x_left,
        x_right,
        mass_c_well,
        mass_c_barrier;
        average=:harmonic,
    )

    mass_v = make_piecewise_profile(
        xhalf,
        x_left,
        x_right,
        mass_v_well,
        mass_v_barrier;
        average=:harmonic,
    )

    E_all = []
    Psi_all = []

    # ========================================================
    # Case 1: unperturbed
    # ========================================================

    Hc_unp = build_H1D(x0, Lx; mass=1.0, V=Ec, m=m)
    Hv_unp = build_H1D(x0, Lx; mass=-1.0, V=Ev, m=m)

    Ec_unp, psi_c_unp = eigs(
        Hermitian(Hc_unp),
        nev=Nc,
        sigma=0.0,
        which=:LM,
        maxiter=300_000,
    )

    Ec_unp, psi_c_unp = sort_eigenpairs(Ec_unp, psi_c_unp)

    push!(E_all, Ec_unp)
    push!(Psi_all, psi_c_unp)

    # ========================================================
    # Case 2: effective mass
    # ========================================================

    Hc_eff = build_H1D(x0, Lx; mass=mass_c, V=Ec, m=m)

    Ec_eff, psi_c_eff = eigs(
        Hermitian(Hc_eff),
        nev=Nc,
        sigma=0.0,
        which=:LM,
        maxiter=300_000,
    )

    Ec_eff, psi_c_eff = sort_eigenpairs(Ec_eff, psi_c_eff)

    push!(E_all, Ec_eff)
    push!(Psi_all, psi_c_eff)

    # ========================================================
    # Case 3: full 2x2 kp
    # ========================================================

    Hc_full = build_H1D(x0, Lx; mass=1.0, V=Ec, m=m)
    Hv_full = build_H1D(x0, Lx; mass=-1.0, V=Ev, m=m)

    Pdiag = spdiagm(0 => P_values)
    D = build_SD(m, dx)

    w11 = sparse(Hc_full)
    w22 = sparse(Hv_full)

    w12 = -im * (Pdiag*D + D* Pdiag)/2
    w21 = w12'

    H_full = sparse([
        w11 w12
        w21 w22
    ])

    E_full, psi_full = eigs(
        Hermitian(H_full),
        nev=Nc,
        sigma=Ec_eff[1],
        which=:LM,
        maxiter=300_000,
    )

    E_full_sorted, psi_full_sorted = sort_eigenpairs(E_full, psi_full)

    E_full_selected, psi_c_full_selected, psi_v_full_selected, kp_diagnostics =
        select_conduction_states(
            E_full_sorted,
            psi_full_sorted;
            threshold = 0.5,
        )

    push!(E_all, E_full_selected)
    push!(Psi_all, psi_c_full_selected)
    println("Material: $system_name, Lwell = $Lwell Å")
    return (
        Lwell=Lwell,
        Lx=Lx,
        x=collect(xint),
        xhalf=collect(xhalf),
        dx=dx,
        E_all=E_all,
        Psi_all=Psi_all,
        Psi_c_full=psi_c_full_selected,
        Psi_v_full=psi_v_full_selected,
        Ec=Ec,
        Ev=Ev,
        P=P,
        Ec_values=Ec_values,
        Ev_values=Ev_values,
        P_values=P_values,
        mass_c=mass_c,
        mass_v=mass_v,
        system_name=system_name,
        band_edges=edges,
        x_left=x_left,
        x_right=x_right,
        kp_diagnostics=kp_diagnostics,
    )
end


# ============================================================
# Sweep
# ============================================================

function run_sweep(
    Lwells;
    Lx,
    well,
    barrier,
    system_name,
    N=200,
    m=1000,
    Qc=0.65,
)
    return [
        perturbation_Hamiltonian(
            Lwell;
            Lx=Lx,
            well=well,
            barrier=barrier,
            system_name=system_name,
            m=m,
            Nc=N,
            Qc=Qc,
        )
        for Lwell in Lwells
    ]
end



# ============================================================
# Materials
# ============================================================

xAl = 0.3

well_GaAs = Materials.VUR["GaAs"]
barrier_AlGaAs = Materials.algaas_params(xAl)

well_InAs = Materials.VUR["InAs"]
barrier_AlAs = Materials.VUR["AlAs"]


# ============================================================
# Run
# ============================================================

Lx = 800.0
Lwells = [20.0, 40.0, 60.0, 80.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0, 400.0]

N = 8
m = 1000

Qc_GaAs_AlGaAs = 0.65
Qc_InAs_AlAs = 0.8

results_GaAs_AlGaAs = run_sweep(
    Lwells;
    Lx=Lx,
    well=well_GaAs,
    barrier=barrier_AlGaAs,
    system_name="GaAs / Al$(xAl)Ga$(1 - xAl)As",
    N=N,
    m=m,
    Qc=Qc_GaAs_AlGaAs,
)

results_InAs_AlAs = run_sweep(
    Lwells;
    Lx=Lx,
    well=well_InAs,
    barrier=barrier_AlAs,
    system_name="InAs / AlAs",
    N=N,
    m=m,
    Qc=Qc_InAs_AlAs,
)


# ============================================================
# Plots
# ============================================================

function theme_article()
    merge(
        theme_latexfonts(),
        Theme(
            fontsize = 16,
            Axis = (
                xlabelsize = 20,
                ylabelsize = 20,
                titlesize = 22,
                xticklabelsize = 16,
                yticklabelsize = 16,
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
                labelsize = 18,
                framevisible = false,
            ),
            Lines = (linewidth = 2.2,),
            Scatter = (markersize = 8,)
        )
    )
end


# ============================================================
# Diagnostics helpers
# ============================================================

function get_kp_diagnostics(results; state::Int=1)
    Lwell_values = Float64[]
    energy_values = Float64[]
    energy_ratios = Float64[]
    conduction_weight_values = Float64[]

    for res in results
        diag = res.kp_diagnostics

        if state > length(diag.raw_energies)
            push!(Lwell_values, res.Lwell)
            push!(energy_values, NaN)
            push!(energy_ratios, NaN)
            push!(conduction_weight_values, NaN)
            continue
        end

        E = diag.raw_energies[state]

        push!(Lwell_values, res.Lwell)
        push!(energy_values, E)
        push!(energy_ratios, E / res.band_edges.Ec_barrier)
        push!(conduction_weight_values, diag.conduction_weights[state])
    end

    return (
        Lwell=Lwell_values,
        E=energy_values,
        E_over_barrier=energy_ratios,
        conduction_weight=conduction_weight_values,
    )
end


function print_kp_weight_table(results; iL::Int=1, max_rows::Int=10)
    res = results[iL]
    diag = res.kp_diagnostics

    nrows = min(max_rows, length(diag.raw_energies))

    println()
    println("============================================================")
    println("2x2 kp diagnostics")
    println("System: ", res.system_name)
    println("Lwell = ", res.Lwell, " A")
    println("Lx    = ", res.Lx, " A")
    println("Ec barrier = ", res.band_edges.Ec_barrier, " eV")
    println("============================================================")
    println("state | E [eV]       | E/Ebarrier | cond weight")
    println("------|--------------|------------|------------")

    for state in 1:nrows
        E = diag.raw_energies[state]
        ratio = E / res.band_edges.Ec_barrier

        @printf(
            "%5d | %12.6e | %10.6f | %11.6f\n",
            state,
            E,
            ratio,
            diag.conduction_weights[state],
        )
    end
end


function print_kp_state_summary(results; state::Int=1)
    println()
    println("============================================================")
    println("Full 2x2 kp state summary")
    println("System: ", results[1].system_name)
    println("State = ", state)
    println("============================================================")
    println("Lwell [A] | E [eV]       | E/Ebarrier | cond weight")
    println("----------|--------------|------------|------------")

    for res in results
        diag = res.kp_diagnostics

        if state > length(diag.raw_energies)
            @printf(
                "%9.2f | %12s | %10s | %11s\n",
                res.Lwell,
                "missing",
                "missing",
                "missing",
            )
            continue
        end

        E = diag.raw_energies[state]
        ratio = E / res.band_edges.Ec_barrier

        @printf(
            "%9.2f | %12.6e | %10.6f | %11.6f\n",
            res.Lwell,
            E,
            ratio,
            diag.conduction_weights[state],
        )
    end
end

# ============================================================
# Plotting with CairoMakie
# ============================================================

case_labels_E = [
    L"E_{\mathrm{uncoupled}}",
    L"E_{\mathrm{EMA}}",
    L"E_{\mathrm{coupled}}",
]


case_labels_F = [
    L"F_{\mathrm{uncoupled}}",
    L"F_{\mathrm{EMA}}",
    L"F_{\mathrm{coupled}}",
]
uncoupled_color = Makie.RGBf(0.00, 0.20, 0.80)   # clear blue
ema_color       = Makie.RGBf(0.00, 0.65, 0.20)   # clear green
coupled_color   = Makie.RGBf(0.95, 0.35, 0.00)   # clear orange-red

total_color       = coupled_color
component_c_color = Makie.RGBf(0.00, 0.32, 0.75)  # deep blue
component_v_color = Makie.RGBf(0.45, 0.15, 0.65)  # deep purple
interface_color   = Makie.RGBf(0.10, 0.10, 0.10)

function energy_for_state(results, icase, state)
    return [
        state <= length(res.E_all[icase]) ? res.E_all[icase][state] : NaN
        for res in results
    ]
end


function plot_energy_states_3cols_single_material(
    Lwells;
    results,
    material_label,
    states = 1:3,
    savepath = nothing,
)
    with_theme(theme_article()) do
        fig = Figure(size = (1450, 600), figure_padding = (20, 10, 10, 5))

        # Global title for all three subplots
        Label(
            fig[1, 1:4],
            material_label,
            fontsize = 30,
            font = :italic,
            tellwidth = true,
        )

        styles = [:solid, :solid, :dash]
        axes = Axis[]

        for (j, state) in enumerate(states)
            ax = Axis(
                fig[2, j],
                xlabel = L"L_{\mathrm{well}} \ [\mathrm{\AA}]",
                ylabel = L"E \ [\mathrm{eV}]",
                title = "State $state",
                xticklabelspace = 20.0,
                yticklabelspace = 45.0,
                ytickformat = "{:.2f}",
                xlabelsize = 26,
                ylabelsize = 26,
                titlesize = 26,
                xticklabelsize = 23,
                yticklabelsize = 23,
            )

            push!(axes, ax)

            for icase in 1:3
                E = energy_for_state(results, icase, state)

                lines!(
                    ax,
                    Lwells,
                    E,
                    linestyle = styles[icase],
                    label = case_labels_E[icase],
                )

                CairoMakie.scatter!(
                    ax,
                    Lwells,
                    E,
                )
            end

            Ec_barrier = results[1].band_edges.Ec_barrier

            lines!(
                ax,
                [minimum(Lwells), maximum(Lwells)],
                [Ec_barrier, Ec_barrier],
                linestyle = :dash,
                color = :black,
                label = j == 1 ? L"E_{c,\mathrm{barrier}}" : nothing,
            )
        end

        Legend(fig[2, 4], axes[1], valign = :center, labelsize = 28)

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end


function plot_selected_state_article(
    Lwells;
    results,
    material_label,
    iL::Int = 1,
    state::Int = 1,
    probability_density::Bool = true,
    zoom_factor::Float64 = 3.0,
    savepath = nothing,
)
    with_theme(theme_article()) do
        res = results[iL]

        x = res.x
        dx = res.dx

        Lwell = res.Lwell
        x_center = res.Lx / 2

        x_window = zoom_factor * Lwell
        x_min = max(minimum(x), x_center - x_window / 2)
        x_max = min(maximum(x), x_center + x_window / 2)

        fig = Figure(size = (700, 420), figure_padding = (40, 10, 15, 15))

        ax = Axis(
            fig[1, 1],
            xlabel = L"x\, [\mathrm{\AA}]",
            ylabel = probability_density ? L"|F(x)|^2" : L"F(x)",
            title = L"%$(material_label) with L_{\mathrm{well}}=%$(round(res.Lwell, digits=1)) \mathrm{\AA} in \mathrm{state}\ %$state",
            yticklabelspace = 45.0,
            xlabelsize = 26,
            ylabelsize = 26,
            titlesize = 26,
            xticklabelsize = 23,
            yticklabelsize = 23,
        )

        CairoMakie.xlims!(ax, x_min, x_max)

        y_all = Vector{Vector{Float64}}()
        line_styles = Symbol[]
        labels = Any[]

        case_labels_text = [
            "Uncoupled",
            "EMA",
            "Coupled",
        ]

        for icase in 1:3
            if icase == 3
                if state > size(res.Psi_c_full, 2)
                    @warn "Coupled case does not have state = $state. Skipping."
                    continue
                end

                Fc = res.Psi_c_full[:, state]
                Fv = res.Psi_v_full[:, state]

                # Normalize the full two-component spinor:
                # ∫ (|Fc|² + |Fv|²) dx = 1
                norm_total = sqrt(sum(abs2.(Fc) .+ abs2.(Fv)) * dx)

                Fc = Fc ./ norm_total
                Fv = Fv ./ norm_total

                if probability_density
                    y = abs2.(Fc) .+ abs2.(Fv)
                else
                    # There is no unique scalar "total wavefunction" for a two-component spinor.
                    # For wavefunction mode, plot the conduction component.
                    y = real.(Fc)
                end
            else
                psi_mat = res.Psi_all[icase]

                if state > size(psi_mat, 2)
                    @warn "Case $icase does not have state = $state. Skipping."
                    continue
                end

                F = psi_mat[:, state]
                F = normalize_wavefunction(F, dx)

                y = probability_density ? abs2.(F) : real.(F)
            end

            push!(y_all, collect(real.(y)))
            push!(line_styles, icase == 3 ? :dash : :solid)
            push!(labels, case_labels_text[icase])
        end

        for j in eachindex(y_all)
            lines!(
                ax,
                x,
                y_all[j],
                linestyle = line_styles[j],
                label = labels[j],
            )
        end

        if isempty(y_all)
            potential_height = 1.0
        elseif probability_density
            potential_height = maximum([maximum(y) for y in y_all])
        else
            potential_height = maximum([maximum(abs.(y)) for y in y_all])
        end

        Ec_barrier = res.band_edges.Ec_barrier

        Ec_scaled =
            Ec_barrier != 0 ?
            potential_height .* res.Ec_values ./ Ec_barrier :
            res.Ec_values

        lines!(
            ax,
            x,
            Ec_scaled,
            color = :black,
            linestyle = :dot,
            linewidth = 2.2,
            label = L"E_c(x)\ \mathrm{scaled}",
        )

        Legend(fig[1, 2], ax, valign = :center, labelsize = 20)

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end



function plot_coupled_valence_component_article(
    results;
    material_label,
    iL::Int = 1,
    state::Int = 1,
    probability_density::Bool = true,
    zoom_factor::Float64 = 3.0,
    savepath = nothing,
)
    with_theme(theme_article()) do
        res = results[iL]

        x = res.x
        dx = res.dx
        Lwell = res.Lwell
        x_center = res.Lx / 2

        x_window = zoom_factor * Lwell
        x_min = max(minimum(x), x_center - x_window / 2)
        x_max = min(maximum(x), x_center + x_window / 2)

        # Full coupled 2x2 kp components
        Fc = res.Psi_c_full[:, state]
        Fv = res.Psi_v_full[:, state]

        # Correct spinor normalization:
        # ∫ (|Fc|² + |Fv|²) dx = 1
        norm_total = sqrt(sum(abs2.(Fc) .+ abs2.(Fv)) * dx)

        Fc = Fc ./ norm_total
        Fv = Fv ./ norm_total

        yc = probability_density ? abs2.(Fc) : real.(Fc)
        yv = probability_density ? abs2.(Fv) : real.(Fv)

        fig = Figure(size = (720, 420), figure_padding = (40, 10, 15, 15))

        ax = Axis(
            fig[1, 1],
            xlabel = L"x\, [\mathrm{\AA}]",
            ylabel = probability_density ? L"|F(x)|^2" : L"F(x)",
            title = L"%$(material_label) with L_{\mathrm{well}}=%$(round(Lwell, digits=1))\ \mathrm{\AA} in state\ %$state",
            yticklabelspace = 45.0,
            xlabelsize = 26,
            ylabelsize = 26,
            titlesize = 24,
            xticklabelsize = 23,
            yticklabelsize = 23,
        )

        CairoMakie.xlims!(ax, x_min, x_max)

        lines!(
            ax,
            x,
            real.(yc),
            linestyle = :solid,
            label = probability_density ? L"|F_c(x)|^2" : L"F_c(x)",
        )

        lines!(
            ax,
            x,
            real.(yv),
            linestyle = :solid,
            label = probability_density ? L"|F_v(x)|^2" : L"F_v(x)",
        )

        # Vertical dashed lines showing the heterointerfaces
        y_max = maximum([
            maximum(abs.(real.(yc))),
            maximum(abs.(real.(yv))),
        ])

        vlines!(
            ax,
            [res.x_left, res.x_right],
            color = :black,
            linestyle = :dash,
            linewidth = 1.8,
            label = L"\mathrm{Interface}",
        )

        CairoMakie.ylims!(ax, nothing, 1.08 * y_max)

        Legend(fig[1, 2], ax, valign = :center, labelsize = 20)

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end

function plot_coupled_total_density_article(
    results;
    material_label,
    iL::Int = 1,
    state::Int = 1,
    zoom_factor::Float64 = 3.0,
    savepath = nothing,
)
    with_theme(theme_article()) do
        res = results[iL]

        x = res.x
        dx = res.dx
        Lwell = res.Lwell
        x_center = res.Lx / 2

        x_window = zoom_factor * Lwell
        x_min = max(minimum(x), x_center - x_window / 2)
        x_max = min(maximum(x), x_center + x_window / 2)

        # Full coupled 2x2 kp components
        Fc = res.Psi_c_full[:, state]
        Fv = res.Psi_v_full[:, state]

        # Correct spinor normalization:
        # ∫ (|Fc|² + |Fv|²) dx = 1
        norm_total = sqrt(sum(abs2.(Fc) .+ abs2.(Fv)) * dx)

        Fc = Fc ./ norm_total
        Fv = Fv ./ norm_total

        rho_c = abs2.(Fc)
        rho_v = abs2.(Fv)
        rho_total = rho_c .+ rho_v

        fig = Figure(size = (720, 420), figure_padding = (40, 10, 15, 15))

        ax = Axis(
            fig[1, 1],
            xlabel = L"x\, [\mathrm{\AA}]",
            ylabel = L"|F(x)|^2",
            title = L"%$(material_label) with L_{\mathrm{well}}=%$(round(Lwell, digits=1))\ \mathrm{\AA} in state\ %$state",
            yticklabelspace = 65.0,
            xlabelsize = 26,
            ylabelsize = 26,
            titlesize = 24,
            xticklabelsize = 23,
            yticklabelsize = 23,
        )

        CairoMakie.xlims!(ax, x_min, x_max)

        lines!(
            ax,
            x,
            real.(rho_total),
            linestyle = :solid,
            linewidth = 2.8,
            label = L"|F(x)|^2",
        )

        lines!(
            ax,
            x,
            real.(rho_c),
            linestyle = :dash,
            linewidth = 2.0,
            label = L"|F_c(x)|^2",
        )

        lines!(
            ax,
            x,
            real.(rho_v),
            linestyle = :dash,
            linewidth = 2.0,
            label = L"|F_v(x)|^2",
        )

        # Vertical dashed lines showing the heterointerfaces
        vlines!(
            ax,
            [res.x_left, res.x_right],
            color = :black,
            linestyle = :dash,
            linewidth = 1.8,
            label = L"\mathrm{interface}",
        )

        CairoMakie.ylims!(ax, nothing, 1.08 * maximum(real.(rho_total)))

        Legend(fig[1, 2], ax, valign = :center, labelsize = 20)

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end


function plot_energy_vs_Lwell_single_state_article(
    Lwells;
    results,
    material_label,
    state::Int = 1,
    savepath = nothing,
)
    with_theme(theme_article()) do
        fig = Figure(size = (650, 420), figure_padding = (40, 15, 15, 15))

        ax = Axis(
            fig[1, 1],
            xlabel = L"L_{\mathrm{well}}\, \ [\mathrm{\AA}]",
            ylabel = L"E\, \ [\mathrm{eV}]",
            title = L"%$(material_label) in \mathrm{state}\ %$state",
            xticklabelspace = 10.0,
            yticklabelspace = 35.0,
        )

        styles = [:solid, :solid, :dash]

        for icase in 1:3
            E = energy_for_state(results, icase, state)

            lines!(
                ax,
                Lwells,
                E,
                linestyle = styles[icase],
                label = case_labels_E[icase],
            )

            CairoMakie.scatter!(
                ax,
                Lwells,
                E,
            )
        end

        Ec_barrier = results[1].band_edges.Ec_barrier

        lines!(
            ax,
            [minimum(Lwells), maximum(Lwells)],
            [Ec_barrier, Ec_barrier],
            linestyle = :dash,
            color = :black,
            label = L"E_{c,\mathrm{barrier}}",
        )

        Legend(fig[1, 2], ax, valign = :center, size = 24)

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end

function plot_conduction_weight_single_material_article(
    Lwells;
    results,
    material_label,
    state::Int = 1,
    savepath = nothing,
)
    with_theme(theme_article()) do
        fig = Figure(size = (650, 420), figure_padding = (40, 15, 15, 15))

        data = get_kp_diagnostics(results; state = state)

        ax = Axis(
            fig[1, 1],
            xlabel = L"L_{\mathrm{well}}\, [\mathrm{\AA}]",
            ylabel = L"w_c",
            title = L"%$(material_label), \mathrm{state}\ %$state",
            yticklabelspace = 45.0,
            yticks = 0.0:0.25:1.0,
            ytickformat = "{:.2f}",
            xlabelsize = 26,
            ylabelsize = 26,
            titlesize = 26,
            xticklabelsize = 23,
            yticklabelsize = 23,
        )

        CairoMakie.ylims!(ax, 0.0, 1.05)

        lines!(
            ax,
            data.Lwell,
            data.conduction_weight,
            linestyle = :solid,
            label = L"w_c",
        )

        CairoMakie.scatter!(
            ax,
            data.Lwell,
            data.conduction_weight,
        )

        Legend(fig[1, 2], ax, valign = :center, labelsize = 24)

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end

function plot_three_lengths_unc_ema_coupled_article(
    results;
    material_label,
    iLs = [2, 5, 8],
    state::Int = 1,
    zoom_factor::Float64 = 3.0,
    savepath = nothing,
)
    with_theme(theme_article()) do
        fig = Figure(size = (600, 1200), figure_padding = (10, 20, 20, 10))

        axes = Axis[]

        for (row, iL) in enumerate(iLs)
            res = results[iL]

            x = res.x
            dx = res.dx
            Lwell = res.Lwell
            x_center = res.Lx / 2

            x_window = zoom_factor * Lwell
            x_min = max(minimum(x), x_center - x_window / 2)
            x_max = min(maximum(x), x_center + x_window / 2)

            ax = Axis(
                fig[row, 2],
                xlabel = row == length(iLs) ? L"x\, [\mathrm{\AA}]" : "",
                ylabel = L"|F(x)|^2",
                title = L"L_{\mathrm{well}}=%$(round(Lwell, digits=1))\ \mathrm{\AA}",
                yticklabelspace = 55.0,
                xlabelsize = 24,
                ylabelsize = 24,
                titlesize = 24,
                xticklabelsize = 21,
                yticklabelsize = 21,
            )

            push!(axes, ax)

            CairoMakie.xlims!(ax, x_min, x_max)

            y_all = Vector{Vector{Float64}}()

            # Uncoupled
            F_unc = res.Psi_all[1][:, state]
            F_unc = normalize_wavefunction(F_unc, dx)
            y_unc = abs2.(F_unc)
            push!(y_all, collect(real.(y_unc)))

            lines!(
                ax,
                x,
                real.(y_unc),
                color = uncoupled_color,
                linestyle = :solid,
                linewidth = 2.4,
            )

            # Coupled total density
            # Plotted before EMA, so EMA lies visually in front.
            Fc = res.Psi_c_full[:, state]
            Fv = res.Psi_v_full[:, state]

            norm_total = sqrt(sum(abs2.(Fc) .+ abs2.(Fv)) * dx)

            Fc = Fc ./ norm_total
            Fv = Fv ./ norm_total

            y_coupled = abs2.(Fc) .+ abs2.(Fv)
            push!(y_all, collect(real.(y_coupled)))

            lines!(
                ax,
                x,
                real.(y_coupled),
                color = coupled_color,
                linestyle = :solid,
                linewidth = 2.6,
            )

            # EMA
            # Plotted last, so it lies in front of coupled.
            F_ema = res.Psi_all[2][:, state]
            F_ema = normalize_wavefunction(F_ema, dx)
            y_ema = abs2.(F_ema)
            push!(y_all, collect(real.(y_ema)))

            lines!(
                ax,
                x,
                real.(y_ema),
                color = ema_color,
                linestyle = :dash,
                linewidth = 2.8,
            )

            # Interfaces
            vlines!(
                ax,
                [res.x_left, res.x_right],
                color = interface_color,
                linestyle = :dash,
                linewidth = 1.5,
            )

            y_max = maximum([maximum(y) for y in y_all])
            CairoMakie.ylims!(ax, nothing, 1.08 * y_max)
        end

        Legend(
            fig[1:3, 1],
            [
                LineElement(color = uncoupled_color, linestyle = :solid, linewidth = 2.4),
                LineElement(color = ema_color, linestyle = :dash, linewidth = 2.8),
                LineElement(color = coupled_color, linestyle = :solid, linewidth = 2.6),
                LineElement(color = interface_color, linestyle = :dash, linewidth = 1.5),
            ],
            [
                "Uncoupled",
                "EMA",
                "Coupled",
                L"\mathrm{interface}",
            ],
            valign = :center,
            labelsize = 22,
        )

        colsize!(fig.layout, 1, Relative(0.22))
        colsize!(fig.layout, 2, Relative(0.78))
        colgap!(fig.layout, 20)
        rowgap!(fig.layout, 15)

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end
function plot_three_lengths_coupled_components_article(
    results;
    material_label,
    iLs = [2, 5, 8],
    state::Int = 1,
    zoom_factor::Float64 = 3.0,
    savepath = nothing,
)
    with_theme(theme_article()) do
        fig = Figure(size = (600, 1200), figure_padding = (20, 10, 20, 10))

        axes = Axis[]

        for (row, iL) in enumerate(iLs)
            res = results[iL]

            x = res.x
            dx = res.dx
            Lwell = res.Lwell
            x_center = res.Lx / 2

            x_window = zoom_factor * Lwell
            x_min = max(minimum(x), x_center - x_window / 2)
            x_max = min(maximum(x), x_center + x_window / 2)

            Fc = res.Psi_c_full[:, state]
            Fv = res.Psi_v_full[:, state]

            norm_total = sqrt(sum(abs2.(Fc) .+ abs2.(Fv)) * dx)

            Fc = Fc ./ norm_total
            Fv = Fv ./ norm_total

            rho_c = abs2.(Fc)
            rho_v = abs2.(Fv)
            rho_total = rho_c .+ rho_v

            ax = Axis(
                fig[row, 1],
                xlabel = row == length(iLs) ? L"x\, [\mathrm{\AA}]" : "",
                ylabel = L"|F(x)|^2",
                title = L"L_{\mathrm{well}}=%$(round(Lwell, digits=1))\ \mathrm{\AA}",
                yticklabelspace = 55.0,
                xlabelsize = 24,
                ylabelsize = 24,
                titlesize = 24,
                xticklabelsize = 21,
                yticklabelsize = 21,
            )

            push!(axes, ax)

            CairoMakie.xlims!(ax, x_min, x_max)

            lines!(
                ax,
                x,
                real.(rho_total),
                color = total_color,
                linestyle = :solid,
                label = L"|F(x)|^2",
            )

            lines!(
                ax,
                x,
                real.(rho_c),
                color = component_c_color,
                linestyle = :dash,
                label = L"|F_c(x)|^2",
            )

            lines!(
                ax,
                x,
                real.(rho_v),
                color = component_v_color,
                linestyle = :dash,
                label = L"|F_v(x)|^2",
            )

            vlines!(
                ax,
                [res.x_left, res.x_right],
                color = :black,
                linestyle = :dash,
                linewidth = 1.5,
                label = row == 1 ? L"\mathrm{interface}" : nothing,
            )

            CairoMakie.ylims!(ax, nothing, 1.08 * maximum(real.(rho_total)))
        end

        Legend(fig[1:3, 2], axes[1], valign = :center, labelsize = 22)

        colsize!(fig.layout, 1, Relative(0.78))
        colsize!(fig.layout, 2, Relative(0.22))
        colgap!(fig.layout, 20)
        rowgap!(fig.layout, 15)

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end

function print_confinement_weight_table(
    results;
    iLs = [11, 5, 3],
    state::Int = 2,
)
    println()
    println("============================================================")
    println("Confinement weight table")
    println("System: ", results[1].system_name)
    println("State = ", state)
    println("============================================================")
    println("index | Lwell [Å] | EMA       | coupled")
    println("------|-----------|-----------|-----------")

    for iL in iLs
        res = results[iL]

        x = res.x
        dx = res.dx
        inside = (x .> res.x_left) .& (x .< res.x_right)

        # Case 2: EMA
        F_ema = res.Psi_all[2][:, state]
        rho_ema = abs2.(F_ema)
        w_ema = sum(rho_ema[inside]) * dx / (sum(rho_ema) * dx)

        # Case 3: coupled, total spinor density
        Fc = res.Psi_c_full[:, state]
        Fv = res.Psi_v_full[:, state]
        rho_coupled = abs2.(Fc) .+ abs2.(Fv)
        w_coupled = sum(rho_coupled[inside]) * dx / (sum(rho_coupled) * dx)

        @printf(
            "%5d | %9.1f | %9.6f | %9.6f\n",
            iL,
            res.Lwell,
            w_ema,
            w_coupled,
        )
    end
end


# ============================================================
# Display  plots
# ============================================================
display(
    plot_energy_states_3cols_single_material(
        Lwells;
        results = results_GaAs_AlGaAs,
        material_label = L"\mathrm{GaAs}/\mathrm{Al}_{0.3}\mathrm{Ga}_{0.7}\mathrm{As}",
        states = 1:3,
        savepath = joinpath(plot_dir, "energy_states_GaAs_AlGaAs_3colsf.pdf"),
    )
)

display(
    plot_energy_states_3cols_single_material(
        Lwells;
        results = results_InAs_AlAs,
        material_label = L"\mathrm{InAs} / \mathrm{AlAs}",
        states = 1:3,
        savepath = joinpath(plot_dir, "energy_states_InAs_AlAs_3colsf.pdf"),
    )
)

display(
    plot_selected_state_article(
        Lwells;
        results = results_GaAs_AlGaAs,
        material_label = L"\mathrm{GaAs}/\mathrm{Al}_{0.3}\mathrm{Ga}_{0.7}\mathrm{As}",
        iL = 11,
        state = 7,
        probability_density = true,
        zoom_factor = 100.0,
        savepath = joinpath(plot_dir, "wavefunction_GaAs_state2f.pdf"),
    )
)

display(
    plot_selected_state_article(
        Lwells;
        results = results_InAs_AlAs,
        material_label = L"\mathrm{InAs}/\mathrm{AlAs}",
        iL = 2,
        state = 2,
        probability_density = true,
        zoom_factor = 3.0,
        savepath = joinpath(plot_dir, "wavefunction_InAs_dev2f.pdf"),
    )
)


display(
    plot_coupled_valence_component_article(
        results_GaAs_AlGaAs;
        material_label = L"\mathrm{GaAs}/\mathrm{Al}_{0.3}\mathrm{Ga}_{0.7}\mathrm{As}",
        iL = 9,
        state = 2,
        probability_density = false,
        zoom_factor = 3.0,
        savepath = joinpath(plot_dir, "coupled_components_GaAs_state1f.pdf"),
    )
)

display(
    plot_coupled_valence_component_article(
        results_InAs_AlAs;
        material_label = L"\mathrm{InAs}/\mathrm{AlAs}",
        iL = 2,
        state = 2,
        probability_density = false,
        zoom_factor = 3.0,
        savepath = joinpath(plot_dir, "coupled_components_InAs_state1f_real.pdf"),
    )
)

display(
    plot_coupled_total_density_article(
        results_InAs_AlAs;
        material_label = L"\mathrm{InAs}/\mathrm{AlAs}",
        iL = 12,
        state = 2,
        zoom_factor = 3.0,
        savepath = joinpath(plot_dir, "coupled_total_density_InAs_state1f.pdf"),
    )
)

display(
    plot_coupled_total_density_article(
        results_GaAs_AlGaAs;
        material_label = L"\mathrm{GaAs}/\mathrm{Al}_{0.3}\mathrm{Ga}_{0.7}\mathrm{As}",
        iL = 3,
        state = 2,
        zoom_factor = 2.0,
        savepath = joinpath(plot_dir, "coupled_total_density_GaAs_state1f.pdf"),
    )
)


selected_iLs = [11, 5, 4]  

display(
    plot_three_lengths_unc_ema_coupled_article(
        results_GaAs_AlGaAs;
        material_label = L"\mathrm{GaAs}/\mathrm{Al}_{0.3}\mathrm{Ga}_{0.7}\mathrm{As}",
        iLs = selected_iLs,
        state = 2,
        zoom_factor = 1.5,
        savepath = joinpath(plot_dir, "three_lengths_GaAs_unc_ema_coupled.pdf"),
    )
)

display(
    plot_three_lengths_coupled_components_article(
        results_GaAs_AlGaAs;
        material_label = L"\mathrm{GaAs}/\mathrm{Al}_{0.3}\mathrm{Ga}_{0.7}\mathrm{As}",
        iLs = selected_iLs,
        state = 2,
        zoom_factor = 1.5,
        savepath = joinpath(plot_dir, "three_lengths_GaAs_coupled_components.pdf"),
    )
)



display(
    plot_three_lengths_unc_ema_coupled_article(
        results_InAs_AlAs;
        material_label = L"\mathrm{InAs}/\mathrm{AlAs}",
        iLs = selected_iLs,
        state = 2,
        zoom_factor = 1.5,
        savepath = joinpath(plot_dir, "three_lengths_InAs_unc_ema_coupled.pdf"),
    )
)

display(
    plot_three_lengths_coupled_components_article(
        results_InAs_AlAs;
        material_label = L"\mathrm{InAs}/\mathrm{AlAs}",
        iLs = selected_iLs,
        state = 2,
        zoom_factor = 1.5,
        savepath = joinpath(plot_dir, "three_lengths_InAs_coupled_components.pdf"),
    )
)
# ============================================================
# Weight diagnostics
# ============================================================
print_confinement_weight_table(
    results_GaAs_AlGaAs;
    iLs = [11, 5, 4],
    state = 2,
)
print_confinement_weight_table(
    results_InAs_AlAs;
    iLs = [11, 5, 4],
    state = 2,
)

# ============================================================
# Weight plots
# ============================================================

material_labels = [
    L"\mathrm{GaAs} / \mathrm{Al}_{0.3}\mathrm{Ga}_{0.7}\mathrm{As}",
    L"\mathrm{InAs} / \mathrm{AlAs}",
]

results_list = [
    results_GaAs_AlGaAs,
    results_InAs_AlAs,
]




for state in 1:N
    display(
        plot_energy_vs_Lwell_single_state_article(
            Lwells;
            results = results_GaAs_AlGaAs,
            material_label = L"\mathrm{GaAs} / \mathrm{Al}_{0.3}\mathrm{Ga}_{0.7}\mathrm{As}",
            state = state,
            savepath = joinpath(
                plot_dir,
                "appendix_energy_GaAs_AlGaAs_state$(state)f.pdf",
            ),
        )
    )

    display(
        plot_energy_vs_Lwell_single_state_article(
            Lwells;
            results = results_InAs_AlAs,
            material_label = L"\mathrm{InAs} / \mathrm{AlAs}",
            state = state,
            savepath = joinpath(
                plot_dir,
                "appendix_energy_InAs_AlAs_state$(state)f.pdf",
            ),
        )
    )
end

display(
    plot_conduction_weight_single_material_article(
        Lwells;
        results = results_InAs_AlAs,
        material_label = L"\mathrm{InAs} / \mathrm{AlAs}",
        state = 1,
        savepath = joinpath(
            plot_dir,
            "conduction_weight_InAs_AlAs_state1f.pdf",
        ),
    )
)

# ============================================================
# Notes
# ============================================================

# nevne Lowdin/Schur
# ved aa endre NC og NV vil bolgefunksjonene i variational case endre seg.
# Hoy nok er oppnadd naar de er like
# Bruker unp bolgefunksjoner i var, gir bedre konvergens.
# Mulig eff mass ikke er stabil nok
# Ved aa ikke bruke symetrisk w12, dette gjor at case 3 og case 4
# gaar fra diff = 0.001300579062613 eV til 0.005703184399542 eV
# relativt til full E 0.0226% til 0.0950%
# Vis at dersom Nc=Nv er for lav vil bolgen oscilere
# ved tol paa 1e-3 var det fortsatt tegn til osilering i var case,
# ved 1e-6 er det borte.
# Skru av coupling og se hvor mange Nc og Nv som trengs.
# Mener at det skal vaere faa

# Endre global fase
# Bare velge conduction band
# middelverdier på paramtere -> stabil E

function plot_total_density_two_ways(
    results;
    material_label,
    iL::Int = 1,
    state::Int = 1,
    zoom_factor::Float64 = 3.0,
    savepath = nothing,
)
    with_theme(theme_article()) do
        res = results[iL]

        x = res.x
        dx = res.dx
        Lwell = res.Lwell
        x_center = res.Lx / 2

        x_window = zoom_factor * Lwell
        x_min = max(minimum(x), x_center - x_window / 2)
        x_max = min(maximum(x), x_center + x_window / 2)

        Fc = res.Psi_c_full[:, state]
        Fv = res.Psi_v_full[:, state]

        psi_total = vcat(Fc, Fv)

        # Same normalization, using the full spinor
        norm_total = sqrt(sum(abs2, psi_total) * dx)

        Fc = Fc ./ norm_total
        Fv = Fv ./ norm_total
        psi_total = psi_total ./ norm_total

        ngrid = length(x)

        # Method 1: split full spinor back into components
        rho_from_full =
            abs2.(psi_total[1:ngrid]) .+
            abs2.(psi_total[ngrid+1:2*ngrid])

        # Method 2: use stored Fc and Fv directly
        rho_from_components = abs2.(Fc) .+ abs2.(Fv)

        fig = Figure(size = (760, 420), figure_padding = (40, 10, 15, 15))

        ax = Axis(
            fig[1, 1],
            xlabel = L"x\, [\mathrm{\AA}]",
            ylabel = L"|F(x)|^2",
            title = L"%$(material_label),\ L_{\mathrm{well}}=%$(round(Lwell, digits=1))\ \mathrm{\AA},\ state\ %$state",
            yticklabelspace = 55.0,
            xlabelsize = 26,
            ylabelsize = 26,
            titlesize = 24,
            xticklabelsize = 23,
            yticklabelsize = 23,
        )

        CairoMakie.xlims!(ax, x_min, x_max)

        lines!(
            ax,
            x,
            real.(rho_from_full),
            linestyle = :solid,
            linewidth = 2.8,
            label = L"\mathrm{from\ full\ spinor}",
        )

        lines!(
            ax,
            x,
            real.(rho_from_components),
            linestyle = :dash,
            linewidth = 2.2,
            label = L"|F_c(x)|^2 + |F_v(x)|^2",
        )

        vlines!(
            ax,
            [res.x_left, res.x_right],
            color = :black,
            linestyle = :dash,
            linewidth = 1.5,
            label = L"\mathrm{interface}",
        )

        CairoMakie.ylims!(
            ax,
            nothing,
            1.08 * maximum(real.(rho_from_components)),
        )

        Legend(fig[1, 2], ax, valign = :center, labelsize = 20)

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end
display(
    plot_total_density_two_ways(
        results_InAs_AlAs;
        material_label = L"\mathrm{InAs}/\mathrm{AlAs}",
        iL = 1,
        state = 1,
        zoom_factor = 3.0,
        savepath = joinpath(plot_dir, "total_density_two_ways_InAs_state1.pdf"),
    )
)

function plot_coupled_real_components_and_density(
    results;
    material_label,
    iL::Int = 1,
    state::Int = 1,
    zoom_factor::Float64 = 3.0,
    savepath = nothing,
)
    with_theme(theme_article()) do
        res = results[iL]

        x = res.x
        dx = res.dx
        Lwell = res.Lwell
        x_center = res.Lx / 2

        x_window = zoom_factor * Lwell
        x_min = max(minimum(x), x_center - x_window / 2)
        x_max = min(maximum(x), x_center + x_window / 2)

        Fc = copy(res.Psi_c_full[:, state])
        Fv = copy(res.Psi_v_full[:, state])

        # Normalize full spinor:
        # ∫ (|Fc|² + |Fv|²) dx = 1
        norm_total = sqrt(sum(abs2.(Fc) .+ abs2.(Fv)) * dx)

        Fc = Fc ./ norm_total
        Fv = Fv ./ norm_total

        # Global phase alignment:
        # Make Fc real and positive at its largest-amplitude point.
        iref = argmax(abs.(Fc))
        phase = exp(-im * angle(Fc[iref]))

        Fc = phase .* Fc
        Fv = phase .* Fv

        yc = real.(Fc)

        # Because the kp coupling contains -i d/dx, Fv is mostly imaginary
        # when Fc is chosen real.
        yv = imag.(Fv)

        rho_total = abs2.(Fc) .+ abs2.(Fv)

        fig = Figure(size = (750, 900), figure_padding = (35, 15, 15, 15))

        Label(
            fig[0, 1],
            L"%$(material_label),\ L_{\mathrm{well}}=%$(round(Lwell, digits=1))\ \mathrm{\AA},\ state\ %$state",
            fontsize = 28,
            font = :italic,
        )

        ax_c = Axis(
            fig[1, 1],
            xlabel = "",
            ylabel = L"\mathrm{Re}\,F_c(x)",
            title = L"\mathrm{Conduction\ component}",
            yticklabelspace = 55.0,
            xlabelsize = 24,
            ylabelsize = 24,
            titlesize = 24,
            xticklabelsize = 21,
            yticklabelsize = 21,
        )

        ax_v = Axis(
            fig[2, 1],
            xlabel = "",
            ylabel = L"\mathrm{Im}\,F_v(x)",
            title = L"\mathrm{Valence\ component}",
            yticklabelspace = 55.0,
            xlabelsize = 24,
            ylabelsize = 24,
            titlesize = 24,
            xticklabelsize = 21,
            yticklabelsize = 21,
        )

        ax_rho = Axis(
            fig[3, 1],
            xlabel = L"x\, [\mathrm{\AA}]",
            ylabel = L"|F(x)|^2",
            title = L"\mathrm{Total\ density}",
            yticklabelspace = 55.0,
            xlabelsize = 24,
            ylabelsize = 24,
            titlesize = 24,
            xticklabelsize = 21,
            yticklabelsize = 21,
        )

        for ax in (ax_c, ax_v, ax_rho)
            CairoMakie.xlims!(ax, x_min, x_max)

            vlines!(
                ax,
                [res.x_left, res.x_right],
                color = :black,
                linestyle = :dash,
                linewidth = 1.5,
            )
        end

        lines!(
            ax_c,
            x,
            yc,
            linestyle = :solid,
            label = L"\mathrm{Re}\,F_c(x)",
        )

        lines!(
            ax_v,
            x,
            yv,
            linestyle = :solid,
            label = L"\mathrm{Im}\,F_v(x)",
        )

        lines!(
            ax_rho,
            x,
            real.(rho_total),
            linestyle = :solid,
            label = L"|F_c(x)|^2 + |F_v(x)|^2",
        )

        y_cmax = maximum(abs.(yc))
        y_vmax = maximum(abs.(yv))

        CairoMakie.ylims!(ax_c, -1.08 * y_cmax, 1.08 * y_cmax)
        CairoMakie.ylims!(ax_v, -1.08 * y_vmax, 1.08 * y_vmax)
        CairoMakie.ylims!(ax_rho, nothing, 1.08 * maximum(real.(rho_total)))

        rowgap!(fig.layout, 18)

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end
display(
    plot_coupled_real_components_and_density(
        results_InAs_AlAs;
        material_label = L"\mathrm{InAs}/\mathrm{AlAs}",
        iL = 3,
        state = 2,
        zoom_factor = 2.0,
        savepath = joinpath(plot_dir, "real_components_density_InAs_state2.pdf"),
    )
)



# ============================================================
# Apparent confinement effective mass from E(Lwell)
# Only EMA and coupled cases
# ============================================================

function apparent_mass_from_local_fit(
    Lwells,
    energies,
    state::Int;
    window::Int = 5,
)
    @assert isodd(window) "window must be odd, e.g. 3, 5, or 7"

    half_window = Int((window - 1) ÷ 2)

    L_center_values = Float64[]
    mass_values = Float64[]
    slope_values = Float64[]
    intercept_values = Float64[]

    for i in (1 + half_window):(length(Lwells) - half_window)
        inds = (i - half_window):(i + half_window)

        L_fit = Float64.(Lwells[inds])
        E_fit = Float64.(energies[inds])

        if any(isnan, E_fit)
            push!(L_center_values, Lwells[i])
            push!(mass_values, NaN)
            push!(slope_values, NaN)
            push!(intercept_values, NaN)
            continue
        end

        # Fit:
        # E(L) = E0 + A / L^2
        x = 1.0 ./ L_fit.^2
        X = hcat(ones(length(x)), x)

        coeffs = X \ E_fit

        E0_fit = coeffs[1]
        A_fit = coeffs[2]

        # m_conf*/m0 = (hbar²/2m0) * pi² * n² / A
        m_eff =
            Constants.hbar2_over_2me *
            π^2 *
            state^2 /
            A_fit

        push!(L_center_values, Lwells[i])
        push!(mass_values, m_eff)
        push!(slope_values, A_fit)
        push!(intercept_values, E0_fit)
    end

    return (
        Lwell = L_center_values,
        mass = mass_values,
        slope = slope_values,
        intercept = intercept_values,
    )
end


function apparent_mass_for_case(
    Lwells,
    results,
    icase::Int,
    state::Int;
    window::Int = 5,
)
    energies = energy_for_state(results, icase, state)

    return apparent_mass_from_local_fit(
        Lwells,
        energies,
        state;
        window = window,
    )
end


# ============================================================
# Plot: apparent mass vs well width for one material and state
# Only EMA and coupled
# ============================================================

function plot_apparent_mass_vs_Lwell_article(
    Lwells;
    results,
    material_label,
    state::Int = 1,
    window::Int = 5,
    savepath = nothing,
)
    with_theme(theme_article()) do
        fig = Figure(
            size = (720, 440),
            figure_padding = (45, 15, 15, 15),
        )

        ax = Axis(
            fig[1, 1],
            xlabel = L"L_{\mathrm{well}}\, [\mathrm{\AA}]",
            ylabel = L"m_{\mathrm{conf}}^*/m_0",
            title = L"%$(material_label),\ \mathrm{state}\ %$state",
            xlabelsize = 26,
            ylabelsize = 26,
            titlesize = 25,
            xticklabelsize = 22,
            yticklabelsize = 22,
            yticklabelspace = 45.0,
        )

        case_indices = [2, 3]

        case_labels = [
            L"\mathrm{EMA}",
            L"\mathrm{Coupled}",
        ]

        line_styles = [
            :solid,
            :dash,
        ]

        for (j, icase) in enumerate(case_indices)
            data = apparent_mass_for_case(
                Lwells,
                results,
                icase,
                state;
                window = window,
            )

            lines!(
                ax,
                data.Lwell,
                data.mass,
                linestyle = line_styles[j],
                linewidth = 2.4,
                label = case_labels[j],
            )

            scatter!(
                ax,
                data.Lwell,
                data.mass,
                markersize = 8,
            )
        end

        Legend(
            fig[1, 2],
            ax,
            valign = :center,
            labelsize = 21,
        )

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end


# ============================================================
# Plot: several states for one material
# Only EMA and coupled
# ============================================================

function plot_apparent_mass_multiple_states_article(
    Lwells;
    results,
    material_label,
    states = 1:3,
    window::Int = 5,
    savepath = nothing,
)
    with_theme(theme_article()) do
        fig = Figure(
            size = (1450, 520),
            figure_padding = (25, 15, 15, 15),
        )

        Label(
            fig[1, 1:length(states)],
            material_label,
            fontsize = 30,
            font = :italic,
        )

        axes = Axis[]

        case_indices = [2, 3]

        case_labels = [
            L"\mathrm{EMA}",
            L"\mathrm{Coupled}",
        ]

        line_styles = [
            :solid,
            :dash,
        ]

        for (j, state) in enumerate(states)
            ax = Axis(
                fig[2, j],
                xlabel = L"L_{\mathrm{well}}\, [\mathrm{\AA}]",
                ylabel = L"m_{\mathrm{conf}}^*/m_0",
                title = L"\mathrm{State}\ %$state",
                xlabelsize = 25,
                ylabelsize = 25,
                titlesize = 25,
                xticklabelsize = 21,
                yticklabelsize = 21,
                yticklabelspace = 45.0,
            )

            push!(axes, ax)

            for (k, icase) in enumerate(case_indices)
                data = apparent_mass_for_case(
                    Lwells,
                    results,
                    icase,
                    state;
                    window = window,
                )

                lines!(
                    ax,
                    data.Lwell,
                    data.mass,
                    linestyle = line_styles[k],
                    linewidth = 2.4,
                    label = case_labels[k],
                )

                scatter!(
                    ax,
                    data.Lwell,
                    data.mass,
                    markersize = 8,
                )
            end
        end

        Legend(
            fig[2, length(states) + 1],
            axes[1],
            valign = :center,
            labelsize = 23,
        )

        if !isnothing(savepath)
            save(savepath, fig)
        end

        fig
    end
end


# ============================================================
# Optional: global fit over chosen Lwell interval
# Only EMA and coupled
# ============================================================

function global_apparent_mass_fit(
    Lwells,
    results,
    icase::Int,
    state::Int;
    fit_indices = eachindex(Lwells),
)
    energies = energy_for_state(results, icase, state)

    L_fit = Float64.(Lwells[fit_indices])
    E_fit = Float64.(energies[fit_indices])

    valid = .!isnan.(E_fit)

    L_fit = L_fit[valid]
    E_fit = E_fit[valid]

    x = 1.0 ./ L_fit.^2
    X = hcat(ones(length(x)), x)

    coeffs = X \ E_fit

    E0_fit = coeffs[1]
    A_fit = coeffs[2]

    m_eff =
        Constants.hbar2_over_2me *
        π^2 *
        state^2 /
        A_fit

    return (
        mass = m_eff,
        slope = A_fit,
        intercept = E0_fit,
    )
end


function print_global_mass_table(
    Lwells,
    results;
    material_name,
    states = 1:3,
    fit_indices = 5:length(Lwells),
)
    case_indices = [2, 3]

    case_names = [
        "EMA",
        "Coupled",
    ]

    println()
    println("============================================================")
    println("Global apparent confinement mass fit")
    println("Material: ", material_name)
    println("Fit over Lwell indices: ", fit_indices)
    println("============================================================")
    println("state | case       | m_conf*/m0")
    println("------|------------|------------")

    for state in states
        for (j, icase) in enumerate(case_indices)
            fit = global_apparent_mass_fit(
                Lwells,
                results,
                icase,
                state;
                fit_indices = fit_indices,
            )

            @printf(
                "%5d | %-10s | %10.6f\n",
                state,
                case_names[j],
                fit.mass,
            )
        end
    end
end


# ============================================================
# Generate plots
# ============================================================

mass_window = 5

display(
    plot_apparent_mass_multiple_states_article(
        Lwells;
        results = results_GaAs_AlGaAs,
        material_label = L"\mathrm{GaAs}/\mathrm{Al}_{0.3}\mathrm{Ga}_{0.7}\mathrm{As}",
        states = 1:3,
        window = mass_window,
        savepath = joinpath(
            plot_dir,
            "apparent_mass_GaAs_AlGaAs_states1to3_EMA_coupled.pdf",
        ),
    )
)

display(
    plot_apparent_mass_multiple_states_article(
        Lwells;
        results = results_InAs_AlAs,
        material_label = L"\mathrm{InAs}/\mathrm{AlAs}",
        states = 1:3,
        window = mass_window,
        savepath = joinpath(
            plot_dir,
            "apparent_mass_InAs_AlAs_states1to3_EMA_coupled.pdf",
        ),
    )
)


# ============================================================
# Optional single-state plots
# ============================================================

display(
    plot_apparent_mass_vs_Lwell_article(
        Lwells;
        results = results_GaAs_AlGaAs,
        material_label = L"\mathrm{GaAs}/\mathrm{Al}_{0.3}\mathrm{Ga}_{0.7}\mathrm{As}",
        state = 1,
        window = mass_window,
        savepath = joinpath(
            plot_dir,
            "apparent_mass_GaAs_AlGaAs_state1_EMA_coupled.pdf",
        ),
    )
)

display(
    plot_apparent_mass_vs_Lwell_article(
        Lwells;
        results = results_InAs_AlAs,
        material_label = L"\mathrm{InAs}/\mathrm{AlAs}",
        state = 1,
        window = mass_window,
        savepath = joinpath(
            plot_dir,
            "apparent_mass_InAs_AlAs_state1_EMA_coupled.pdf",
        ),
    )
)


# ============================================================
# Optional printed global mass tables
# ============================================================

print_global_mass_table(
    Lwells,
    results_GaAs_AlGaAs;
    material_name = "GaAs / Al0.3Ga0.7As",
    states = 1:3,
    fit_indices = 5:length(Lwells),
)

print_global_mass_table(
    Lwells,
    results_InAs_AlAs;
    material_name = "InAs / AlAs",
    states = 1:3,
    fit_indices = 5:length(Lwells),
)
