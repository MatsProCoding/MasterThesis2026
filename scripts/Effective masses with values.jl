if !isdefined(Main, :Project)
    include("../src/Project.jl")
end

using .Project.Constants
using .Project.Effective_Masses


using CairoMakie
using LinearAlgebra


function theme_article()
    merge(
        CairoMakie.theme_latexfonts(),
        CairoMakie.Theme(
            fontsize = 16,
            Axis = (
                xlabelsize = 22 ,
                ylabelsize = 22,
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
                labelsize = 19,
                framevisible = true,
            ),
            Lines = (linewidth = 3,),
            Scatter = (markersize = 8, strokewidth = 0.8, strokecolor = :black)
        )
    )
end
function bands(k, state, H)
    vals = sort(real(eigvals(Hermitian(H(k)))))
    return vals[state]
end

function effective_mass(dk, H; nbands=2)
    masses = Float64[]
    K_values = (-2:2) .* dk

    for band in 1:nbands
        E = [bands(k, band, H) for k in K_values]
        d2E_dk2 = (-E[5] + 16E[4] - 30E[3] + 16E[2] - E[1]) / (12 * dk^2)
        m_eff = Constants.hbar2_over_me / d2E_dk2
        push!(masses, m_eff)
    end

    return masses
end

function make_2x2(Ec, Ev, Ep)
    W11(k) = Ec + Constants.hbar2_over_2me * k^2 
    W22(k) = Ev + Constants.hbar2_over_2me * k^2
    W12(k) =  k * sqrt(Constants.hbar2_over_2me * Ep) 
    W21(k) = W12(k)'    
    
    H_pert(k) = [W11(k)  W12(k)
                    W21(k)  W22(k)]

    return H_pert
end



function make_2x2_extra(Ec, Ev, Ep, F; Epeff = Ep)
    W11(k) = Ec + Constants.hbar2_over_2me * k^2 *(1+2F)
    W22(k) = Ev - Constants.hbar2_over_2me * k^2 *(1-2F)
    W12(k) =  k * sqrt(Constants.hbar2_over_2me * Epeff) 
    W21(k) = W12(k)'    
    
    H_pert(k) = [W11(k)  W12(k)
                    W21(k)  W22(k)]

    return H_pert
end

# function make_4x4(Ec, Ev, Ep, F)
#     W11(k) = Ec + Constants.hbar2_over_2me * k^2 *(1+2F)
#     W22(k) = Ev - Constants.hbar2_over_2me * k^2
#     W12(k) = k * sqrt(Constants.hbar2_over_2me * Ep)
#     W21(k) = W12(k)'
    
#     H_pert(k) = [W11(k)  W12(k)  0.0  0.0
#                     W21(k)  W22(k)  0.0  0.0
#                     0.0     0.0     W22(k)  W12(k)
#                     0.0     0.0     W21(k)  W22(k)]

#     return H_pert
# end


dk = 0.00001 

Material = ["GaAs", "AlAs", "InAs"]
Eg = [1.519, 3.099, 0.417]
Ep = [28.8, 21.1, 21.5]
EPeff = [27.04,20.6,18.04]
F = [-1.94, -0.48, -2.9]

E_v = 0.0 

mass_material_2x2 = []
mass_material_2x2_f = []
mass_material_2x2_f_and_eff = []
# mass_material_4x4 = []
mass_material_2x2_f
E_aktual = []
for i in 1:length(Material)
    H2x2 = make_2x2(Eg[i], E_v, Ep[i])
    H2x2_f = make_2x2_extra(Eg[i], E_v, Ep[i], F[i])
    H2x2_eff = make_2x2_extra(Eg[i], E_v, Ep[i], F[i], Epeff=EPeff[i])
    # H4x4 = make_4x4(Eg[i], E_v, Ep[i], F[i])

    m2x2 = effective_mass(dk, H2x2, nbands=2)
    m2x2_f = effective_mass(dk, H2x2_f, nbands=2)
    m2x2_f_and_eff = effective_mass(dk, H2x2_eff, nbands=2)
    # m4x4 = effective_mass(dk, H4x4, nbands=4)

    push!(mass_material_2x2, m2x2)
    push!(mass_material_2x2_f, m2x2_f)
    push!(mass_material_2x2_f_and_eff, m2x2_f_and_eff)
    # push!(mass_material_4x4, m4x4)
end
H2x2_GaAs = make_2x2(Eg[1], E_v, Ep[1])
eigvals(H2x2_GaAs(0.0))
for i in mass_material_2x2
    println("2x2: ", i)
end

correct_m = [0.067, 0.026, 0.15]
correct_m_range = ["0.065 - 0.07","0.06 - 0.015","0.023 - 0.030"]
mass_material_2x2_f_and_eff

println("Effective masses (in units of electron mass):")
for i in 1:length(Eg)
    println("Material $(Material[i]): m* = $(mass_material_2x2[i][end]) (correct: $(correct_m[i]) [$(correct_m_range[i])])")
end

println("\nEffective masses with 2x2 model (with F):")
for i in 1:length(Eg)
    println("Material $(Material[i]): m* = $(mass_material_2x2_f[i][end]) (correct: $(correct_m[i]) [$(correct_m_range[i])])")
end


println("\nEffective masses with 2x2 model (with F and effective Ep):")
for i in 1:length(Eg)
    println("Material $(Material[i]): m* = $(mass_material_2x2_f_and_eff[i][end]) (correct: $(correct_m[i]) [$(correct_m_range[i])])")
end
# println("\nEffective masses with 4x4 model:")
# for i in 1:length(Eg)
#     println("Material $(Material[i]): m* = $(mass_material_4x4[i][end]) (correct: $(correct_m[i]) [$(correct_m_range[i])])  difference: $(abs(mass_material_4x4[i][end] - correct_m[i]))")
# end

# --- band helpers for 2x2 (just reuse your bands()) ---
# bands(k, state, H) must already exist

# ----------------- data -----------------
mass_material_2x2
kgrid = -0.2:0.002:0.2
sonon2= []
for i in 1:length(Material)
    mstar_ac = mass_material_2x2[i][end]
    msstar_pop = mass_material_2x2[i][1]
    E_akkk = Constants.hbar2_over_me/mstar_ac * kgrid.^2
    hdhd = Constants.hbar2_over_me/msstar_pop * kgrid.^2
    push!(E_aktual, E_akkk)
    push!(sonon2, hdhd)
end

mass_material_2x2

function make_fig(name="Test")
    CairoMakie.with_theme(theme_article()) do 
            
        fig = Figure(size=(850, 400))

        # Material 1 - GaAs
        H2_GaAs = make_2x2(Eg[1], E_v, Ep[1])
        ax1 = Axis(fig[1,1], xlabel="k (Å⁻¹)", ylabel="Energy (eV)", title="GaAs (2×2)", limits=(nothing, (-1.6, 3.2)))
        lines!(ax1, kgrid, bands.(kgrid, 1, Ref(H2_GaAs)), label="Valence")
        lines!(ax1, kgrid, bands.(kgrid, 2, Ref(H2_GaAs)), label="Conduction")
        lines!(ax1, kgrid, E_aktual[1].+Eg[1], label="Parabolic fit", linestyle=:dash)
        lines!(ax1, kgrid, sonon2[1], label="Parabolic fit with Ep_eff", linestyle=:dashdot)
        # Material 2 - AlAs
        H2_AlAs = make_2x2(Eg[2], E_v, Ep[2])
        ax2 = Axis(fig[1,2], xlabel="k (Å⁻¹)", ylabel="Energy (eV)", title="AlAs (2×2)", limits=(nothing, (-0.6, 4.4)))
        lines!(ax2, kgrid, bands.(kgrid, 1, Ref(H2_AlAs)), label="Valence")
        lines!(ax2, kgrid, bands.(kgrid, 2, Ref(H2_AlAs)), label="Conduction")
        lines!(ax2, kgrid, E_aktual[2].+Eg[2], label="Parabolic fit", linestyle=:dash)
        lines!(ax2, kgrid, sonon2[2], label="Parabolic fit with Ep_eff", linestyle=:dashdot)

        # Material 3 - InAs
        H2_InAs = make_2x2(Eg[3], E_v, Ep[3])
        ax3 = Axis(fig[1,3], xlabel="k (Å⁻¹)", ylabel="Energy (eV)", title="InAs (2×2)", limits=(nothing, (-2, 2.4)))
        l1=lines!(ax3, kgrid, bands.(kgrid, 1, Ref(H2_InAs)), label="Valence")
        l2=lines!(ax3, kgrid, bands.(kgrid, 2, Ref(H2_InAs)), label="Conduction")
        l3=lines!(ax3, kgrid, E_aktual[3].+Eg[3], label="Parabolic fit", linestyle=:dash)
        l4=lines!(ax3, kgrid, sonon2[3], label="Parabolic fit with Ep_eff", linestyle=:dashdot)
        Legend(fig[1,4], [l2, l1], ["Conduction", "Valence"], framevisible=false)

        display(fig)
        save("$(name)_effective_masses_2x2.pdf", fig)
    end
end

make_fig("2x2_enkel")

#%%
# ----------------- plot grid 2x3 -----------------
p = Plots.plot(layout = (2,3), size=(1000,600),
         xlabel="k (Å⁻¹)", ylabel="Energy (eV)", ylims=(-2, 5))

for i in 1:3
    
    H2 = make_2x2(Eg[i], E_v, Ep[i], F[i])
    H4 = make_4x4(Eg[i], E_v, EPeff[i], F[i])

    # ---- 2x2 (første rad) ----
    Plots.plot!(p[i], kgrid, k -> bands(k,1,H2),
        label="Valence", linewidth=2)

    Plots.plot!(p[i], kgrid, k -> bands(k,2,H2),
        label="Conduction", linewidth=2)

    title!(p[i], Material[i] * " (2×2)")


    # ---- 4x4 (andre rad) ----
    Plots.plot!(p[i+3], kgrid, k -> bands(k,1,H4),
        label="Band 1", linewidth=2)

    Plots.plot!(p[i+3], kgrid, k -> bands(k,2,H4),
        label="Band 2", linewidth=2)

    Plots.plot!(p[i+3], kgrid, k -> bands(k,3,H4),
        label="Band 3", linewidth=2, linestyle=:dash)

    Plots.  plot!(p[i+3], kgrid, k -> bands(k,4,H4),
        label="Band 4", linewidth=2)

    title!(p[i+3], Material[i] * " (4×4)")

end

display(p)

# Her må vi fine ut av F


dk_list = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002, 0.000001]
conv_masses_gaas = []
conv_masses_inas = []
conv_masses_alas = []
for dk in dk_list
    H2x2_conv_GaAs = make_2x2(Eg[1], Ev, Ep[1])
    m2x2_conv_GaAs = effective_mass(dk, H2x2_conv_GaAs, nbands=2)
    H2x2_conv_InAs = make_2x2(Eg[3], Ev, Ep[3])
    m2x2_conv_InAs = effective_mass(dk, H2x2_conv_InAs, nbands=2)
    H2x2_conv_AlAs = make_2x2(Eg[2], Ev, Ep[2])
    m2x2_conv_AlAs = effective_mass(dk, H2x2_conv_AlAs, nbands=2)
    push!(conv_masses_gaas, m2x2_conv_GaAs[2])
    push!(conv_masses_inas, m2x2_conv_InAs[2])
    push!(conv_masses_alas, m2x2_conv_AlAs[2])
end
conv_masses

CairoMakie.with_theme(theme_article()) do
    fig = Figure()
    ax = Axis(fig[1,1],
        xlabel = "dk (Å⁻¹)",
        ylabel = L"Effective mass ($m_0$)",
        title = "Convergence of CB effective mass",
        xscale = log10,
        xreversed = true   # høy dk til venstre, lav til høyre
    )

    CairoMakie.scatter!(ax, dk_list, conv_masses_gaas, markersize=8)
    l1=CairoMakie.lines!(ax, dk_list, conv_masses_gaas)
    CairoMakie.scatter!(ax, dk_list, conv_masses_inas, markersize=8)
    l2= CairoMakie.lines!(ax, dk_list, conv_masses_inas)
    CairoMakie.scatter!(ax, dk_list, conv_masses_alas, markersize=8)
    l3=CairoMakie.lines!(ax, dk_list, conv_masses_alas)
    Legend(fig[1,2], [l3, l1, l2], ["AlAs","GaAs", "InAs"], framevisible=false)
    display(fig)
    save("effective_mass_convergence_2x2.pdf", fig)
end

# Referanselinje hvis du vil
# hlines!(ax, [0.067], linestyle=:dash, color=:red, label="Vurgaftman m*=0.067")

conv_masses_gaas_f_and_eff = []
conv_masses_inas_f_and_eff = []
conv_masses_alas_f_and_eff = []
for dk in dk_list
    H2x2_conv_GaAs_f_and_eff = make_2x2_extra(Eg[1], Ev, Ep[1], F[1], Epeff=EPeff[1])
    m2x2_conv_GaAs_f_and_eff = effective_mass(dk, H2x2_conv_GaAs_f_and_eff, nbands=2)
    H2x2_conv_InAs_f_and_eff = make_2x2_extra(Eg[3], Ev, Ep[3], F[3], Epeff=EPeff[3])
    m2x2_conv_InAs_f_and_eff = effective_mass(dk, H2x2_conv_InAs_f_and_eff, nbands=2)
    H2x2_conv_AlAs_f_and_eff = make_2x2_extra(Eg[2], Ev, Ep[2], F[2], Epeff=EPeff[2])
    m2x2_conv_AlAs_f_and_eff = effective_mass(dk, H2x2_conv_AlAs_f_and_eff, nbands=2)
    push!(conv_masses_gaas_f_and_eff, m2x2_conv_GaAs_f_and_eff[2])
    push!(conv_masses_inas_f_and_eff, m2x2_conv_InAs_f_and_eff[2])
    push!(conv_masses_alas_f_and_eff, m2x2_conv_AlAs_f_and_eff[2])
end


fig2 = Figure()
ax2 = Axis(fig2[1,1],
    xlabel = "dk (Å⁻¹)",
    ylabel = "Effective mass (m₀)",
    title = "Convergence of CBM effective mass (GaAs, 2×2)",
    xscale = log10,
    xreversed = true   # høy dk til venstre, lav til høyre
)

CairoMakie.scatter!(ax2, dk_list, conv_masses_gaas_f_and_eff, markersize=8)
l1=CairoMakie.lines!(ax2, dk_list, conv_masses_gaas_f_and_eff)
h1 = CairoMakie.hlines!(ax2, correct_m[1], linestyle=:dash, color=:red, label="Vurgaftman m*=0.067")

CairoMakie.scatter!(ax2, dk_list, conv_masses_inas_f_and_eff, markersize=8)
l2= CairoMakie.lines!(ax2, dk_list, conv_masses_inas_f_and_eff)
h2 = CairoMakie.hlines!(ax2, correct_m[2], linestyle=:dash, color=:blue, label="Vurgaftman m*=0.026")

CairoMakie.scatter!(ax2, dk_list, conv_masses_alas_f_and_eff, markersize=8)
l3=CairoMakie.lines!(ax2, dk_list, conv_masses_alas_f_and_eff)
h3 = CairoMakie.hlines!(ax2, correct_m[3], linestyle=:dash, color=:green, label="Vurgaftman m*=0.15")

Legend(fig2[1,2], [l1, l2, l3, h1, h2, h3], ["GaAs", "InAs", "AlAs", "Vurgaftman m*=0.067", "Vurgaftman m*=0.026", "Vurgaftman m*=0.15"], framevisible=false)
display(fig2)
fig2 = Figure(size=(800, 600))
ax_top = Axis(fig2[1,1],
    xlabel = "dk", ylabel = "Effective mass (m₀)",
    title = "Convergence of CBM effective mass",
    xscale = log10, xreversed = true)

ax_bot = Axis(fig2[2,1],
    xlabel = "dk", ylabel = "Relative error",
    title = "|m* - m*_ref| / m*_ref",
    xscale = log10, yscale = log10, xreversed = true)

colors = [:blue, :red, :green]
labels = ["GaAs", "InAs", "AlAs"]
all_masses = [conv_masses_gaas_f_and_eff, conv_masses_inas_f_and_eff, conv_masses_alas_f_and_eff]

for (masses, ref, label, color) in zip(all_masses, correct_m, labels, colors)
    l = CairoMakie.lines!(ax_top, dk_list, masses, color=color)
    CairoMakie.scatter!(ax_top, dk_list, masses, markersize=8, color=color)
    CairoMakie.hlines!(ax_top, ref, linestyle=:dash, color=color)

    rel_err = abs.(masses .- ref) ./ ref
    CairoMakie.lines!(ax_bot, dk_list, rel_err, color=color)
    CairoMakie.scatter!(ax_bot, dk_list, rel_err, markersize=8, color=color)
end

Legend(fig2[1,2], [LineElement(color=c) for c in colors], labels, framevisible=false)
display(fig2)

##
# Set valence band of GaAs as energy reference = 0
function make_QW_hamiltonian(z_grid, Ec_profile, Ev_profile, Ep_profile)
    N = length(z_grid)
    dz = z_grid[2] - z_grid[1]
    
    H = zeros(Complex{Float64}, 2N, 2N)
    C = Constants.hbar2_over_2me

    for i in 1:N
        # ── Diagonal: W11 and W22 at k=0 ──
        H[i,   i]   = Ec_profile[i]        # W11(0) = Ec
        H[N+i, N+i] = Ev_profile[i]        # W22(0) = Ev

        # ── W11 kinetic: C*k² → second difference ──
        if i > 1
            H[i, i-1] += -C / dz^2
            H[i, i]   +=  C / dz^2
        end
        if i < N
            H[i, i+1] += -C / dz^2
            H[i, i]   +=  C / dz^2
        end

        # ── W22 kinetic: -C*k² → second difference ──
        if i > 1
            H[N+i, N+i-1] += -C / dz^2
            H[N+i, N+i]   +=  C / dz^2
        end
        if i < N
            H[N+i, N+i+1] += -C / dz^2
            H[N+i, N+i]   +=  C / dz^2
        end

        # ── W12/W21: k*sqrt(C*Ep) → first difference ──
        kp = sqrt(C * Ep_profile[i])
        if i < N
            H[i,     N+i+1] += -1im * kp / (2*dz)
            H[N+i+1, i]     +=  1im * kp / (2*dz)
        end
        if i > 1
            H[i,     N+i-1] +=  1im * kp / (2*dz)
            H[N+i-1, i]     += -1im * kp / (2*dz)
        end
    end

    return Hermitian(H)
end

function in_well(z, L_well)
    return abs(z) < L_well / 2
end

# Define your QW structure
L_well  = 400         # 10 nm well  
N       = 800
z_grid  = range(-600, 600, length=N)
Ev_GaAs   = 0.0        # eV  (reference)
Eg_GaAs   = 1.519      # eV  (at 4K)
Ec_GaAs   = Ev_GaAs + Eg_GaAs   # = 1.519 eV
Ep_GaAs   = Ep[1]
F_GaAs    = F[1]
x = 0.2
Eg_AlGaAs = 1.776  # eV  recalculate for x=0.2
F_AlGaAs  = F[2]*0.2 + F[1]*0.8
Ep_AlGaAs = Ep[2]*0.2 + Ep[1]*0.8
   

delta_Ec  = 0.88 * (Eg_AlGaAs - Eg_GaAs)   # 88:12 band offset rule
delta_Ev  = 0.12 * (Eg_AlGaAs - Eg_GaAs)

Ec_AlGaAs = Ec_GaAs + delta_Ec
Ev_AlGaAs = Ev_GaAs - delta_Ev

# Build potential profiles
Eg_profile = [in_well(z, L_well) ? Eg_GaAs : Eg_AlGaAs for z in z_grid]
Ec_profile = [in_well(z, L_well) ? Ec_GaAs : Ec_AlGaAs for z in z_grid]
Ev_profile = [in_well(z, L_well) ? Ev_GaAs : Ev_AlGaAs for z in z_grid]
Ep_profile = [in_well(z, L_well) ? Ep_GaAs : Ep_AlGaAs for z in z_grid]
Epeff_profile = Ep_profile
F_profile  = [in_well(z, L_well) ? F_GaAs : F_AlGaAs for z in z_grid]
# Sanity check effective masses
m_eff_well    = 1.0 / (1.0 + 2*F_GaAs   + Ep_GaAs/Eg_GaAs)
m_eff_barrier = 1.0 / (1.0 + 2*F_AlGaAs + Ep_AlGaAs/Eg_AlGaAs)
println("m_eff well    = ", m_eff_well,    " m0")  # expect ~0.067
println("m_eff barrier = ", m_eff_barrier, " m0")  # expect ~0.080

H = make_QW_hamiltonian(z_grid, Ec_profile, Ev_profile, Ep_profile)

    # Get eigenvalues
energies_real = sort(real(eigvals(Hermitian(H))))
E_electron = filter(e -> e > Ec_GaAs, energies_real)
E_e1  = first(filter(e -> e > Ec_GaAs, energies_real))
E_e2 = energies_real[findfirst(e -> e > E_e1, energies_real)]
E_e3 = energies_real[findfirst(e -> e > E_e2, energies_real)]
E_hh1 = last(filter(e -> e < Ev_GaAs, energies_real))
println("First electron energy: ", E_e1, " eV")
println("First hole energy:     ", E_hh1, " eV")
println("E_transition:          ", E_e1 - E_hh1, " eV")
# Lowest positive eigenvalues → electron levels
# Highest negative eigenvalues → hole levels

    
