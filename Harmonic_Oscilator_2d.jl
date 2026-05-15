# ===== Initializing =====
if !isdefined(Main, :Project)
    include("../src/Project.jl")
end

using .Project.Constants
using .Project.Solver_1D
using .Project.Math
using .Project.Eval
include("plots.jl")
# ===== External packages =====

using Arpack
using Printf
using CairoMakie
# ===== Parameters =====
lw = 800
L0 = -lw
Lx = lw
Ly = lw
mstar=0.067
omega = 0.010/sqrt(2 * Constants.hbar2_over_2me / mstar)

V_HO(x,y) = 0.5 * omega^2 * x^2 + 0.5 * omega^2 * y^2 



# ===== Analytical results =====


function energy_analytical(n)
    return sqrt(2 * Constants.hbar2_over_2me/mstar) * omega * (n - 0.5)
end

function energy_analytical_2D(nx, ny)
    return energy_analytical(nx) + energy_analytical(ny)
end


energy_analytical_2D(1,1)

function psi_analytical_2(x, y, nx, ny)

    beta = sqrt(omega / sqrt(2 * Constants.hbar2_over_2me/mstar))

    eta_x = beta * x
    eta_y = beta * y
    # Hermite-polynom
    H_x = hermiteH(nx-1, eta_x)* exp(-eta_x^2 / 2)
    H_y = hermiteH(ny-1, eta_y) * exp(-eta_y^2 / 2)

    # normalisering

    return  H_x * H_y

end

# ===== Comparison of analytical and numerical =====


ms = [(32,32),(64, 64), (128, 128), (256, 256), (512, 512)]

function builder(N)
    m, n = N
    xint, xhalf, dx = make_grid(L0, Lx, m)
    yint, yhalf, dy = make_grid(L0, Ly, n)
    H = build_H2D(L0, Lx, L0, Ly; mass_x=mstar, mass_y=mstar, V=V_HO, m=m, n=n, display_info = false)
    E, psi = eigs(H, nev=7, which=:SR, maxiter=300_000)
        return (
        E = real(E),
        psi = psi,
        x = xint,
        y = yint
    )
end

res = convergence_report(ms, builder, energy_analytical_2D; nmax=5, type="HO_2D_new_2", save_plot=true)

E = res.E
psi = res.psi
xint = res.x
yint = res.y

energy = energy_report(E, energy_analytical_2D; nmax=1, display_info = true)
#wave = wavefunction_report(xint, V_HO, E, psi, psi_analytical; nmax=5, display_info = true, display_plot = true)

psi12 = reshape(psi[:,2], ms[end][1]-1, ms[end][2]-1)
psi21 = reshape(psi[:,3], ms[end][1]-1, ms[end][2]-1)


add = vec(psi21 - psi12')
sub = vec(psi12 - psi21')
dx = xint[2] - xint[1]
dA = dx^2
psi12   = normalize_wavefunction(add, dA)
psi21 = normalize_wavefunction(sub, dA)


psi3 = reshape(psi[:,4], ms[end][1]-1, ms[end][2]-1)
psi4 = reshape(psi[:,5], ms[end][1]-1, ms[end][2]-1)

add = vec(psi3 + psi4')
sub = vec(psi3 - psi4')
dx = xint[2] - xint[1]
dA = dx^2
psi13   = normalize_wavefunction(add, dA)
psi31 = normalize_wavefunction(sub, dA)

im_list_ho = []
CairoMakie.heatmap(xint, yint, reshape(psi[:,6], length(xint), length(yint)))
function plot_wavefunction_2d(x, y, psi_vec, psi_ana, E, E_ana)

    Nx = length(x)
    Ny = length(y)
    dA = (x[2] - x[1]) * (y[2] - y[1])

    # reshape ONLY if needed
    psi2D = ndims(psi_vec) == 1 ? reshape(psi_vec, Nx, Ny) : psi_vec
    psi_ana2D = ndims(psi_ana) == 1 ? reshape(psi_ana, Nx, Ny) : psi_ana

    psi_num_v = vec(real(psi2D))
    psi_ana_v = vec(real(psi_ana2D))
    psi_num_v = normalize_wavefunction(psi_num_v, dA)
    psi_ana_v = normalize_wavefunction(psi_ana_v, dA)
    # Global phase/sign alignment before error metrics
    if real(dot(psi_num_v, psi_ana_v)) < 0
        psi_num_v .*= -1
    end

    overlap_cos = abs(dot(psi_num_v, psi_ana_v)) / (norm(psi_num_v) * norm(psi_ana_v))
    l2_err = sqrt(sum(abs2, psi_num_v .- psi_ana_v) * dA)

    fig = Figure(size = (900, 500))
    push!(im_list_ho, fig)
    # ---------------- NUMERICAL ----------------
    ax1 = Axis(fig[1, 1],
        title = "Numerical wavefunction",
        xlabel = "x", ylabel = "y", titlesize = 18*1.5, xlabelsize = 18*1.5, ylabelsize = 18*1.5,
        xticklabelsize = 13*1.5, yticklabelsize = 13*1.5
    )
    CairoMakie.heatmap!(ax1, x, y, reshape(psi_num_v, Nx, Ny); colormap = :inferno)
    # radius of classical turning point
    r = sqrt(2E) / omega

    θ = LinRange(0, 2π, 300)

    CairoMakie.lines!(ax1,
        r .* cos.(θ),
        r .* sin.(θ),
        color = :white,
        linewidth = 2
    )

    scal=2
    CairoMakie.xlims!(ax1, -scal*r, scal*r)
    CairoMakie.ylims!(ax1, -scal*r, scal*r)

    # ---------------- ANALYTICAL ----------------
    ax2 = Axis(fig[1, 2],
        title = "Analytical wavefunction",
        xlabel = "x", ylabel = "y", titlesize = 18*1.5, xlabelsize = 18*1.5, ylabelsize = 18*1.5,
        xticklabelsize = 13*1.5, yticklabelsize = 13*1.5
    )
    CairoMakie.heatmap!(ax2, x, y, psi_ana2D; colormap = :inferno)

    CairoMakie.lines!(ax2,
        r .* cos.(θ),
        r .* sin.(θ),
        color = :white,
        linewidth = 2
    )
    CairoMakie.xlims!(ax2, -scal*r, scal*r)
    CairoMakie.ylims!(ax2, -scal*r, scal*r)

    Label(
        fig[0, :],
        @sprintf("E_num = %.6f eV, E_ana = %.6f eV, dot = %.6f, L2 = %.3e", E, E_ana, overlap_cos, l2_err),
        fontsize = 16
    )

   
    return (dot_product = overlap_cos, l2_error = l2_err)
end

cases = [
    (label = "(1,1)", psi_num = psi[:,1], psi_ana = psi_analytical_2.(xint, yint', 1, 1), E_num = E[1], E_ana = energy_analytical_2D(1, 1)),
    (label = "(2,1)", psi_num = psi12, psi_ana = psi_analytical_2.(xint, yint', 2, 1), E_num = E[2], E_ana = energy_analytical_2D(2, 1)),
    (label = "(1,2)", psi_num = psi21, psi_ana = psi_analytical_2.(xint, yint', 1, 2), E_num = E[3], E_ana = energy_analytical_2D(1, 2)),
    (label = "(1,3)", psi_num = psi13, psi_ana = psi_analytical_2.(xint, yint', 1, 3), E_num = E[4], E_ana = energy_analytical_2D(1, 3)),
    (label = "(3,1)", psi_num = psi31, psi_ana = psi_analytical_2.(xint, yint', 3, 1), E_num = E[5], E_ana = energy_analytical_2D(3, 1)),
    (label = "(2,2)", psi_num = psi[:,6], psi_ana = psi_analytical_2.(xint, yint', 2, 2), E_num = E[6], E_ana = energy_analytical_2D(2, 2)),
]

psi_plotting = reshape(psi[:,4], ms[end][1]-1, ms[end][2]-1)
CairoMakie.heatmap(xint, yint, psi_plotting; colormap=:inferno)
E_ho = [c.E_num for c in cases]
wavefig = Figure(size = (1200, 2100))
dot_vals = Float64[]
l2_vals = Float64[]
im_list_ho = []
for (i, c) in enumerate(cases)
    Nx = length(xint)
    Ny = length(yint)
    psi_num2D = ndims(c.psi_num) == 1 ? reshape(c.psi_num, Nx, Ny) : c.psi_num
    psi_ana2D = ndims(c.psi_ana) == 1 ? reshape(c.psi_ana, Nx, Ny) : c.psi_ana

    dA = (xint[2] - xint[1]) * (yint[2] - yint[1])
    psi_num_v = normalize_wavefunction(vec(real(psi_num2D)), dA)
    psi_ana_v = normalize_wavefunction(vec(real(psi_ana2D)), dA)
    if real(dot(psi_num_v, psi_ana_v)) < 0
        psi_num_v .*= -1
    end
    dot_val = abs(dot(psi_num_v, psi_ana_v)) / (norm(psi_num_v) * norm(psi_ana_v))
    l2_val = sqrt(sum(abs2, psi_num_v .- psi_ana_v) * dA)
    push!(dot_vals, dot_val)
    push!(l2_vals, l2_val)

    ax_num = Axis(
        wavefig[i, 1],
        title = "$(c.label) numerical",
        xlabel = "x",
        ylabel = "y"
    )
    CairoMakie.heatmap!(ax_num, xint, yint, psi_num2D; colormap = :thermal)
    r = sqrt(2 * c.E_num) / omega
    θ = LinRange(0, 2pi, 300)
    CairoMakie.lines!(ax_num, r .* cos.(θ), r .* sin.(θ), color = :white, linewidth = 2)
    scal = 2
    CairoMakie.xlims!(ax_num, -scal * r, scal * r)
    CairoMakie.ylims!(ax_num, -scal * r, scal * r)
    
    ho_fig = Figure(size=(400,400))
    ho_ax_num = Axis(
        ho_fig[1, 1],
        title = "$(c.label) numerical",
        xlabel = "x",
        ylabel = "y"
    )
    psi_ana2D = reshape(c.psi_ana, Nx, Ny)
    ho_heat=CairoMakie.heatmap!(ho_ax_num, xint, yint, psi_num2D; colormap = :thermal)
    r = sqrt(2 * c.E_num) / omega
    θ = LinRange(0, 2pi, 300)
    CairoMakie.lines!(ho_ax_num, r .* cos.(θ), r .* sin.(θ), color = :white, linewidth = 2)
    scal = 2
    CairoMakie.xlims!(ho_ax_num, -scal * r, scal * r)
    CairoMakie.ylims!(ho_ax_num, -scal * r, scal * r)
    cool_bar = CairoMakie.Colorbar(ho_fig[1, 2], ho_heat, width = 15)
    push!(im_list_ho, psi_num2D)


    ax_ana = Axis(
        wavefig[i, 2],
        title = "$(c.label) analytical",
        xlabel = "x",
        ylabel = "y"
    )
    CairoMakie.heatmap!(ax_ana, xint, yint, psi_ana2D; colormap = :thermal)
    CairoMakie.lines!(ax_ana, r .* cos.(θ), r .* sin.(θ), color = :white, linewidth = 2)
    CairoMakie.xlims!(ax_ana, -scal * r, scal * r)
    CairoMakie.ylims!(ax_ana, -scal * r, scal * r)
end
display(wavefig)
xint


im_list_ho

metrics = [(cases[i].label, dot_vals[i], l2_vals[i]) for i in eachindex(cases)]

println("\nWavefunction Comparison Metrics")
println("-----------------------------------------------")
@printf("%-8s  %-14s  %-14s\n", "State", "Dot product", "L2 error")
println("-----------------------------------------------")
for (state, dot_val, l2_val) in metrics
    @printf("%-8s  %14.8f  %14.6e\n", state, dot_val, l2_val)
end
println("-----------------------------------------------")



wavefunction_report_2D(
    xint,
    yint,
    ms[end][1],
    ms[end][2],
    shape,
    E,
    psi,
    energy_analytical_2D,
    psi_analytical_2;
    nmax = 5,
    display_info = true,
    display_plot = true
)