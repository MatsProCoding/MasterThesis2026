# ===== Initializing =====
if !isdefined(Main, :Project)
    include("../src/Project.jl")
end

using .Project.Constants
using .Project.Solver_1D
using .Project.Math
using .Project.Eval

# ===== External packages =====

using Arpack
using Plots
# ===== Parameters =====

L = 400.0
a = 400.0

x0 = -L
Lx = L


W0 = -a
W1 = a

Lwell = W1 - W0
mstar = 1
V0 = 1.0e9

Vs(x) = abs(x) < a ? 0.0 : V0



# ===== Analytical results =====

psi_analytical(x, n) =
    (W0 <= x <= W1) ? sin(n * pi * (x - W0) / Lwell) : 0.0

energy_analytical(n) =
    Constants.hbar2_over_2me/mstar * pi^2 * (n / Lwell)^2


# ===== Comparison of analytical and numerical =====


ms = [32,64,128, 256, 512,1024]
using LinearAlgebra

function builder(m)
    xint, xhalf, dx = make_grid(x0, Lx, m)
    H = build_H1D(x0, Lx; mass=mstar, V=Vs, m=m, display_info = false)
    E, psi = eigs(Hermitian(H), nev=6, which=:SR, maxiter=300_000)
    return (
        E = E,
        psi = psi,
        x = xint,
        y = nothing
    )
end

res = convergence_report(ms, builder, energy_analytical; nmax=5, type="Infinite_Well_1D_2", save_plot=true)
E = res.E
psi = res.psi
xint = res.x



function Vplot(x)
    eps = 0.05
    if (W0 - eps <= x <= W0 + eps) || (W1 - eps <= x <= W1 + eps)
        return 25
    else
        return 0.0
    end
end

wave_plot_1D(xint, Vplot, energy_analytical,E, psi, psi_analytical;nmax=5, scaling=0.05 , name="Infinite_Well_1D_2", save_plot=true)

energy = energy_report(E, energy_analytical; nmax=5, display_info = true)

wave = wavefunction_report(xint, Vplot, E, psi, psi_analytical; nmax=5, display_info = true, display_plot = true)

# Må compare absolutt rmse, ikke relativ for p verdi

wave_plot_1D(xint, Vs, energy_analytical,E, psi, psi_analytical;nmax=5, scaling=2.3 , name="Infinite_Well_1D", save_plot=true)
