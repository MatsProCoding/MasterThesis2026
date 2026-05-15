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
lw = 800
L0 = -lw
Lx = lw
mstar=0.067
omega = 0.010/sqrt(2 * Constants.hbar2_over_2me / mstar)

Vs(x) = 0.5 * omega^2 * x^2 



# ===== Analytical results =====


function energy_analytical(n)
    return sqrt(2 * Constants.hbar2_over_2me/mstar) * omega * (n - 0.5)
end


function psi_analytical(x, n)

    beta = sqrt(omega / sqrt(2 * Constants.hbar2_over_2me/mstar))

    ξ = beta * x

    # Hermite-polynom
    H = hermiteH(n-1, ξ)

    # normalisering

    return  H * exp(-ξ^2 / 2)

end

# ===== Comparison of analytical and numerical =====


ms = [32, 64, 128, 256, 512, 1024, 2048]

function builder(m)
    xint, xhalf, dx = make_grid(L0, Lx, m)
    H = build_H1D(L0, Lx; mass=mstar, V=Vs, m=m, display_info = false)
    E, psi = eigs(H, nev=7, which=:SR, maxiter=300_000)
        return (
        E = E,
        psi = psi,
        x = xint,
        y = nothing
    )
end

res = convergence_report(ms, builder, energy_analytical; nmax=5, type="HO")
E = res.E
psi = res.psi
xint = res.x
E_ana(n) = energy_analytical(n)
energy = energy_report(E, E_ana; nmax=7, display_info = true)
#wave = wavefunction_report(xint, Vs, E, psi, psi_analytical; nmax=5, display_info = true, display_plot = true)


wave_plot_1D_prob(xint, Vs, E_ana, E, psi, psi_analytical, nmax=5)
wave_plot_1D_prob_diffcollor(xint, Vs, E_ana, E, psi, psi_analytical, nmax=5)



