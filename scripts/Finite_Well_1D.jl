

#=
function unique_roots(roots; tol=1e-6)
    if isempty(roots)
        return Float64[]
    end

    roots = sort(roots)
    out = [roots[1]]

    for r in roots[2:end]
        if abs(r - out[end]) > tol
            push!(out, r)
        end
    end

    return out
end

function bisect_root(f, a, b; tol=1e-12, maxit=200)
    fa = f(a)
    fb = f(b)

    if !(isfinite(fa) && isfinite(fb)) || fa * fb > 0
        return nothing
    end

    for _ in 1:maxit
        mid = (a + b) / 2
        fm = f(mid)

        if !isfinite(fm)
            return nothing
        end

        if abs(fm) < tol || abs(b - a) < tol
            return mid
        end

        if fa * fm < 0
            b = mid
            fb = fm
        else
            a = mid
            fa = fm
        end
    end

    return (a + b) / 2
end

function find_roots_scan(f, xmin, xmax; nscan=20000, tol=1e-12, root_tol=1e-8,
                         pole_cutoff=1e6, max_roots=100)

    xs = range(xmin, xmax, length=nscan)
    roots = Float64[]

    for i in 1:length(xs)-1
        x1, x2 = xs[i], xs[i+1]
        y1, y2 = f(x1), f(x2)

        # skip intervals with non-finite values
        if !(isfinite(y1) && isfinite(y2))
            continue
        end

        # skip likely poles / asymptotes
        if abs(y1) > pole_cutoff || abs(y2) > pole_cutoff
            continue
        end

        # root must be bracketed
        if y1 * y2 < 0
            root = bisect_root(f, x1, x2; tol=tol)

            if root !== nothing
                fr = f(root)

                # accept only actual roots, not poles
                if isfinite(fr) && abs(fr) < root_tol
                    push!(roots, root)

                    if length(roots) >= max_roots
                        break
                    end
                end
            end
        end
    end

    return unique_roots(roots)
end
d(E) = E - cot(E)

a = find_roots_scan(d, 0, 10)

using Pkg
Pkg.add("Roots")

using Roots

cot(x) = cos(x)/sin(x)
d(E) = E - cot(E)


=#







# ===== Initializing =====
if !isdefined(Main, :Project)
    include("../src/Project.jl")
end

using .Project.Constants
using .Project.Solver_1D
using .Project.Math
using .Project.Eval


using Arpack

# ===== External packages =====

L = 800.0
a = 10.0

x0 = -L
Lx = L

W0 = -a
W1 = a

V0 = 0.22
mstar = 0.067

function V(x)
    edge_width = 1e-9 # veldig smal toleranse
    if abs(abs(x) - a) < edge_width
        return V0/2   # "Middle ground" akkurat på grensen
    elseif abs(x) < a
        return 0.0
    else
        return V0
    end
end


# ===== Analytical results =====

k(E) = sqrt(E/(Constants.hbar2_over_2me/mstar))
kappa(E) = sqrt((V0 - E)/(Constants.hbar2_over_2me/mstar))

even(E) = k(E) * tan(k(E) * a) - kappa(E)
odd(E) = -k(E) * cot(k(E) * a) - kappa(E)


E_odd = find_roots(odd, 1e-9, V0)
E_even = find_roots(even, 1e-9, V0)

E_an = sort([E_odd; E_even])

E_ana(n) = E_an[n]

function psi_analytical(x, n)

    E = E_ana(n)

    k_val = k(E)
    κ = kappa(E)

    # parity
    if isodd(n)

        if abs(x) <= a
            ψ = cos(k_val * x)
        else
            ψ = cos(k_val * a) * exp(-κ*(abs(x)-a))
        end

    else

        if abs(x) <= a
            ψ = sin(k_val * x)
        else
            ψ = sign(x) * sin(k_val * a) * exp(-κ*(abs(x)-a))
        end

    end

    return ψ
end



ms = [32, 64, 128, 256, 512, 1024, 2048]
using LinearAlgebra
function builder(m)
    xint, xhalf, dx = make_grid(x0, Lx, m)
    H = build_H1D(x0, Lx; mass=mstar, V=V, m=m, display_info = false)
    E, psi = eigs(Hermitian(H), nev=100, which=:SR, maxiter=300_000)
    return (
        E = E,
        psi = psi,
        x = xint,
        y = nothing
    )
end


CairoMakie.plot(xinr, abs2.(ggggg[:, 1]).+eeee[1])

CairoMakie.plot(xinr, abs2.(ggggg[:,5]).+eeee[5])
CairoMakie.plot(xinr, abs2.(ggggg[:,6]).+eeee[6])
CairoMakie.plot!(xinr, abs2.(ggggg[:,7]).+eeee[7])
display(CairoMakie.current_figure())

wave_plot_1D(xinr, V, E_ana, eeee, ggggg, psi_analytical; nmax=3, scaling=500, name="Finite_Well_1D_new_unbound", save_plot=true)


res = convergence_report(ms, builder, E_ana; nmax=6, type="Finite_Well_1D_cairo_textbig_5state", save_plot=false)
E = res.E
psi = res.psi
xint = res.x

 

energy = energy_report(E, E_ana; nmax=5, display_info = true)

wave = wavefunction_report(xint, V, E, psi, psi_analytical; nmax=5, display_info = true, display_plot = true, x_window = (-10, 10))



wave_plot_1D(xint, V, E_ana,E, psi, psi_analytical; nmax=6, scaling=2.5, name="Finite_Well_1D_new", save_plot=true)

# Gjør søk på metoder for å finne røtter. Laget en egen versjon, men roots er mer robust og man kan velge metode, finn ut hvilken
# Hard vegg fra f.eks. av Killingbeck eller Simos) 
# Bruk bare 4 laveste nivåer, da nivåene blir mer og mer ustabile
