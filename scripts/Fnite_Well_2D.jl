# ===== Initializing =====
if !isdefined(Main, :Project)
    include("../src/Project.jl")
end

using .Project.Constants
using .Project.Solver_2D
using .Project.Math
using .Project.Eval

using Arpack
using LinearAlgebra
using CairoMakie

# ===== Parameters =====

m = 200
n = 200

L = 200.0
a = 100.0
x0 = -L
Lx = L
y0 = -L
Ly = L

mstar = 0.067

z0 = a * sqrt(V0 / (Constants.hbar2_over_2me * 0.067))
println("z0 = $z0")  # skal være mellom π og 2π for 2 bundne tilstander
mx = mstar
my = mstar

V0 = 0.25

# Velg potensialtype her
use_smoothed_potential = true
nsmooth = 4   # antall samplepunkter per celle i smoothing

# ===== Grid =====

xint, xhalf, dx = make_grid(x0, Lx, m)
yint, yhalf, dy = make_grid(y0, Ly, n)

# ===== Hard 1D potential =====

Vx_hard(x) = abs(x) <= a ? 0.0 : V0
Vy_hard(y) = abs(y) <= a ? 0.0 : V0

# ===== Smoothed 1D potential =====
# Celle-gjennomsnitt i 1D over intervallet [xc - d/2, xc + d/2]

function cell_avg_potential_1D(xc, d, Vfun; ns=4)
    s = 0.0
    for i in 1:ns
        xs = xc + ((i - 0.5)/ns - 0.5) * d
        s += Vfun(xs)
    end
    return s / ns
end

Vx_smeared(x, dx; ns=4) = cell_avg_potential_1D(x, dx, Vx_hard; ns=ns)
Vy_smeared(y, dy; ns=4) = cell_avg_potential_1D(y, dy, Vy_hard; ns=ns)

# ===== Valgt potensial =====

V_hard(x, y) = Vx_hard(x) + Vy_hard(y)
V_smeared(x, y, dx, dy; ns=4) = Vx_smeared(x, dx; ns=ns) + Vy_smeared(y, dy; ns=ns)



# Denne brukes bare til visning
shape(x, y) = (abs(x) <= a && abs(y) <= a)

# ===== 1D semi-analytical finite well =====

c = Constants.hbar2_over_2me/mstar
cot(x) = cos(x) / sin(x)

k(E) = sqrt(E / c)
kappa(E) = sqrt((V0 - E) / c)

even(E) = k(E) * tan(k(E) * a) - kappa(E)
odd(E)  = -k(E) * cot(k(E) * a) - kappa(E)

E_odd  = find_roots(odd, 1e-9, V0)
E_even = find_roots(even, 1e-9, V0)

E_1D = sort([E_odd; E_even])

println("\n--- 1D bound states ---")
for (i, E) in enumerate(E_1D)
    println(i, ": ", E)
end

# ===== 2D analytical energy =====

function energy_analytical_finite_2D(nx, ny)
    if nx > length(E_1D) || ny > length(E_1D)
        return NaN
    end
    return E_1D[nx] + E_1D[ny]
end

# ===== 1D analytical wavefunction =====

function psi1D(x, n)
    E = E_1D[n]
    k_val = k(E)
    κ = kappa(E)

    if isodd(n)
        if abs(x) <= a
            return cos(k_val * x)
        else
            return cos(k_val * a) * exp(-κ * (abs(x) - a))
        end
    else
        if abs(x) <= a
            return sin(k_val * x)
        else
            return sign(x) * sin(k_val * a) * exp(-κ * (abs(x) - a))
        end
    end
end

# ===== 2D analytical wavefunction =====

function psi_analytical_finite_2D(x, y, nx, ny)
    if nx > length(E_1D) || ny > length(E_1D)
        return 0.0
    end
    return psi1D(x, nx) * psi1D(y, ny)
end

# ===== Numeric solver =====

function builder(N; use_smoothed=false, ns=4)
    m, n = N

    xint, xhalf, dx = make_grid(x0, Lx, m)
    yint, yhalf, dy = make_grid(y0, Ly, n)

    Vfun = use_smoothed ?
        ((x, y) -> V_smeared(x, y, dx, dy; ns=ns)) :
        V_hard
    
    H = build_H2D(
        x0, Lx, y0, Ly;
        mass_x = mx,
        mass_y = my,
        V = Vfun,
        m = m,
        n = n,
        display_info = false
    )

    E, psi = eigs(
        Hermitian(H),
        nev = 10,
        sigma = 0.0,
        which = :LM,
        maxiter = 300_000
    )

    return (
        E = E,
        psi = psi,
        x = xint,
        y = yint
    )
end

# ===== Convergence study =====

ms = [
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256)]

nmax = min(8, length(E_1D)^2)

res = convergence_report(
    ms,
    N -> builder(N; use_smoothed=use_smoothed_potential, ns=nsmooth),
    energy_analytical_finite_2D;
    nmax = 6,
    display_info = true,
    display_plot = true,
    type = "finite_2D_new",
    save_plot = true
)

E = res.E
psi = res.psi
xint = res.x
yint = res.y
length(xint), length(yint)

E_fin = []
for i in E
    push!(E_fin, i)
end
# ===== Energy comparison =====

energy = energy_report(
    E,
    energy_analytical_finite_2D;
    nmax = nmax,
    display_info = true
)

# ===== Wavefunction comparison =====

wave = wavefunction_report_2D(
    xint,
    yint,
    ms[end][1],
    ms[end][2],
    shape,
    E,
    psi,
    energy_analytical_finite_2D,
    psi_analytical_finite_2D;
    nmax = min(6, nmax),
    display_info = true,
    display_plot = true,
    plot_quantity = :real
)


function theme_article()
        merge(
            CairoMakie.theme_latexfonts(),
            CairoMakie.Theme(
                fontsize = 16,
                Axis = (
                    xlabelsize = 24,
                    ylabelsize = 24,
                    titlesize = 24,
                    xticklabelsize = 20,
                    yticklabelsize = 20,
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
                    labelsize = 20,
                    framevisible = true,
                ),
                Lines = (linewidth = 2.2,),
                Scatter = (markersize = 8, strokewidth = 0.8, strokecolor = :black)
            )
        )
end
Vhard_vals = [V_hard(x, y) for y in yint, x in xint]
lxint = length(xint)

Vsmeared_vals = [V_smeared(x, y, dx, dy; ns=nsmooth) for y in yint, x in xint]
with_theme(theme_article()) do
    fig = Figure(size=(750,750))
    ax = Axis(fig[1, 1],
        xlabel="x", ylabel="y",
        title="V(x,y) = Vₓ(x) + Vᵧ(y)",
        xticks=([-a, 0, a], ["-Lx", "0", "Lx"]),
        yticks=([-a, 0, a], ["-Ly", "0", "Ly"]),
        aspect=DataAspect())
    hm = CairoMakie.heatmap!(ax, xint, yint, Vhard_vals, colormap=["#000004", "#bb3755", "#fcffa4"])
    Legend(fig[1, 2], [PolyElement(color="#000004"), PolyElement(color="#bb3755"), PolyElement(color="#fcffa4")], ["0", L"$V_0$", L"$2V_0$"], framevisible=false)
    display(fig)
    save("finite_well_potential.pdf", fig)
end
im_list_fin = []

heatplots = [[1,1], [2,1], [2,2]]
states = [1,2,4]
for (i, state) in enumerate(states)
    dx = xint[2] - xint[1]
    dA = dx^2
    k, l = heatplots[i]
    psi_num = real(psi[:, state])
    psi_num = normalize_wavefunction(psi_num, dA)
    psi_num = reshape(psi_num, length(xint), length(yint))
    
    psi_an = psi_analytical_finite_2D.(xint, yint', k, l)
    psi_an = normalize_wavefunction(vec(psi_an), dA)
    psi_an = reshape(psi_an, length(xint), length(yint))'

    degen=false
    if abs.(E[state+1] - E[state]) < 1e-6
        degen = true
    elseif state > 1 && abs.(E[state-1] - E[state]) < 1e-6
        degen = true
    end

    if degen
        psi0 = reshape(psi[:,state], length(xint), length(xint))
        psi1 = reshape(psi[:,state +1 ], length(xint), length(xint))

        add = vec(psi0 + psi1')
        sub = vec(psi0 - psi1')

        psi_num = normalize_wavefunction(add, dA)
        psi_num = reshape(psi_num, length(xint), length(yint))
    end

    overlap = (dot(vec(conj.(psi_num)), vec(psi_an)) * dA)
    if abs(overlap) < 1e-3
        k, l = l, k
        psi_an = psi_analytical_finite_2D.(xint, yint', k, l)
        psi_an = normalize_wavefunction(vec(psi_an), dA)
        psi_an = reshape(psi_an, length(xint), length(yint))'
        overlap = (dot(vec(psi_num), vec(psi_an)) * dA)
    end
    if real(overlap) < 0
        psi_num .*= -1
    end

    
    titel = "Numerical ($k,$l), E = $(round(E[state], digits=4)), psi overlap = $(round(overlap, digits=4))"
    p1 = CairoMakie.Figure(size=(430,400))
    ax_p1 = CairoMakie.Axis(p1[1, 1], xlabel = "x", ylabel = "y", title = titel)
    hm_an = CairoMakie.heatmap!(ax_p1, xint, yint, psi_num, colormap=:inferno)
    cool_bar = CairoMakie.Colorbar(p1[1, 2], hm_an, width = 15)
    CairoMakie.lines!(ax_p1, [-100, 100, 100, -100, -100], [-100, -100, 100, 100, -100],
    color=:white, linewidth=2)
    push!(im_list_fin, psi_num)

end
