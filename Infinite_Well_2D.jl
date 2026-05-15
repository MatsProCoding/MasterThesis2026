# ===== Initializing =====
if !isdefined(Main, :Project)
    include("../src/Project.jl")
end

using .Project.Constants
using .Project.Solver_2D
using .Project.Math
using .Project.Eval

# ===== External packages =====

using Arpack
using Plots
using CairoMakie
# ===== Parameters =====

m = 200
n = 200

L = 200.0
a = 100.0
b = a


x0 = -L
Lx = L 
y0 = -L
Ly = L



W0_x = -a
W1_x = a
W0_y = -b
W1_y = b

Lwell_x = W1_x - W0_x
Lwell_y = W1_y - W0_y

V0 = 1.0e9


xint, xhalf, dx = make_grid(x0, Lx, m)
yint, yhalf, dy = make_grid(y0, Ly, n)


function theme_article()
    merge(
        CairoMakie.theme_latexfonts(),
        CairoMakie.Theme(
            fontsize = 16,
            Axis = (
                xlabelsize = 18,
                ylabelsize = 18,
                titlesize = 18,
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
                labelsize = 16,
                framevisible = true,
            ),
            Lines = (linewidth = 2.2,),
            Scatter = (markersize = 8, strokewidth = 0.8, strokecolor = :black)
        )
    )
end

function V(x, y)
    edge_width = 1e-9 # veldig smal toleranse
    if abs(abs(x) - a) <= edge_width || abs(abs(y) - b) <= edge_width
        return V0   # "Middle ground" akkurat på grensen
    elseif abs(x) < a && abs(y) < b
        return 0.0
    else
        return V0
    end
end

### plot V for å sjekke at den ser riktig ut
V_plot = [V(x,y) for x in xint, y in yint]


with_theme(theme_article()) do
    fig = Figure(size=(500, 500))
    ax = Axis(fig[1, 1], title="Potential V(x,y)", xlabel="x", ylabel="y",
        xticks=([-a, 0, a], [L"-L_x", "0", L"L_x"]),
        yticks=([-a, 0, a], [L"-L_y", "0", L"L_y"]),
        aspect=DataAspect())
    hm = CairoMakie.heatmap!(ax, xint, yint, V_plot, colormap=["#000004", "#fcffa4"])
    Legend(fig[1, 2], [PolyElement(color="#000004"), PolyElement(color="#fcffa4")], ["0", "∞"], framevisible=false)
    display(fig)
    save("infinite_well_potential.pdf", fig)
end



function cell_avg_potential(xc, yc, dx, dy, V0, inside_polygon; ns=4)
    count_outside = 0
    total = ns^2

    for a in 1:ns
        for b in 1:ns
            xs = xc + ((a - 0.5)/ns - 0.5)*dx
            ys = yc + ((b - 0.5)/ns - 0.5)*dy
            if !inside_polygon(xs, ys)
                count_outside += 1
            end
        end
    end

    return V0 * count_outside / total
end


inside_shape(x, y) = (W0_x ≤ x ≤ W1_x && W0_y ≤ y ≤ W1_y)


V_smeared(x, y) = cell_avg_potential(x, y, dx, dy, V0, inside_shape)

V_hard(x, y) = inside_shape(x,y) ? 0.0 : V0


plots = [
    Plots.heatmap(xint, yint, (x,y) -> V_smeared(x,y), title="Smeared Potential", xlabel="x", ylabel="y"),
    Plots.heatmap(xint, yint, (x,y) -> V_hard(x,y), title="Hard Potential", xlabel="x", ylabel="y")
]

display(Plots.plot(plots[2], size=(1200,400)))




# ===== Analytical results =====

psi_analytical_2d_inf(x,y,nx,ny) =
    (W0_x ≤ x ≤ W1_x && W0_y ≤ y ≤ W1_y) ? sin(nx*pi*(x-W0_x)/Lwell_x)*sin(ny*pi*(y-W0_y)/Lwell_y) : 0.0

energy_analytical_2d_inf(nx,ny) =
    Constants.hbar2_over_2me/0.067 * pi^2 * ((nx/Lwell_x)^2 + (ny/Lwell_y)^2)


# ===== Comparison of analytical and numerical =====
using LinearAlgebra
using KrylovKit
#H = build_H2D(x0, Lx, y0, Ly; mass_x=1.0, mass_y=1.0, V=Vfun, m=m, n=n, display_info = true)


#=
time_start = time()
E, psi = eigs(H, nev=6, sigma= 0.0, which=:LM, maxiter=300_000)
println("Time taken for eigs sigma: ", time() - time_start, " seconds, Relative error: ", abs(E[1] - E_correct) / abs(E_correct))
time_start = time()
E, psi = eigs(H, nev=6, which=:SR, maxiter=300_000)
println("Time taken for eigs SR: ", time() - time_start, " seconds, Relative error: ", abs(E[1] - E_correct) / abs(E_correct))
time_start = time()
E, psi = eigs(Hermitian(H), nev=6, sigma= 0.0, which=:LM, maxiter=300_000)
println("Time taken for eigs hermitian sigma: ", time() - time_start, " seconds, Relative error: ", abs(E[1] - E_correct) / abs(E_correct))
time_start = time()
E, psi = eigs(Hermitian(H), nev=6, which=:SR, maxiter=300_000)
println("Time taken for eigs hermitian SR: ", time() - time_start, " seconds, Relative error: ", abs(E[1] - E_correct) / abs(E_correct))
=#





function builder(N)
    m, n = N
    xint, xhalf, dx = make_grid(x0, Lx, m)
    yint, yhalf, dy = make_grid(y0, Ly, n)
    coord_xhalf = coord_order(xhalf, yint)
    coord_yhalf = coord_order(xint, yhalf)
    mx = 0.067
    my = 0.067
    
    Vf = (x,y) -> cell_avg_potential(x, y, dx, dy, V0, inside_shape)
    H = build_H2D(x0, Lx, y0, Ly; mass_x=mx, mass_y=my, V=Vf, m=m, n=n, display_info=false)
    E, psi = eigs(Hermitian(H), nev=10, sigma=0.0, which=:LM, maxiter=300_000)

    return (
        E = E,
        psi = psi,
        x = xint,
        y = yint
    )
end


ms_hit = [
    (32,32),
    (64,64),
    (128,128),
    (256,256),
    (512,512)
]

ms_miss = [
    (30,30),
    (62,62),
    (126,126),
    (254,254),
    (510,510)
]

ms_mixed = [
    (31,31),   # miss
    (32,32),   # hit
    (63,63),   # miss
    (64,64),   # hit
    (127,127), # miss
    (128,128), # hit
    (255,255), # miss
    (256,256), # hit
]

E, psi, xint, yint = convergence_report(ms_hit, builder, energy_analytical_2d_inf; nmax=6, type="infinite_2D",save_plot=true)

energy_report(E, energy_analytical_2d_inf; nmax=6, display_info = true) 
function Vplot(x)
    eps = 0.05
    if (W0 - eps <= x <= W0 + eps) || (W1 - eps <= x <= W1 + eps)
        return 25
    else
        return 0.0
    end
end
E_inf=[]
for i in E
    push!(E_inf, i)
end

energy = energy_report(E, energy_analytical_2d_inf; nmax=6, display_info = true)
wave = wavefunction_report_2D(xint, yint, ms_hit[end][1], ms_hit[end][1],  inside_shape, E, psi,energy_analytical_2d_inf, psi_analytical_2d_inf; nmax=7, display_info = true, display_plot = true, plot_quantity =:real)


plot_heat = psi[:,2]
plot_heat_m = reshape(plot_heat, length(xint), length(yint))
plot_heat2 = psi[:,3]
plot_heat2_m = reshape(plot_heat2, length(xint), length(yint))
CairoMakie.heatmap(xint, yint, plot_heat_m)
CairoMakie.heatmap(xint, yint, plot_heat2_m)

# psi_ordered_m = reshape(psi_ordered, length(xint), length(yint))
# psi_ordered2_m = reshape(psi_ordered2, length(xint), length(yint))

psi_12 = vec(plot_heat_m - plot_heat2_m')
psi_21 = vec(plot_heat_m + plot_heat2_m')
psi_12 = normalize_wavefunction(psi_12, dx^2)
psi_21 = normalize_wavefunction(psi_21, dx^2)
psi_12_m = reshape(psi_12, length(xint), length(yint))
psi_21_m = reshape(psi_21, length(xint), length(yint))


ana_12 = [psi_analytical_2d_inf(x,y,1,2) for x in xint, y in yint]
ana_21 = [psi_analytical_2d_inf(x,y,2,1) for x in xint, y in yint]
ana_12 = normalize_wavefunction(vec(ana_12), dx^2)
ana_21 = normalize_wavefunction(vec(ana_21), dx^2)
ana_12_m = reshape(ana_12, length(xint), length(yint))
ana_21_m = reshape(ana_21, length(xint), length(yint))
psi_12_m = psi_12_m .* -1
psi_21_m = psi_21_m .* -1
CairoMakie.heatmap(xint, yint, psi_12_m)
CairoMakie.heatmap(xint, yint, ana_12_m)
CairoMakie.heatmap(xint, yint, psi_21_m)
CairoMakie.heatmap(xint, yint, ana_21_m)
dx = xint[2] - xint[1]

l2_error_12 = sqrt(sum(abs2, vec(psi_12_m) .- vec(ana_12_m)) * dx^2)
l2_error_21 = sqrt(sum(abs2, vec(psi_21_m) .- vec(ana_21_m)) * dx^2)


sqrt(sum(abs2, vec(psi_num) .- vec(psi_an)) * dA)

n=4
psi_test_something = psi[:,n]
psi_test_something = reshape(psi[:,n], ms_hit[end][1]-1, ms_hit[end][2]-1)

fig= Figure()
ax = Axis(fig[1, 1], title="Test plot", xlabel="x")
#CairoMakie.heatmap!(ax, xint, yint, psi_test_something, colormap=:inferno)
CairoMakie.contourf!(ax, xint, yint, psi_test_something, colormap=:inferno)
display(fig)


function theme_article()
    merge(
        CairoMakie.theme_latexfonts(),
        CairoMakie.Theme(
            fontsize = 16,
            Axis = (
                xlabelsize = 18,
                ylabelsize = 18,
                titlesize = 18,
                xticklabelsize = 15,
                yticklabelsize = 15,
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
E[2]
E[2+1]
im_list_inf = []

heatplots = [[1,1], [1,2], [2,2]]
states = [1,2,4]
for (i, state) in enumerate(states)
    dx = xint[2] - xint[1]
    dA = dx^2
    k, l = heatplots[i]
    psi_num = real(psi[:, state])
    psi_num = normalize_wavefunction(psi_num, dA)
    psi_num = reshape(psi_num, length(xint), length(yint))
    
    psi_an = psi_analytical_2d_inf.(xint, yint', k, l)
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
        psi_an = psi_analytical_2d_inf.(xint, yint', k, l)
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
    
    push!(im_list_inf, psi_num)

end

for i in im_list_inf
    display(i)
end







# enkelt test av kode med og uten hermitian wraper avlsørte dramatisk kortere tid for herm
# Sammenligne når ms treffer nøyaktig på punktet og når det ikke gjør det, og se hvordan det påvirker konvergensen error og p. 
# Plot grafer for når ms treffer, ikke treff og blandet å se at den fortsatt konergerer til en verdi fo gitt m og n
# Vis at degen plot blir rart om du ikke har med degen!
# For høyere brønn enn bredere må eval wave ha ikke transponert numerisk psi, men analytisk må. Dette er fordi num er bygget opp med x så y, mens analytisk er omvendt i heatmap
# For Lx != Ly bryter metoden sammen for lik brønn dims, dette er kun for infinite potential mistenker jeg, men dersom bredden på brønn er lik bredden på grid kan den være asymetrisk