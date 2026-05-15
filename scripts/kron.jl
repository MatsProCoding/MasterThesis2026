if !isdefined(Main, :Project)
    include("src/Project.jl")
end

using .Project.Math
using .Project.Solver_2D
using LinearAlgebra
using SparseArrays


# Some grid defined on unequal intervals and lengths for testing 
x0 = 0.0
y0 = 0.0
Lx = 5.0
Ly = 10.0
m = 5
n = 10
x_int, y_int, x_half, y_half, dx, dy = make_grid(x0, Lx, y0, Ly, m, n)


# First-order forward difference matrices
D_x = build_FD(m, dx)
D_y = build_FD(n, dy)

Ix = spdiagm(0 => ones(Float64, m-1))
Iy = spdiagm(0 => ones(Float64, n-1))
x = kron(Iy, D_x)
y = kron(D_y, Ix)

F = -x' *x -y' *y

# Second-order central difference matrices
S_x = -D_x' * D_x
S_y = -D_y' * D_y

S = kron(Iy, S_x) + kron(S_y, Ix)

println("Is using first-order forward difference matrices equal to second-order central difference matrices? ", F == S )
