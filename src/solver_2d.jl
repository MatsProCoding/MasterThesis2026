module Solver_2D

using SparseArrays
using LinearAlgebra

using ..Math
using ..Constants

# Make coordininate order for 2D grid, first x, then y
function coord_order(x,y)
    coord = vec([(xi, yi) for xi in x, yi in y])
    return coord
end


# Make 2D matrices for Kronecker product
function make_2D(A, B, m, n)
    Ix = spdiagm(0 => ones(Float64, m-1))
    Iy = spdiagm(0 => ones(Float64, n-1))
    x = kron(Iy, A)
    y = kron(B, Ix)
    return x,y
end

# Make mass matrix for position-dependent mass on half grid

function mass_coord_order(x_int, y_int, x_half, y_half)
    # Node i+-dx, j
    dx_pnts = coord_order(x_half, y_int)
    
    # Node i, j+- dy
    dy_pnts = coord_order(x_int, y_half)

    return dx_pnts, dy_pnts
end

# Make mass matrix for position-dependent mass on half grid
function build_mass(coord, mass)
    if mass isa Number
        M = (1.0 / Float64(mass)) * spdiagm(0 => ones(Float64, length(coord)))
        mode = 1
    elseif mass isa Function
        M = spdiagm(0 => Float64[1 / mass(x, y) for (x, y) in coord])
        mode = 2
    elseif mass isa Vector
        M = spdiagm(0 => Float64.(1.0 ./ mass))
        mode = 3
    else
        error("Mass must be a Number, Function, or Vector")
    end

    return M, mode
end



    

# Make potential matrix on grid points
function build_potential(coord, V)
    if V isa Number
        Z = Float64(V) .* ones(length(coord))
        println("Using constant potential: V* = ", V)
    elseif V isa Function
        Z = Float64[V(x, y) for (x, y) in coord]
    else
        error("Potential must be a Number or Function")
    end
    
    return spdiagm(0 => Z)
end

# Make Hamiltonian matrix for 1D problem with position-dependent mass and potential
function build_H2D(x_min, x_max, y_min, y_max; mass_x=1.0, mass_y=1.0, V=0.0, m = 200, n=200, display_info = false)
    # Create grid points
    x_int, y_int, x_half, y_half, dx, dy = make_grid(x_min, x_max, y_min, y_max, m, n)

    # Create derivation matrix matrix
    D_x = build_FD(m, dx)
    D_y = build_FD(n, dy)

    # Create combined 2D derivation matrices
    D_x, D_y = make_2D(D_x, D_y, m, n)

    # Create diagonal mass matrix for vertical and horisontal points
    m_x, m_y = mass_coord_order(x_int, y_int, x_half, y_half)
    M_x, mode = build_mass(m_x, mass_x)
    M_y, _ = build_mass(m_y, mass_y)

    # Built T, the kinetic part of the Hamiltonian
    T = -D_x' * M_x * D_x + -D_y' * M_y * D_y

    # Create potential diagonal matrix
    v_pnts = coord_order(x_int, y_int)
    Z = build_potential(v_pnts, V)

    # Combine T and V with units
    H = -Constants.hbar2_over_2me * T + Z

    if display_info
    mass_command = ["Using constant mass: m*",  "Using position-dependent mass from function on half grid.", "Using position-dependent mass from vector on half grid."]
        println(mass_command[mode])
        println("Hermitian: ", ishermitian(H),"\n")
    end
    return H
end


export build_H2D, mass_coord_order, coord_order, make_2D, build_mass, build_potential

end

# Kommentar. alle nummer må gjøres om til til float, som masse og potensial, hvis ikke sliter den med å gjøre om til hermitian
