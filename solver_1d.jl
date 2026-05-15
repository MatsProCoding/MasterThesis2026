module Solver_1D

using SparseArrays
using LinearAlgebra

using ..Math
using ..Constants

# Make mass matrix for position-dependent mass on half grid

function build_mass(x, mass)
    if mass isa Number
        M = (1.0 / Float64(mass)) * spdiagm(0 => ones(Float64, length(x)))
        mode = 1
    elseif mass isa Function
        M = spdiagm(0 => Float64[1 / mass(xi) for xi in x])
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
function build_potential(x, V)
    if V isa Number
        Z = Float64(V) .* ones(length(x))
        println("Using constant potential: V* = ", V)
    elseif V isa Function
        Z = Float64[V(xi) for xi in x]
    else
        error("Potential must be a Number or Function")
    end
    
    return spdiagm(0 => Z)
end


# Make Hamiltonian matrix for 1D problem with position-dependent mass and potential
function build_H1D(x_min, x_max; mass=1.0, V=0.0, m = 200, display_info = false)
    # Create grid points
    x_int, x_half, dx = make_grid(x_min, x_max, m)

    # Create derivation matrix matrix
    D = build_FD(m, dx)
    
    # Create diagonal mass matrix
    M, mode = build_mass(x_half, mass)    

    T = - D' * M * D 

    # Create potential diagonal matrix
    Z = build_potential(x_int, V)

    # Combine T and V with units
    H = - Constants.hbar2_over_2me * T + Z

    if display_info
    mass_command = ["Using constant mass: m*",  "Using position-dependent mass from function on half grid.", "Using position-dependent mass from vector on half grid."]
        println(mass_command[mode])
        println("Hermitian: ", ishermitian(H),"\n")
    end
    return H
end


export build_H1D, build_mass, build_potential

end

# Kommentar. alle nummer må gjøres om til til float, som masse og potensial, hvis ikke sliter den med å gjøre om til hermitian