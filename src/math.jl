module Math

using SparseArrays
using Roots

# Make grid points for interior grid and half grid
function make_grid(x_min, x_max, m)
    dx = (x_max - x_min) / m
    x = x_min:dx:x_max
    x_int = x[2:end-1]
    x_half = x_min + dx/2 : dx : x_max - dx/2
    return x_int, x_half, dx
end

function make_grid(x_min, x_max, y_min, y_max, m, n)
    dx = (x_max - x_min) / m
    dy = (y_max - y_min) / n

    x = x_min:dx:x_max
    y = y_min:dy:y_max

    x_int = x[2:end-1]
    y_int = y[2:end-1]

    x_half = x_min + dx/2 : dx : x_max - dx/2
    y_half = y_min + dy/2 : dy : y_max - dy/2

    return x_int, y_int, x_half, y_half, dx, dy
end

# Forward difference matrix for first derivative
function build_FD(m, dm)
    T =  spdiagm(m, m-1,
        0 => 1.0*ones(Float64, m-1)/dm,
        -1 => -1.0*ones(Float64, m-1)/dm)
    return T
end


# Central difference matrix for first derivative
function build_SD(m, dm)
    matrix =  spdiagm(
        1 => ones(Float64, m-2),
        -1 => -ones(Float64, m-2))
    c = 2*dm
    return matrix / c 
end

# Find roots for a function
function find_roots(f, x0, x1; N=5000, tol=1e-9)
    xs = range(x0, x1, length=N)
    roots = Float64[]

    for i in 1:length(xs)-1
        a, b = xs[i], xs[i+1]
        fa, fb = f(a), f(b)

        # hopp over intervaller nær singulariteter
        if !isfinite(fa) || !isfinite(fb)
            continue
        end

        # eksakt null på gridpunkt
        if abs(fa) < tol
            if all(abs(a - r) > 1e-6 for r in roots)
                push!(roots, a)
            end
        end

        # fortegnsskifte => bracketed root
        if sign(fa) != sign(fb)
            try
                r = find_zero(f, (a, b), Bisection())
                fr = f(r)

                if isfinite(fr) && abs(fr) < tol
                    if all(abs(r - rr) > 1e-6 for rr in roots)
                        push!(roots, r)
                    end
                end
            catch
            end
        end
    end

    sort!(roots)
    return roots
end

# Hermitian polynomial

function hermiteH(n, x)

    if n == 0
        return 1.0
    elseif n == 1
        return 2*x
    end

    Hnm2 = 1.0
    Hnm1 = 2*x

    for k in 2:n
        Hn = 2*x*Hnm1 - 2*(k-1)*Hnm2
        Hnm2 = Hnm1
        Hnm1 = Hn
    end

    return Hnm1
end


function normalize_wavefunction(psi, dA)
    nrm = sqrt(sum(abs2, psi) * dA)
    return psi ./ nrm
end


export build_FD, build_SD, make_grid, find_roots, hermiteH, normalize_wavefunction

end
