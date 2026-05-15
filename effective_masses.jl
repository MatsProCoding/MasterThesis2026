module Effective_Masses

using LinearAlgebra
using ..Constants

# H(k)
function bands(k, state, H)
    vals = sort(real(eigvals(Hermitian(H(k)))))
    return vals[state]
end

# H(x,k)
function bands(x, k, state, H)
    vals = sort(real(eigvals(Hermitian(H(x, k)))))
    return vals[state]
end

# H(k)
function effective_mass(dk, H; nbands=2)
    masses = Float64[]
    K_values = (-2:2) .* dk

    for band in 1:nbands
        E = [bands(k, band, H) for k in K_values]
        d2E_dk2 = (-E[5] + 16E[4] - 30E[3] + 16E[2] - E[1]) / (12 * dk^2)
        m_eff = Constants.hbar2_over_me / d2E_dk2
        push!(masses, m_eff)
    end

    return masses
end

# H(x,k)
function effective_mass(x, dk, H; nbands=2)
    masses = [Float64[] for _ in 1:nbands]
    K_values = (-2:2) .* dk
    for x_vals in x 
        for band in 1:nbands
            E = [bands(x_vals, k, band, H) for k in K_values]
            d2E_dk2 = (-E[5] + 16E[4] - 30E[3] + 16E[2] - E[1]) / (12 * dk^2)
            m_eff = Constants.hbar2_over_me / d2E_dk2
            push!(masses[band], m_eff)
        end
    end
    return masses
end

export effective_mass, bands

end