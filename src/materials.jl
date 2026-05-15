module Materials

const VUR = Dict(
    "GaAs" => Dict(
        :Eg      => 1.519,
        :me      => 0.067,
        :Ep      => 28.8,
        :F       => -1.94,
        :so      => 0.341,
        :gamma1  => 6.98,
        :gamma2  => 2.06,
        :gamma3  => 2.93,
    ),

    "AlAs" => Dict(
        :Eg      => 3.099,
        :me      => 0.15,
        :Ep      => 21.1,
        :F       => -0.48,
        :so      => 0.280,
        :gamma1  => 3.76,
        :gamma2  => 0.82,
        :gamma3  => 1.42,
    ),

    "InAs" => Dict(
        :Eg      => 0.417,
        :me      => 0.026,
        :Ep      => 21.5,
        :F       => -2.90,
        :so      => 0.390,
        :gamma1  => 20.0,
        :gamma2  => 8.5,
        :gamma3  => 9.2,
    )
)

Eg_bowing_AlGaAs(x) = -0.127 + 1.310 * x

lerp(a, b, x) = (1 - x) * a + x * b

"""
    algaas_params(xAl; bowing=true)

Beregn Vurgaftman-lignende parametre for Al_xGa_{1-x}As:
Eg (med bowing hvis `bowing=true`), me, Ep, F, so, gamma1, gamma2, gamma3.
"""
function algaas_params(xAl; bowing::Bool=true)
    x = xAl
    g = 1 - x

    Eg_ga = VUR["GaAs"][:Eg]
    Eg_al = VUR["AlAs"][:Eg]

    Eg_alloy = if bowing
        bE = Eg_bowing_AlGaAs(x)
        g * Eg_ga + x * Eg_al - x * g * bE
    else
        g * Eg_ga + x * Eg_al
    end

    me_alloy     = lerp(VUR["GaAs"][:me],     VUR["AlAs"][:me],     x)
    Ep_alloy     = lerp(VUR["GaAs"][:Ep],     VUR["AlAs"][:Ep],     x)
    F_alloy      = lerp(VUR["GaAs"][:F],      VUR["AlAs"][:F],      x)
    so_alloy     = lerp(VUR["GaAs"][:so],     VUR["AlAs"][:so],     x)
    gamma1_alloy = lerp(VUR["GaAs"][:gamma1], VUR["AlAs"][:gamma1], x)
    gamma2_alloy = lerp(VUR["GaAs"][:gamma2], VUR["AlAs"][:gamma2], x)
    gamma3_alloy = lerp(VUR["GaAs"][:gamma3], VUR["AlAs"][:gamma3], x)

    return (
        Eg      = Eg_alloy,
        me      = me_alloy,
        Ep      = Ep_alloy,
        F       = F_alloy,
        so      = so_alloy,
        gamma1  = gamma1_alloy,
        gamma2  = gamma2_alloy,
        gamma3  = gamma3_alloy,
    )
end



export VUR, Eg_bowing_AlGaAs, algaas_params

end # module
