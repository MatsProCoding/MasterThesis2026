module Constants

# === Fundamental constants (eV·Å) ===

const hbarc = 1973.269804           # ħ·c  [eV·Å]
const me_c2 = 0.510998910e6           # electron rest energy mₑc² [eV]
const alpha_fine = 1 / 137.035999177  # fine-structure constant
const kB = 8.617333262e-5 # eV⋅K−1[1]

# === Derived constants ===

# ħ² / mₑ   (useful for Schrödinger, k·p, transport operators)
const hbar2_over_me  = hbarc^2 / me_c2      # [eV·Å²]

# ħ² / 2mₑ  (standard kinetic prefactor)
const hbar2_over_2me = hbar2_over_me / 2    # [eV·Å²]

const e2_over_eps0 = 4 * pi * hbarc * alpha_fine 

const kBT_room = kB * 298.15 # eV⋅K−1[1]


export hbarc, me_c2, alpha_fine
export hbar2_over_me, hbar2_over_2me, e2_over_eps0
export kB, kBT_room

end # module
