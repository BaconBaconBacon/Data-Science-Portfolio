import scipy.constants as const

""" 
SI

\mu_0 = 1.25663706212(19)×10^{−6} \frac{N}{A^2} \sim \left[\frac{kg m}{s^2A^2}\right]

H\sim \frac{A}{m}

B \sim Tesla \sim \frac{kg}{s^2A}

 M \sim \frac{A}{M} 

m \sim \frac{J}{T}

CGS

H\sim Oerstad \rightarrow \frac{10^3}{4\pi}\frac{A}{m}

B \sim Gauss \rightarrow \frac{1}{10000} T

m \sim erg/G \rightarrow emu \rightarrow 10^3\frac{J}{T}

 M \sim \frac{erg}{G\cdot cm^3} =\frac{emu}{cm^3} \rightarrow 10^3\frac{A}{m}

"""

kb_SI = const.Boltzmann  # J / K -> m^2 kg / s^2 / K
kb_CGS = kb_SI * 10 ** (7)  # erg/ K   [ 10^7 erg = J]
mub_SI = const.value("Bohr magneton")  # J/T
mub_CGS = mub_SI * 10**3  # erg / G
mu0_SI = const.value("vacuum mag. permeability")  # N/A^2
mu0_CGS = 1
Na = 6.0221408e23  # atoms/mol


YbPdBi_molar_mass = 488.44  # g/mol
YbPdBi_density = 10.84  # g/cm^3, from material project
