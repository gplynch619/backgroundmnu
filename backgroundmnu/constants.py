class const:
    G = 6.674e-11 #N m^2 / kg^2
    c_m = 2.99792458e8 # m/s
    c_km = 2.99792458e5 # km/s
    kb = 8.617333262e-5 #eV/K
    hbar = 6.582119569e-16 #eV s
    PI = 3.141592653589793
    Mpc_over_m = 3.08567758e22
    J_over_eV = 6.24150907e18
    rho100 = 3*(1e5/Mpc_over_m)**2*c_m**2 / (8*PI*G) * J_over_eV / (100**3) # critical density for a H0=100km/s/Mpc universe in eV/cm^3
    c_Mpc = c_m/Mpc_over_m
    m_H = 9.3895e8 # hydrogen mass in eV
    m_He = 3.7284e9 # helium mass in eV
    m_p = 9.382720813e8 # proton mass in eV
    T0 = 2.72548
    hfactor = 3.241e-18 # 100 km/s/Mpc in 1/s
    sigma_T = 6.65246e-25 #cm^2
