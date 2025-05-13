import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import fsolve
import pickle as pkl
from copy import deepcopy
from typing import Mapping, Iterable, NamedTuple, Sequence, Union, Optional, Callable

import pyhyrec as pyhy
from constants import const

# Internal functions used for computing quantities
# called from calculate above, or from other functions 
# in this set.
#########################################################

class Background():

    def __init__(self, new_params = None) -> None:
        
        self.with_dcdm = False

        self._defaults = {
            "omega_de" : 0.3106861654538187, 
            "omega_b0" : 0.02237,
            "omega_cdm0" : 0.1200,
            "Nmassive" : 1,
            "mnu" : 0.058, #eV
            "YHe"  : 0.245,
            "T0" : const.T0, # K
            "Tnu_massless": (4/11.)**(1/3),
            "Tnu_massive": 0.71611} #in units of T0}
        
        with open("./data/integral_tables.pkl", "rb") as f:
            int_tables = pkl.load(f)

        if self.with_dcdm:
            with open("/home/gplynch/projects/cobaya/dcdm_densities_Gam470_Om00315.pkl", "rb") as f:
                dcdm_densities = pkl.load(f)
            self.omega_dcdm = CubicSpline(dcdm_densities["z"], dcdm_densities["omega_dcdm"])
            self.omega_dr = CubicSpline(dcdm_densities["z"], dcdm_densities["omega_dr"])

        self.rho_nu_integral_table = int_tables["rho"]
        self.p_nu_integral_table = int_tables["p"] 

        self.create_params(new_params)

        if self.with_dcdm:
            zstar_fid = 1090
            omega_dcdm_star = self.omega_dcdm(zstar_fid)/(1+zstar_fid)**3
            self._params["omega_cdm0"]-=omega_dcdm_star

        for key, value in self._params.items():
            self.__dict__[key] = value
        
        self.z_switch_tabulate = 3000
        self.coeff_switch_tabulate = 1e5
        #self.tabulate_functions()

        self.Neff = 3.044 - self.Nmassive*(self.Tnu_massive/(4/11)**(1/3))**4


        zgrid = np.linspace(0, self.z_switch_tabulate, 10000)
        Hz = [self._Hubble(z) for z in zgrid]
        self.Hubble_table = CubicSpline(zgrid, Hz)

        self.h = self.Hubble(0)/100

        self.calculate_recombination()

        tau = self.optical_depth()
        tau_int = interp1d(tau.t, tau.y[0])
        self.z_star = fsolve(lambda z: tau_int(z)-1, 1080)[0]

        tau_baryon = self.baryon_optical_depth()
        tau_baryon_int = interp1d(tau_baryon.t, tau_baryon.y[0])
        self.z_drag = fsolve(lambda z: tau_baryon_int(z)-1, 1080)[0]

        # self.z_star = 1089.5
        # self.z_drag = 1060

    def create_params(self, new_params):
        self._params = self._defaults.copy()

        if new_params is None:
            return

        for key, value in new_params.items():
            # can only set parameters already present in the default dictionnary
            if key in self._defaults:
                self._params[key] = value
            else:
                raise ValueError("{} is not a recognized parameter.".format(key))

    def rho_gamma(self,z):
        return np.pi**2/15 * (const.kb**4 / (const.hbar*const.c_m)**3) / (100**3) * (self.T0)**4*(1+z)**4

    def rho_nu_massless(self, z, Neff = None): # energy density of massless (ultra-relativistic) neutrinos
        if Neff is not None:
            return Neff*(7./8)*(self.Tnu_massless)**(4.)*self.rho_gamma(z)
        else:
            #return self.Neff*(7./8)*(self.Tnu)**(4.)*self.rho_gamma(z)
            return self.Neff*(7./8)*(self.Tnu_massless)**(4.)*self.rho_gamma(z)

    def compute_rho_nu_massive(self, z): # energy density of massive neutrinos
        if self.mnu==0.0:
            return 0.0
        mi = self.mnu/self.Nmassive
        Tnu_K = self.Tnu_massive*self.T0
        coeff1 = Tnu_K**4*(const.kb**4 / (const.hbar*const.c_m)**3) / (100**3) * (1+z)**4 / np.pi**2
        coeff2 = mi**2 / (Tnu_K*const.kb*(1+z))**2
        if coeff2<self.coeff_switch_tabulate:
            integral = self.rho_nu_integral_table(coeff2)
        else:
            integral = scipy.integrate.quad(lambda x: x**2/(np.exp(x)+1)*np.sqrt(x**2 + coeff2), 0, np.inf, epsabs=1e-11)[0]
        return coeff1*self.Nmassive*integral

    # def compute_p_nu_total(self, z): # pressure of massive neutrinos
    #     Tnu_K = self.Tnu_massive*self.T0
    #     coeff1 = Tnu_K**4*(_kb_**4 / (_hbar_*_c_)**3) / (100**3) * (1+z)**4 / np.pi**2
    #     coeff2 = self.mnu**2 / (Tnu_K*_kb_*(1+z))**2
    #     if coeff2<self.coeff_switch_tabulate:
    #         integral = self.p_nu_integral_table(coeff2)
    #     else:
    #         integral = scipy.integrate.quad(lambda x: (x**4/(3*np.sqrt(x**2 + coeff2)))*1/(np.exp(x)+1) , 0, np.inf, epsabs=1e-11)[0]
    #     return coeff1*integral
    
    def rho_nu_massive(self, z): #wrapper function for energy density of massive neutrinos
        return self.compute_rho_nu_massive(z)
        
    # def p_nu_total(self,z): # wrapper for pressure of massive neutrinos
    #     return self.compute_p_nu_total(z)

    # def rho_nu_r(self, z):
    #     if self.mnu==0.0:
    #         return self.rho_nu_massless(z) # return contribution from massless neutrinos
    #     else:
    #         return self.rho_nu_massless(z) + 3*self.p_nu_total(z) #return contribution from massless neutrinos, plus radiationlike component of massive neutrinos

    # def rho_nu_m(self, z):
    #     if self.mnu==0:
    #         return 0
    #     else:
    #         rho_nu_m = self.rho_nu_massive(z) - 3*self.p_nu_total(z)
    #         return rho_nu_m # return (energy of massive nu - 3 * pressure of massive nu), which is the matterlike component of massive neutrinos

    def rho_r(self, z):
        return self.rho_gamma(z)+self.rho_nu_massless(z) #energy density of photons, ultra-relativisitc, and radiation-like component of massive neutrinos

    def omega_nu_massless(self, z):
        return self.rho_nu_massless(z)/const.rho100
    
    # def omega_nu_r(self,z):
    #     return self.rho_nu_r(z)/_rho100_
    
    def omega_nu_massive(self,z):
        return self.rho_nu_massive(z)/const.rho100
    
    def omega_gamma(self, z):
        return self.rho_gamma(z)/const.rho100
    
    def omega_r(self,z):
        return self.rho_r(z)/const.rho100

    def omega_nu(self, z):

        return self.omega_nu_massless(z) + self.omega_nu_massive(z) #energy of massles and massive neutrinos
        ## Total omega_nu 
    
    def omega_b(self, z):
        return self.omega_b0*(1+z)**3
    
    def omega_cdm(self, z):
        return self.omega_cdm0*(1+z)**3

    # def omega_m(self, z):
    #     sign = np.sign(self.mnu)
    #     return self.omega_b(z) + self.omega_cdm(z) + sign*self.omega_nu_m(z)

    def _Hubble(self,z):
        
        h2 = self.omega_cdm(z) + self.omega_b(z) + self.omega_gamma(z) + self.omega_nu(z) + self.omega_de
        if self.with_dcdm:
            h2 = h2 + self.omega_dcdm(z) + self.omega_dr(z) - self.omega_dcdm(0) - self.omega_dr(0)
        massless_h2 = (self.massless_Hubble(z)*const.c_Mpc/const.hfactor)**2
        #hz = np.sqrt(h2)*factor/_cMpc_
        #massless_hz = self.massless_Hubble(z)
        hfinal = massless_h2 + np.sign(self.mnu)*np.abs(h2 - massless_h2)
        return const.hfactor*np.sqrt(hfinal)/const.c_Mpc# in 1/Mpc

    def Hubble(self, z):
        if z<self.z_switch_tabulate:
            return self.Hubble_table(z)
        else:
            return self._Hubble(z)

    def massless_Hubble(self, z): # hubble factor assuming 3.044 massless neutrinos, the given omega_b and omega_c, to be used in calculations except for DA
        h2 = self.omega_cdm(z) + self.omega_b(z) + self.omega_gamma(z) + self.rho_nu_massless(z, 3.044)/const.rho100 + self.omega_de

        return const.hfactor*np.sqrt(h2)/const.c_Mpc # in 1/Mpc

    def R(self, z):
        return ((3*self.omega_b0)/(4*self.omega_gamma(0)))*(1+z)**-1
    
    def cs(self, z):
        return np.sqrt(1/(3*(1+self.R(z))))

    def angular_distance(self, z, z_r=0):
        return scipy.integrate.quad(lambda zp: 1/self.Hubble(zp), z_r, z)[0]/(1+z)

    def calculate_recombination(self):

        # if self.mnu!=0:
        #     Nmnu = 1
        # else:
        #     Nmnu = 0

        # cosmo_hy = pyhy.HyRecCosmoParams({"h" : self.h, 
        #         "T0" : self.T0,
        #         "Omega_b" : self.omega_b0/self.h**2,
        #         "Omega_cb" : (self.omega_b0+self.omega_cdm0)/self.h**2,
        #         "Neff": 3.044,
        #         "Nmnu" : Nmnu,
        #         "mnu1" : self.mnu,
        #         "mnu2" : 0.0 ,
        #         "mnu3" : 0.0})
        # inj = pyhy.HyRecInjectionParams()
        # z, xe, _ = pyhy.call_run_hyrec(cosmo_hy(), inj())

        zmin = 0
        zmax = 8000.
        DLNA_SWIFT = 4e-3
        Nz = int((np.log((1.+zmax)/(1.+zmin))/DLNA_SWIFT) + 2)
        dz = zmax/Nz
        z = np.arange(start=zmin, stop=zmax, step=dz) # this matches the spacing that HYREC assumes 

        hub = np.array([self.Hubble(zp) for zp in z])*const.c_Mpc
        #hub = np.array([self.massless_Hubble(zp) for zp in z])*_cMpc_
        z, xe, _ = pyhy.call_run_hyrec_with_hubble(pyhy.HyRecCosmoParams()(), pyhy.HyRecInjectionParams()(), hub.flatten())

        # reio = ReionizationModel(z_reio=7.67, YHe=self.YHe)
        # xe_reio = reio.xe(z, xe)

        #self.Xe = CubicSpline(z,xe_reio)
        self.Xe = CubicSpline(z,xe)


    def hydrogen_density(self,z): #n_H(z) in 1/cm^3
        return self.omega_b0*const.rho100*(1-self.YHe)/const.m_H*(1+z)**3

    def Thomson_scattering_rate(self, z):
        return self.Xe(z)*self.hydrogen_density(z)*const.sigma_T * const.Mpc_over_m*100 #numerical factor is to conver Hubble from 1/Mpc to 1/cm
    
    def optical_depth(self, z_end=2000):
        integrand = lambda z,y: self.Thomson_scattering_rate(z)/(self.Hubble(z)*(1+z))
        #integrand = lambda z,y: self.Thomson_scattering_rate(z)/(self.massless_Hubble(z)*(1+z))
        return scipy.integrate.solve_ivp(integrand, (0,z_end), [0], t_eval=np.linspace(0,z_end,int(z_end)))
    
    def baryon_optical_depth(self, z_end=2000):
        integrand = lambda z,y: self.Thomson_scattering_rate(z)/(self.Hubble(z)*(1+z))/self.R(z)
        #integrand = lambda z,y: self.Thomson_scattering_rate(z)/(self.massless_Hubble(z)*(1+z))/self.R(z)
        return scipy.integrate.solve_ivp(integrand, (0,z_end), [0], t_eval=np.linspace(0,z_end,int(z_end)))

    def sound_horizon(self, z):
        return scipy.integrate.quad(lambda zp: self.cs(zp)/self.Hubble(zp), z, np.inf)[0]
        #return scipy.integrate.quad(lambda zp: self.cs(zp)/self.massless_Hubble(zp), z, np.inf)[0]
    
    # def sound_horizon(self, z):
    #     a = 1/(1+z)
    #     astar = 1/(1+self.z_star)
    #     x = a/astar
    #     Rstar = self.R(self.z_star)
    #     xeq = self.omega_r0/((self.omega_b0+self.omega_cdm0)*astar)
    #     coeff_Mpc = 2998
    #     coeff = coeff_Mpc * (a/np.sqrt(self.omega_r0)) * 2 * np.sqrt(xeq/(3*Rstar))
    #     arg = (np.sqrt(Rstar)*np.sqrt(x+xeq) + np.sqrt(1+Rstar*x))/(1+np.sqrt(Rstar*xeq))
    #     return coeff*np.log(arg)
    
    def DA(self, z):
        return self.angular_distance(z)*(1+z)

    def theta_star(self):
        rs_star = self.sound_horizon(self.z_star)
        DA_star = self.angular_distance(self.z_star)

        return rs_star/DA_star/(1+self.z_star)
    
class ReionizationModel:

    def __init__(self, z_reio, delta_z_reio=0.5, YHe=0.2453):
        
        self.YHe = YHe
        _mHe_to_mH_ = 3.9715
        #_mHe_to_mH_ = 4.0

        #xe_after_reio: H + singly ionized He (checked before that denominator is non-zero) */
        self.xe_after_reio = 1. + YHe/(_mHe_to_mH_*(1.-YHe))

        self.reio_width = delta_z_reio
        self.reio_exponent = 1.5
        self.helium_fullreio_fraction = YHe/(_mHe_to_mH_*(1.-YHe))
        self.helium_fullreio_redshift = 3.5
        self.helium_fullreio_width = 0.5

        self.reio_redshift = z_reio

        #infer hydrogen start redshift redshift. 8 is the reionization_start_factor default from precision.h
        self.reio_start = self.reio_redshift + 8*self.reio_width
        self.xe = np.vectorize(self._xe)


    def _xe(self, z, x_before):

        if z>self.reio_start:
            return x_before
        else:
            argument = ((1+self.reio_redshift)**self.reio_exponent-(1+z)**(self.reio_exponent))/(self.reio_exponent*(1+self.reio_redshift)**(self.reio_exponent-1))/self.reio_width
            
            x = (self.xe_after_reio-x_before)*(np.tanh(argument)+1)/2. + x_before
            
            argument = (self.helium_fullreio_redshift - z)/self.helium_fullreio_width
            
            x+= self.helium_fullreio_fraction*(np.tanh(argument)+1.)/2.

        return x
