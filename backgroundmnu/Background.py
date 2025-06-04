import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import fsolve
import pickle as pkl
from copy import deepcopy
import os
from importlib.resources import files
from typing import Mapping, Iterable, NamedTuple, Sequence, Union, Optional, Callable, Protocol

import pyhyrec as pyhy
from .constants import con

class MnuModel(Protocol):
    def calculate_hubble(self, background, z):
        pass

class SymmetricSignedMass(MnuModel):

    def calculate_hubble(self, background, z):
        h2 = background.omega_cdm(z) + background.omega_b(z) + background.omega_gamma(z) + \
             background.omega_nu_massless(z) + background.omega_nu_massive(z)  + background.omega_de(z)

        massless_h2 = (background.massless_Hubble(z)*con.c_Mpc/con.hfactor)**2
        hfinal = massless_h2 + np.sign(background.mnu)*(h2 - massless_h2)

        return con.hfactor*np.sqrt(hfinal)/con.c_Mpc# in 1/Mpc

class SubtractRestMass(MnuModel):

    def calculate_hubble(self, background, z):
        sign = np.sign(background.mnu)

        rho = background.omega_nu_massive(z)
        p = background.p_nu_massive(z)

        omega_m = background.omega_b(z) + background.omega_cdm(z) + sign*(rho - 3*p)
        omega_r = background.omega_gamma(z) + background.omega_nu_massless(z) + 3*p
        h2 = omega_m + omega_r + background.omega_de(z)

        return con.hfactor*np.sqrt(h2)/con.c_Mpc# in 1/Mpc

class ZeroMass(MnuModel):

    def calculate_hubble(self, background, z):
        return background.massless_Hubble(z)

class Background():

    def __init__(self, new_params = None) -> None:
        
        self._defaults = {
            "omega_de0" : 0.3106861654538187, 
            "omega_b0" : 0.02237,
            "omega_cdm0" : 0.1200,
            "Nmassive" : 1,
            "mnu" : 0.058, #eV
            "YHe" : 0.245,
            "T0" : con.T0, # K
            "Tnu_massless": (4/11.)**(1/3),
            "Tnu_massive": 0.71611,
            "w0" : -1,
            "wa" : 0,
            "mnu_model": "symmetric",
            "with_reio": False} #in units of T0
        
        # Use pkg_resources to find the data file, regardless of where the package is imported from

        with open(os.path.join(files("backgroundmnu"), "data", "integral_tables.pkl"), 'rb') as f:
            int_tables = pkl.load(f)
    

        self.rho_nu_integral_table = int_tables["rho"]
        self.p_nu_integral_table = int_tables["p"] 

        self.create_params(new_params)


        self.set_mnu_model(self.mnu_model)

        self.setup()


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
        
        if self._params["mnu"] == 0.0: 
            self._params["Nmassive"] = 0
        
        for key, value in self._params.items():
            self.__dict__[key] = value

    def setup(self):

        self.coeff_switch_tabulate = 1e5

        self.Neff = 3.044 - self.Nmassive*(self.Tnu_massive/(4./11)**(1/3))**4

        self.calculate_recombination()
        self.calculate_optical_depths()

    def set_mnu_model(self, mnu_model_name):
        mnu_models = {
            "symmetric": SymmetricSignedMass(),
            "subtract_rest_mass": SubtractRestMass(),
            'massless': ZeroMass()
        }
        if mnu_model_name not in mnu_models:
            raise ValueError(f"Invalid mnu model: {mnu_model_name}. Available models are {', '.join(mnu_models.keys())}")
        self.mnu_model = mnu_models[mnu_model_name]

    def calculate_optical_depths(self):
        tau = self.optical_depth()
        tau_int = interp1d(tau.t, tau.y[0])
        self.z_star = fsolve(lambda z: tau_int(z)-1, 1080)[0]

        tau_baryon = self.baryon_optical_depth()
        tau_baryon_int = interp1d(tau_baryon.t, tau_baryon.y[0])
        self.z_drag = fsolve(lambda z: tau_baryon_int(z)-1, 1080)[0]

    #########################################################
    # Energy densities for different species
    #########################################################

    def omega_gamma(self,z):
        return np.pi**2/15 * (con.kb**4 / (con.hbar*con.c_m)**3) / (100**3) * (self.T0)**4*(1+z)**4 / con.rho100

    def omega_nu_massless(self, z, Neff = None): # energy density of massless (ultra-relativistic) neutrinos
        if Neff is not None:
            return Neff*(7./8)*(self.Tnu_massless)**(4.)*self.omega_gamma(z)
        else:
            #return self.Neff*(7./8)*(self.Tnu)**(4.)*self.rho_gamma(z)
            return self.Neff*(7./8)*(self.Tnu_massless)**(4.)*self.omega_gamma(z)

    def omega_nu_massive(self, z): # energy density of massive neutrinos
        if self.mnu==0.0:
            return 0.0
        mi = self.mnu/self.Nmassive
        Tnu_K = self.Tnu_massive*self.T0
        coeff1 = Tnu_K**4*(con.kb**4 / (con.hbar*con.c_m)**3) / (100**3) * (1+z)**4 / np.pi**2
        coeff2 = mi**2 / (Tnu_K*con.kb*(1+z))**2

        def get_integral(c2):
            if c2<self.coeff_switch_tabulate:
                return self.rho_nu_integral_table(c2)
            else:
                return scipy.integrate.quad(lambda x: x**2/(np.exp(x)+1)*np.sqrt(x**2 + c2), 0, np.inf, epsabs=1e-11)[0]
        
        if np.isscalar(coeff2):
            integral = get_integral(coeff2)
        else:
            integral = np.vectorize(get_integral)(coeff2)
        return coeff1*self.Nmassive*integral/con.rho100

    def p_nu_massive(self, z): # pressure of massive neutrinos
        Tnu_K = self.Tnu_massive*self.T0
        coeff1 = Tnu_K**4*(con.kb**4 / (con.hbar*con.c_m)**3) / (100**3) * (1+z)**4 / np.pi**2
        coeff2 = self.mnu**2 / (Tnu_K*con.kb*(1+z))**2

        def get_integral(c2):
            if c2<self.coeff_switch_tabulate:
                return self.p_nu_integral_table(c2)
            else:
                return scipy.integrate.quad(lambda x: (x**4/(3*np.sqrt(x**2 + c2)))*1/(np.exp(x)+1) , 0, np.inf, epsabs=1e-11)[0]

        if np.isscalar(coeff2):
            integral = get_integral(coeff2)
        else:
            integral = np.vectorize(get_integral)(coeff2)
        return coeff1*self.Nmassive*integral/con.rho100

    def omega_b(self, z):
        return self.omega_b0*(1+z)**3
    
    def omega_cdm(self, z):
        return self.omega_cdm0*(1+z)**3    

    def omega_de(self, z):
        a = 1/(1+z)
        return self.omega_de0*a**(-3*(1+self.w0+self.wa))*np.exp(-3*self.wa*(1-a))
    #########################################################
    # Distances
    #########################################################
    def Hubble(self,z):
        """Calculate Hubble parameter at redshift z. Implementation depends on the model for signed neutrino mass, which are implemented in subclasses."""
        return self.mnu_model.calculate_hubble(self, z)

    def massless_Hubble(self, z): # hubble factor assuming 3.044 massless neutrinos, the given omega_b and omega_c, to be used in calculations except for DA
        h2 = self.omega_cdm(z) + self.omega_b(z) + self.omega_gamma(z) + self.omega_nu_massless(z, 3.044) + self.omega_de(z)

        return con.hfactor*np.sqrt(h2)/con.c_Mpc # in 1/Mpc

    def DA(self, z):
        return self.angular_distance(z)*(1+z)

    def angular_distance(self, z, z_r=0):
        return scipy.integrate.quad(lambda zp: 1/self.Hubble(zp), z_r, z)[0]/(1+z)

    def h(self):
        return self.Hubble(0)/100*con.c_km
    #########################################################
    # Plasma quantities
    #########################################################

    def R(self, z):
        return ((3*self.omega_b0)/(4*self.omega_gamma(0)))*(1+z)**-1
    
    def cs(self, z):
        return np.sqrt(1/(3*(1+self.R(z))))

    def sound_horizon(self, z):
        #return scipy.integrate.quad(lambda zp: self.cs(zp)/self.Hubble(zp), z, np.inf)[0]
        return scipy.integrate.quad(lambda zp: self.cs(zp)/self.massless_Hubble(zp), z, np.inf)[0]

    def hydrogen_density(self,z): #n_H(z) in 1/cm^3
        return self.omega_b0*con.rho100*(1-self.YHe)/con.m_H*(1+z)**3

    #########################################################
    # Recombination quantities
    #########################################################

    def calculate_recombination(self):

        zmin = 0
        zmax = 8000.
        DLNA_SWIFT = 4e-3
        Nz = int((np.log((1.+zmax)/(1.+zmin))/DLNA_SWIFT) + 2)
        dz = zmax/Nz
        z = np.arange(start=zmin, stop=zmax, step=dz) # this matches the spacing that HYREC assumes 

        #hub = np.array([self.Hubble(zp) for zp in z])*con.c_Mpc
        hub = np.array([self.massless_Hubble(zp) for zp in z])*con.c_Mpc
        z, xe, _ = pyhy.call_run_hyrec_with_hubble(pyhy.HyRecCosmoParams()(), pyhy.HyRecInjectionParams()(), hub.flatten())

        if self.with_reio:
            reio = ReionizationModel(z_reio=7.67, YHe=self.YHe)
            xe_reio = reio.xe(z, xe)
            self.Xe = CubicSpline(z,xe_reio)
        else:
            self.Xe = CubicSpline(z,xe)

    def Thomson_scattering_rate(self, z):
        return self.Xe(z)*self.hydrogen_density(z)*con.sigma_T * con.Mpc_over_m*100 #numerical factor is to conver Hubble from 1/Mpc to 1/cm
    
    def optical_depth(self, z_end=2000):
        integrand = lambda z,y: self.Thomson_scattering_rate(z)/(self.Hubble(z)*(1+z))
        #integrand = lambda z,y: self.Thomson_scattering_rate(z)/(self.massless_Hubble(z)*(1+z))
        return scipy.integrate.solve_ivp(integrand, (0,z_end), [0], t_eval=np.linspace(0,z_end,int(z_end)))
    
    def baryon_optical_depth(self, z_end=2000):
        integrand = lambda z,y: self.Thomson_scattering_rate(z)/(self.Hubble(z)*(1+z))/self.R(z)
        #integrand = lambda z,y: self.Thomson_scattering_rate(z)/(self.massless_Hubble(z)*(1+z))/self.R(z)
        return scipy.integrate.solve_ivp(integrand, (0,z_end), [0], t_eval=np.linspace(0,z_end,int(z_end)))

    def theta_star(self):
        rs_star = self.sound_horizon(self.z_star)
        DA_star = self.angular_distance(self.z_star)

        return rs_star/DA_star/(1+self.z_star)

class ReionizationModel:

    def __init__(self, z_reio, delta_z_reio=0.5, YHe=0.2453):
        
        self.YHe = YHe
        _mHe_to_mH_ = 3.9715

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
