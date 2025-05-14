from typing import Dict, List, Optional, Sequence, Union, Mapping, Any, NamedTuple, Callable

import numpy as np
from copy import deepcopy

from cobaya.theory import Theory
from cobaya.tools import Pool1D, Pool2D, PoolND, combine_1d
from cobaya.log import LoggedError
from cobaya.typing import InfoDict

from .Background import Background
from .constants import con

H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": con.c_km}

class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence, None] = None
    z_pool: Optional[PoolND] = None
    post: Optional[Callable] = None

class BackgroundTheory(Theory):

#########################################################
# Functions needed or used by cobaya
#########################################################

    extra_args: InfoDict
    _defaults = {
        "omega_de0": 0.3106861654538187,
        "omega_b0": 0.02237,
        "omega_cdm0": 0.1200,
        "mnu": 0.058,
        "w0": -1.0,
        "wa": 0.0,
        "Nmassive": 1,
        "YHe": 0.245,
        "T0": con.T0,
        "Tnu_massless": (4/11.)**(1/3),
        "Tnu_massive": 0.71611,
        "mnu_model": "symmetric",
        "with_reio": False
    }

    # Define acceptable model values
    MNU_MODELS = ["symmetric", "subtract_rest_mass", "massless"]

    def initialize(self):
        """called from __init__ to initialize"""
        # List of all possible parameters
        self.all_parameters = list(self._defaults.keys())
        
        self.collectors = {}
        self._must_provide = {}
        self.input_parameters = {}
        
        # Process extra_args if provided
        if hasattr(self, 'extra_args') and self.extra_args:
            for k, v in self.extra_args.items():
                if k in self.all_parameters:
                    self.input_parameters[k] = v
        
        # Set default neutrino model if not specified
        if "mnu_model" not in self.input_parameters:
            self.input_parameters["mnu_model"] = "symmetric"
            
        # Validate mnu_model
        if self.input_parameters["mnu_model"] not in self.MNU_MODELS:
            raise LoggedError(
                self.log, 
                f"Invalid mnu_model: '{self.input_parameters['mnu_model']}'. "
                f"Must be one of {self.MNU_MODELS}"
            )

    def get_allow_agnostic(self):
        return True

    def get_can_support_params(self):
        return self.all_parameters
    
    def get_requirements(self):
        """
        Return the requirements dictionary for optional parameters.
        This allows the user to specify only the parameters they want to vary.
        """
        # Return empty dict to indicate no hard requirements
        return {}

    def must_provide(self, **requirements):
        # Computed quantities required by the likelihood

        super().must_provide(**requirements)
        self._must_provide = self._must_provide or dict.fromkeys(self.output_params or [])

        for k, v in requirements.items():
            if k in {"Hubble", "angular_diameter_distance"}:
                if k not in self._must_provide:
                    self._must_provide[k] = {}
                if not isinstance(v, Mapping) or "z" not in v:
                    raise LoggedError(
                        self.log,
                        f"The value in the dictionary of requisites {k} must be a "
                        "dictionary containing the key 'z' with a list of redshifts "
                        f"(got instead {{{k}: {v}}})"
                    )
                self._must_provide[k]["z"] = combine_1d(
                    v["z"], self._must_provide[k].get("z"))

        for k, v in self._must_provide.items():
            if k == "Hubble":
                self.set_collector_with_z_pool(
                    k, v["z"], "Hubble", args_names=["z"], arg_array=0)
            elif k == "angular_diameter_distance":
                self.set_collector_with_z_pool(
                    k, v["z"], "angular_distance", args_names=["z"], arg_array=0)
                

    def get_can_provide(self):
        return ['Hubble', "angular_diameter_distance"]

    def get_can_provide_params(self):
        return ['zstar', 'zdrag', 'rdrag', 'rstar', 'theta_star', "h", "w0", "wa"]
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        params = self.input_parameters.copy()
        
        # Add sampled/input parameters, which override any defaults
        for param in self.all_parameters:
            if param in params_values_dict:
                params[param] = params_values_dict[param]

        try:
            self.Background = Background(params)
        except Exception as e:
            print(f"Error creating Background: {e}")
            return False

        for product, collector in self.collectors.items():
            # Special case: sigma8 needs H0, which cannot be known beforehand:
            method = getattr(self.Background, collector.method)
            arg_array = self.collectors[product].arg_array
            if isinstance(arg_array, int):
                arg_array = np.atleast_1d(arg_array)
            if arg_array is None:
                state[product] = method(
                    *self.collectors[product].args, **self.collectors[product].kwargs)
            elif isinstance(arg_array, Sequence) or isinstance(arg_array, np.ndarray):
                arg_array = np.array(arg_array)
                if len(arg_array.shape) == 1:
                    # if more than one vectorised arg, assume all vectorised in parallel
                    n_values = len(self.collectors[product].args[arg_array[0]])
                    state[product] = np.zeros(n_values)
                    args = deepcopy(list(self.collectors[product].args))
                    for i in range(n_values):
                        for arg_arr_index in arg_array:
                            args[arg_arr_index] = \
                                self.collectors[product].args[arg_arr_index][i]
                        state[product][i] = method(
                            *args, **self.collectors[product].kwargs)
                elif len(arg_array.shape) == 2:
                    if len(arg_array) > 2:
                        raise NotImplementedError("Only 2 array expanded vars so far.")
                    # Create outer combinations
                    x_and_y = np.array(np.meshgrid(
                        self.collectors[product].args[arg_array[0, 0]],
                        self.collectors[product].args[arg_array[1, 0]])).T
                    args = deepcopy(list(self.collectors[product].args))
                    result = np.empty(shape=x_and_y.shape[:2])
                    for i, row in enumerate(x_and_y):
                        for j, column_element in enumerate(x_and_y[i]):
                            args[arg_array[0, 0]] = column_element[0]
                            args[arg_array[1, 0]] = column_element[1]
                            result[i, j] = method(
                                *args, **self.collectors[product].kwargs)
                    state[product] = (
                        self.collectors[product].args[arg_array[0, 0]],
                        self.collectors[product].args[arg_array[1, 0]], result)
                else:
                    raise ValueError("arg_array not correctly formatted.")
            elif arg_array in self.collectors[product].kwargs:
                value = np.atleast_1d(self.collectors[product].kwargs[arg_array])
                state[product] = np.zeros(value.shape)
                for i, v in enumerate(value):
                    kwargs = deepcopy(self.collectors[product].kwargs)
                    kwargs[arg_array] = v
                    state[product][i] = method(
                        *self.collectors[product].args, **kwargs)
            else:
                raise LoggedError(self.log, "Variable over which to do an array call "
                                            f"not known: arg_array={arg_array}")
            if collector.post:
                state[product] = collector.post(*state[product])
    
        if want_derived:
            derived_params = {
                'theta_star': self.Background.theta_star(),
                'zdrag': self.Background.z_drag, 
                'z_star': self.Background.z_star, 
                'rdrag': self.Background.sound_horizon(self.Background.z_drag), 
                'rstar': self.Background.sound_horizon(self.Background.z_star),
                'h': self.Background.h()
            }
            
            # Add w0, wa as derived parameters if they weren't sampled
            if "w0" not in params_values_dict and hasattr(self.Background, "w0"):
                derived_params["w0"] = self.Background.w0
            if "wa" not in params_values_dict and hasattr(self.Background, "wa"):
                derived_params["wa"] = self.Background.wa
                
            state['derived'] = derived_params

    def get_Hubble(self, z, units="km/s/Mpc"):
        try:
            return self._get_z_dependent("Hubble", z) * H_units_conv_factor[units]
        except KeyError:
            raise LoggedError(
                self.log, "Units not known for H: '%s'. Try instead one of %r.",
                units, list(H_units_conv_factor))
    
    def get_angular_diameter_distance(self, z):
        r"""
        Returns the physical angular diameter distance in :math:`\mathrm{Mpc}` to the
        given redshift(s) ``z``.

        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        """
        return self._get_z_dependent("angular_diameter_distance", z)
    
    def set_collector_with_z_pool(self, k, zs, method, args=(), args_names=(),
                                  kwargs=None, arg_array=None, post=None, d=1):
        """
        Creates a collector for a z-dependent quantity, keeping track of the pool of z's.

        If ``z`` is an arg, i.e. it is in ``args_names``, then omit it in the ``args``,
        e.g. ``args_names=["a", "z", "b"]`` should be passed together with
        ``args=[a_value, b_value]``.
        """
        if k in self.collectors:
            z_pool = self.collectors[k].z_pool
            z_pool.update(zs)
        else:
            Pool = {1: Pool1D, 2: Pool2D}[d]
            z_pool = Pool(zs)
        # Insert z as arg or kwarg
        kwargs = kwargs or {}
        if d == 1 and "z" in kwargs:
            kwargs = deepcopy(kwargs)
            kwargs["z"] = z_pool.values
        elif d == 1 and "z" in args_names:
            args = deepcopy(args)
            i_z = args_names.index("z")
            args = list(args[:i_z]) + [z_pool.values] + list(args[i_z:])
        elif d == 2 and "z1" in args_names and "z2" in args_names:
            # z1 assumed appearing before z2!
            args = deepcopy(args)
            i_z1 = args_names.index("z1")
            i_z2 = args_names.index("z2")
            args = (list(args[:i_z1]) + [z_pool.values[:, 0]] + list(args[i_z1:i_z2]) +
                    [z_pool.values[:, 1]] + list(args[i_z2:]))
        else:
            raise LoggedError(
                self.log,
                f"I do not know how to insert the redshift for collector method {method} "
                f"of requisite {k}")
        self.collectors[k] = Collector(
            method=method, z_pool=z_pool, args=args, args_names=args_names, kwargs=kwargs,
            arg_array=arg_array, post=post)

    def _get_z_dependent(self, quantity, z, pool=None):
        if pool is None:
            pool = self.collectors[quantity].z_pool
        try:
            i_kwarg_z = pool.find_indices(z)
        except ValueError:
            raise LoggedError(self.log, f"{quantity} not computed for all z requested. "
                                        f"Requested z are {z}, but computed ones are "
                                        f"{pool.values}.")
        return np.array(self.current_state[quantity], copy=True)[i_kwarg_z]