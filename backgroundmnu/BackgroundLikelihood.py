from cobaya.likelihood import Likelihood
import numpy as np
import os
import pickle as pkl


class BackgroundLikelihood(Likelihood):

    input_params = {"omega_b0": None, "omega_cdm0": None, }
    default_data_path = "./data/reduced_3p_likelihood_lcdm.pkl"

    def initialize(self, **params_values):
        
        # if self.data_path is None:
        self.data_path = os.path.abspath(getattr(self, "data_path", self.default_data_path))

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        with open(self.data_path, "rb") as f:
            loaded = pkl.load(f)
        
        self.mean = loaded["mean"]
        self.cov = loaded["cov"]
        self.inv_cov = np.linalg.pinv(self.cov)

    def get_requirements(self):

        return {"theta_star": None}
    
    def logp(self, **params_values):
        theta_star_100 = 100*self.provider.get_param("theta_star")
        omega_b = params_values['omega_b0']
        omega_c = params_values['omega_cdm0']
        v = np.array([omega_b, omega_c, theta_star_100])
        diff_vector = (v-self.mean)
        chi2 = np.dot(diff_vector, np.matmul(self.inv_cov, diff_vector))
        return -chi2 / 2