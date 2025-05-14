from cobaya.likelihood import Likelihood
import numpy as np
import os
import pickle as pkl
from importlib.resources import files

class BackgroundLikelihood(Likelihood):

    input_params = {"omega_b0": None, "omega_cdm0": None, }
    data_path = os.path.join(files("backgroundmnu"), "data")
    data_file = "plikHM_TTTEEE_lowl_lowE"
    mode = "3p"

    def initialize(self, **params_values):
        
        # self.data_file = os.path.abspath(getattr(self, "data_file", self.data_file))
        self.data_file_path = os.path.join(self.data_path, f"{self.data_file}.pkl")
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"Data file not found at {self.data_file_path}")

        print("Loading data from: ", self.data_file_path)

        with open(self.data_file_path, "rb") as f:
            loaded = pkl.load(f)
        if self.mode == "3p":
            self.mean = loaded["mean"]
            self.cov = loaded["cov"]
        elif self.mode == "2p":
            self.mean = loaded["mean"][:2]
            self.cov = loaded["cov"][:2,:2]
        self.inv_cov = np.linalg.pinv(self.cov)

    def get_requirements(self):
        if self.mode == "3p":   
            return {"theta_star": None}
        elif self.mode == "2p":
            return {}
    
    def logp(self, **params_values):
        omega_b = params_values['omega_b0']
        omega_c = params_values['omega_cdm0']
        v = np.array([omega_b, omega_c])
        if self.mode == "3p":
            v = np.append(v, 100*self.provider.get_param("theta_star"))
        diff_vector = (v-self.mean)
        chi2 = np.dot(diff_vector, np.matmul(self.inv_cov, diff_vector))
        return -chi2 / 2