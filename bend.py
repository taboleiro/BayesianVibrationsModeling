import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS

import torch
from torch.distributions import constraints

import matplotlib.pyplot as plt
import seaborn as sns

import graphviz

class inferenceProcess(object):
    def __init__(self):
        self.beam = {}
        self.freq = []
        self.mobility = [] 


    def beamProperties(self):
        """
        Define the physical properties of the beam used for this experiment
        """
        beam = {"length": 0.301,
            "width": 0.026,
            "thickness": 0.003,
            
            "E": 10e10,
            
            "mass": 0.1877
            }

        beam["massPerUnit"] = beam["mass"] / beam["length"]
        beam["volume"] = beam["length"] * beam["width"] * beam["thickness"]
        beam["I"] = beam["width"]*beam["thickness"]**3/12
        return beam

    def normalize(self, x):
        x = (x - x.mean()) / x.std()
        return x

    def run(self):
        self.beam = self.beamProperties()

        # Reading and processing input data
        
        files = ["centerFreqResponse", "center2FreqResponse", "randomFreqResponse"]
        self.Y_exp = []
        for file in files:
            experiment = pd.read_csv("./BayesianVibrationsModeling/Data/bend/"+file+".csv")[20:]
            # Mobility value calculated from input data and converted to torch
            mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
            self.Y_exp.append(abs(mobility))
            #self.Y_exp_norm = (self.Y_exp - self.Y_exp.mean()) / self.Y_exp.std() # Normalization
            self.freq = torch.tensor(experiment["freq"].values) # Freq values(x axis) converted to torch
        #self.Y_exp = pd.read_csv("./BayesianVibrationsModeling/Data/bend/"+file+".csv")[20:]
        self.Y_exp = torch.tensor(self.Y_exp)
        self.train()
        #map_estimate = pyro.param("E").item()
        #print("Our MAP estimate of the Young's modulus is {:.3f}".format(map_estimate)) 


    def mobilityFuncModel(self, E, freq, eta=0.007):
        """
        Calculates the mobility value based on the Young's modulus(E) and the frequency
        Input: 
            E   : Young's modulus
            eta : loss factor
        Output: 
            Y   : Mobility value
        """
        l = self.beam["length"]/2

        # calculating the bending wave number
        w = 2*torch.pi*freq # Angular frequency
        B = E*self.beam["I"] #
        complex_B = E*(1+1j*eta)*self.beam["I"]
        cb = torch.sqrt(w)*(B/self.beam["massPerUnit"])**(1/4) # bending wave velocity
        
        kl = w/(cb)*l # bending wave number
        complex_kl = kl*(1-1j*eta/4)
        N_l = torch.cos(complex_kl)*torch.cosh(complex_kl) + 1
        D_l = torch.cos(complex_kl)*torch.sinh(complex_kl) + torch.sin(complex_kl)*torch.cosh(complex_kl)

        Y = -(1j*l)/ (2*complex_kl*torch.sqrt(complex_B *self.beam["massPerUnit"])) * N_l/D_l
        return abs(Y)

    def errorEstimation(self, E):
        Y_calc = self.mobilityFuncModel(E)
        Y_calc_norm = (Y_calc - Y_calc.mean()) / Y_calc.std()
        #error = abs(self.Y_exp_norm - Y_calc)
        return Y_calc_norm
        """
        def model(self, Y_exp):
            E_theo = 10e10 # theoreticall Young's modulus of brass
            # define the hyperparameters that control the Beta prior
            alpha = torch.tensor(E_theo) # E normalized
            beta = torch.tensor(3.)
            E = pyro.sample("E", dist.Normal(alpha, beta))
            #error = pyro.sample("error", dist.Normal(error_value, 3.))
            E = pyro.sample("E", dist.Normal(E_theo, 1.))
            with pyro.plate("data", len(Y_exp)):
                Y = pyro.sample("Y_est", dist.Normal(self.errorEstimation(E), 1.))#, obs=abs(self.Y_exp_norm))
        """
    def model(self, freq, Y_exp):
        E = pyro.sample("E",dist.Normal(1, 3.))
        with pyro.plate("data", len(Y_exp)):
            y = pyro.sample("y", pyro.distributions.Normal(self.errorEstimation(E*10e10), 1.), obs=Y_exp)
        return y

    def model_YoungDamping(self, x, y_obs):
        # Damping loss factor definition
        eta_mean = pyro.param("eta_mean", dist.Normal(1, 3.))
        eta_var = pyro.param("eta_var", dist.Cauchy(1., 0.))
        eta = pyro.sample("eta", dist.Normal(eta_mean, eta_var))
        # Young's modulus definition
        E_mean = pyro.param("E_mean", dist.Normal(1, 3.))
        E_var = pyro.param("E_var", dist.Cauchy(1., 0.))
        E = pyro.sample("E", dist.Normal(E_mean, E_var))
        with pyro.plate("data", y_obs.shape[1]):
            y_values = self.normalize(self.mobilityFuncModel(E*10e10, x, eta=eta*0.01))
            y = pyro.sample("y", dist.Normal(y_values, 1.), obs=y_obs)
        return y

    def Model_Young(self, x, y_obs):
        E_mean = pyro.param("E_mean", dist.Normal(1, 3.))
        E_var = pyro.param("E_var", dist.Cauchy(1., 0.))
        E = pyro.sample("E", dist.Normal(E_mean, E_var))
        with pyro.plate("data", y_obs.shape[1]):
            y_values = self.normalize(self.mobilityFuncModel(E*10e10, x))
            y = pyro.sample("y", dist.Normal(y_values, 1.), obs=y_obs)
        return y


    def train(self, lr=0.01, n_steps=2000):
        pyro.clear_param_store()
        y_obs = self.Y_exp # Suppose this was the vector of observed y's
        input_x = self.freq
        pyro.render_model(self.Model_Young, model_args=(input_x, self.normalize(y_obs)), render_distributions=True)
        
        nuts_kernel = NUTS(self.Model_Young)
        mcmc = MCMC(nuts_kernel, num_samples=len(self.freq), warmup_steps=500, num_chains=1)        
        mcmc.run(input_x, self.normalize(y_obs))

        # Show summary of inference results
        mcmc.summary()
        posterior_samples = mcmc.get_samples()
        
        sns.displot(posterior_samples["E"]*10e10)
        plt.xlabel("E values")
        plt.show()
        return


if __name__ == "__main__":
    x = inferenceProcess()
    x.run()
