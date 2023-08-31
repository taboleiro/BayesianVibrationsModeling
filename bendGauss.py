import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS

import torch
from torch.distributions import constraints
import torch.distributions as distTorch
import matplotlib.pyplot as plt
import seaborn as sns
import time

import pickle
import graphviz
import utils

class inferenceProcess(object):
    def __init__(self, n_warmup=1, n_samples=1000, n_chains=1):
        self.beam = {}
        self.freq = []
        self.mobility = []

        self.E_mean=9.7e10
        self.E_std =5.0e9
        self.rho_mean=8000.0
        self.rho_std =250.0
        self.eta_mean=0.00505
        self.eta_std = 0.006

        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_chains = n_chains

    def beamProperties(self):
        """
        Define the physical properties of the beam used for this experiment
        """
        beam = {"length": 0.301,
            "width": 0.026,
            "thickness": 0.003,
            
            "E": 10e10,
            
            "mass": 0.1877,
            "freq": [[10, 470, 2],[470, 520, 1],[520, 600, 5], \
                    [600,700, 2],[700,1350, 20],[1350, 1390,2], \
                    [1390,1570,20],[1570,1630,2], [1630,2650,100], \
                    [2650,2750,2],[2750,2950, 20], [2950, 3050, 2]]
            }
        freqValues = [[117.5,122.5,1],[645.5,650.5,1],[1597.5,1600.5,1], [2977.5,2981.5,1]]
        # self.freqVal = utils.createComsolVector("lin", freqValues, param="step", display=False).astype(int)
        beam["massPerUnit"] = beam["mass"] / beam["length"]
        beam["volume"] = beam["length"] * beam["width"] * beam["thickness"]
        beam["I"] = beam["width"]*beam["thickness"]**3/12
        return beam


    def run(self):
        self.beam = self.beamProperties()

        # Reading and processing input data
        
        files = ["centerFreqResponse"]
        self.Y_exp = []
        for file in files:
            experiment = pd.read_csv("./Data/bend/"+file+".csv")[20:]
            # Mobility value calculated from input data and converted to torch
            mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
            self.Y_exp = mobility
            self.freq = torch.tensor(experiment["freq"].values) # Freq values(x axis) converted to torch
        self.Y_exp = torch.tensor(self.Y_exp)
        
        self.train()
        #map_estimate = pyro.param("E").item()
        #print("Our MAP estimate of the Young's modulus is {:.3f}".format(map_estimate)) 

    def destandarize(self, E_norm, rho_norm, eta_norm):
        E = E_norm * self.E_std + self.E_mean
        rho = rho_norm * self.rho_std + self.rho_mean
        eta = eta_norm * self.eta_std + self.eta_mean
        return E, rho, eta

    def mobilityFuncModel(self, E, rho, eta, freq):
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
        massPerUnit = rho*self.beam["thickness"]*self.beam["width"]
        cb = torch.sqrt(w)*(B/massPerUnit)**(1/4) # bending wave velocity
        
        kl = w/(cb)*l # bending wave number
        complex_kl = kl*(1-1j*eta/4)
        N_l = torch.cos(complex_kl)*torch.cosh(complex_kl) + 1
        D_l = torch.cos(complex_kl)*torch.sinh(complex_kl) + torch.sin(complex_kl)*torch.cosh(complex_kl)

        Y = -(1j*l)/ (2*complex_kl*torch.sqrt(complex_B *massPerUnit)) * N_l/D_l
        return abs(Y)

    def model_YoungDampingDensity(self, x, y_obs):
        # Young's modulus definition
        E_norm = pyro.sample("E", dist.Normal(0., 1.))
        # Density definition
        rho_norm = pyro.sample("rho", dist.Normal(0., 1.))
        # Damping loss factor definition
        eta_norm = pyro.sample("eta", dist.Normal(0., 1.))

        E, rho, eta = self.destandarize(E_norm, rho_norm, eta_norm)
        with pyro.plate("data", len(y_obs)):
            y_values = self.mobilityFuncModel(E, rho, eta, x)
            y = pyro.sample("y", dist.Normal(y_values, 0.001), obs=y_obs)
        return y

    def train(self):
        pyro.clear_param_store()
        y_obs = self.Y_exp # Suppose this was the vector of observed y's
        input_x = self.freq#[0:2000]
        pyro.render_model(self.model_YoungDampingDensity, model_args=(input_x, y_obs), render_distributions=True)
        
        nuts_kernel = NUTS(self.model_YoungDampingDensity)
        mcmc = MCMC(nuts_kernel, num_samples=self.n_samples, warmup_steps=self.n_warmup, num_chains=self.n_chains)      
        start = time.time()
        mcmc.run(input_x, y_obs)
        processTime = time.time() - start

        results = dict()
        results["n_warmup"] = self.n_warmup
        results["n_samples"] = self.n_samples
        results["n_chain"] = self.n_chains
        results["samples"] = mcmc.get_samples()
        results["summary"] = mcmc.summary()
        results["time"] = processTime

 
        posterior_samples = mcmc.get_samples()
        E, rho, eta = self.destandarize(posterior_samples["E"], posterior_samples["rho"], posterior_samples["eta"])
        E_est = E[np.argmax(E)]
        rho_est = rho[np.argmax(rho)]
        eta_est = eta[np.argmax(eta)]
        results["error"] = torch.sum((self.Y_exp-self.mobilityFuncModel(E_est, rho_est, eta_est, input_x))**2)
        
        results["Y_est"] = self.mobilityFuncModel(E_est, rho_est, eta_est, input_x)
        results["Y_exp"] = self.Y_exp

        plt.figure(10)
        plt.plot(self.freq, 20*np.log10(self.Y_exp))
        mob = self.mobilityFuncModel(E_est, rho_est, eta_est, input_x)
        
        plt.plot(self.freq, 20*np.log10(mob))
        plt.xscale("log")
        plt.show()

        sns.displot(posterior_samples["E"])
        plt.xlabel("Young's modulus values")
        plt.show()
                
        sns.displot(posterior_samples["rho"])
        plt.xlabel("density values")
        plt.show()
        with open('./PriorGauss_samples6318_1000_1.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return


if __name__ == "__main__":
    x = inferenceProcess()
    x.run()
