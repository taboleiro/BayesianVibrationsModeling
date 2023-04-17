import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from pyro.optim import Adam

import torch
from torch.distributions import constraints

import matplotlib.pyplot as plt
import seaborn as sns

import graphviz
import utils

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
            
            "mass": 0.1877,
            "freq": [[10, 470, 2],[470, 520, 1],[520, 600, 5], \
                    [600,700, 2],[700,1350, 20],[1350, 1390,2], \
                    [1390,1570,20],[1570,1630,2], [1630,2650,100], \
                    [2650,2750,2],[2750,2950, 20], [2950, 3050, 2]]
            }
        freqValues = [[117.5,122.5,1],[645.5,650.5,1],[1597.5,1600.5,1], [2977.5,2981.5,1]]
        self.freqVal = utils.createComsolVector("lin", freqValues, param="step", display=False).astype(int)
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
            experiment = pd.read_csv("./Data/bend/"+file+".csv")[20:]
            # Mobility value calculated from input data and converted to torch
            mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
            self.Y_exp.append(abs(mobility))#[self.freqVal*2]))
            #self.Y_exp_norm = (self.Y_exp - self.Y_exp.mean()) / self.Y_exp.std() # Normalization
            self.freq = torch.tensor(experiment["freq"].values) # Freq values(x axis) converted to torch
        #self.Y_exp = pd.read_csv("./BayesianVibrationsModeling/Data/bend/"+file+".csv")[20:]
        self.Y_exp = torch.tensor(self.Y_exp)
        self.train()
        #map_estimate = pyro.param("E").item()
        #print("Our MAP estimate of the Young's modulus is {:.3f}".format(map_estimate)) 


    def mobilityFuncModel(self, E, freq, rho=8.4e-3, eta=0.007):
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
        massPerUnit = rho*self.beam["thickness"]
        cb = torch.sqrt(w)*(B/massPerUnit)**(1/4) # bending wave velocity
        
        kl = w/(cb)*l # bending wave number
        complex_kl = kl*(1-1j*eta/4)
        N_l = torch.cos(complex_kl)*torch.cosh(complex_kl) + 1
        D_l = torch.cos(complex_kl)*torch.sinh(complex_kl) + torch.sin(complex_kl)*torch.cosh(complex_kl)

        Y = -(1j*l)/ (2*complex_kl*torch.sqrt(complex_B *massPerUnit)) * N_l/D_l
        return abs(Y)


    def model_YoungDampingDensity(self, x, y_obs):
        # Density definition
        #rho_mean = pyro.param("rho_mean", dist.Normal(1, 0.0001))
        #rho_var = pyro.param("rho_var", dist.Cauchy(1., 0.00001))
        rho = pyro.sample("rho", dist.Normal(1, 1))
        # Damping loss factor definition
        #eta_mean = pyro.param("eta_mean", dist.Normal(1, 3.))
        #eta_var = pyro.param("eta_var", dist.Cauchy(1., 0.))
        eta = pyro.sample("eta", dist.Normal(1, 1))
        # Young's modulus definition
        #E_mean = pyro.param("E_mean", dist.Normal(1, .5))
        #E_var = pyro.param("E_var", dist.Cauchy(1., 0.5))
        E = pyro.sample("E", dist.Normal(1, 1))
        with pyro.plate("data", y_obs.shape):
            y_values = self.mobilityFuncModel(E*10e10, x, rho=rho*8.4e-3,eta=eta*0.01)
            y = pyro.sample("y", dist.Normal(y_values, 1.), obs=y_obs)
        return y

    def guide(self, x, y_obs):
        # Density definition
        #rho_mean = pyro.param("rho_mean", dist.Normal(1, 0.0001))
        #rho_var = pyro.param("rho_var", dist.Cauchy(1., 0.00001))
        rho = pyro.sample("rho", dist.Normal(1.5, 1))
        # Damping loss factor definition
        #eta_mean = pyro.param("eta_mean", dist.Normal(1, 3.))
        #eta_var = pyro.param("eta_var", dist.Cauchy(1., 0.))
        eta = pyro.sample("eta", dist.Normal(1.1, 1))
        # Young's modulus definition
        #E_mean = pyro.param("E_mean", dist.Normal(1, .5))
        #E_var = pyro.param("E_var", dist.Cauchy(1., 0.5))
        E = pyro.sample("E", dist.Normal(0.9996, 1))


    def train(self):
        pyro.clear_param_store()
        y_obs = self.Y_exp # Suppose this was the vector of observed y's
        input_x = torch.tensor(self.freq)#[0:2000]
        #pyro.render_model(, model_args=(input_x, y_obs), render_distributions=True)

        # setup the optimizer
        adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)

        # setup the inference algorithm
        svi = SVI(self.model_YoungDampingDensity, self.guide, optimizer, loss=Trace_ELBO())

        # do gradient steps
        n_steps = 500
        for step in range(n_steps):
            svi.step(input_x, y_obs[0])
            if step % 100 == 0:
                print('.', end='')
        nuts_kernel = NUTS(self.model_YoungDampingDensity)
        mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=500, num_chains=1)        
        mcmc.run(input_x, y_obs)

        # Show summary of inference results
        mcmc.summary()
        posterior_samples = mcmc.get_samples()
        
        sns.displot(posterior_samples["E"]*10e10)
        plt.xlabel("Young's modulus values")
        plt.show()
                
        sns.displot(posterior_samples["rho"]*10e10)
        plt.xlabel("density values")
        plt.show()
        return


if __name__ == "__main__":
    x = inferenceProcess()
    x.run()
