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

import graphviz
import utils

class inferenceProcess(object):
    def __init__(self):
        self.beam = {}
        self.freq = []
        self.mobility = [] 

        self.Ehigh = torch.tensor(12e10)
        self.Elow = torch.tensor(9e10)

        self.rhohigh = torch.tensor(8.8e3)
        self.rholow = torch.tensor(7.3e3)

        self.etahigh = torch.tensor(0.01)
        self.etalow = torch.tensor(0.0001)

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
        
        files = ["centerFreqResponse"]#, "center2FreqResponse", "randomFreqResponse"]
        self.Y_exp = []
        for file in files:
            experiment = pd.read_csv("./BayesianVibrationsModeling/Data/bend/"+file+".csv")[20:]
            # Mobility value calculated from input data and converted to torch
            mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
            self.Y_exp = mobility
            # self.Y_exp.append(abs(mobility))#[self.freqVal*2]))
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
        massPerUnit = rho*self.beam["thickness"]*self.beam["width"]
        cb = torch.sqrt(w)*(B/massPerUnit)**(1/4) # bending wave velocity
        
        kl = w/(cb)*l # bending wave number
        complex_kl = kl*(1-1j*eta/4)
        N_l = torch.cos(complex_kl)*torch.cosh(complex_kl) + 1
        D_l = torch.cos(complex_kl)*torch.sinh(complex_kl) + torch.sin(complex_kl)*torch.cosh(complex_kl)

        Y = -(1j*l)/ (2*complex_kl*torch.sqrt(complex_B *massPerUnit)) * N_l/D_l
        return abs(Y)

    def zzzmobilityFuncModel(self, E, freq, eta=0.007):
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

    def model_YoungDampingDensity(self, x, y_obs):
        # Density definition
        #rho_mean = pyro.param("rho_mean", dist.Normal(1, .25), constraint=constraints.positive)
        #rho_var = pyro.param("rho_var", dist.Cauchy(1., 0.))
        #rho = pyro.sample("rho", dist.Normal(1, .25))
        #rho = pyro.sample("rho", dist.Uniform(self.rholow, self.rhohigh))
        rho = pyro.sample("rho", dist.Beta(1, 1))
        # Damping loss factor definition
        #eta_mean = pyro.param("eta_mean", dist.Normal(1, 3.), constraint=constraints.positive)
        #eta_var = pyro.param("eta_var", dist.Cauchy(1., 0.))
        #eta = pyro.sample("eta", dist.Normal(1, 3.))
        eta = pyro.sample("eta", dist.Beta(1, 1))
        # Young's modulus definition
        #E_mean = pyro.param("E_mean", dist.Normal(0.99, .25), constraint=constraints.positive)
        #E_var = pyro.param("E_var", dist.Cauchy(1., 0.))
        #E = pyro.sample("E", dist.Normal(0.99, .25))
        E = pyro.sample("E", dist.Beta(1, 1))

        # E lim
        if E < self.Elow: 
            E = self.Elow
        elif E > self.Ehigh:
            E = self.Ehigh
        """
        # rho lim
        if rho < self.rholow: 
            rho = self.rholow
        elif rho > self.rhohigh:
            rho = self.rhohigh
        """
        # eta lim
        if eta < self.etalow: 
            eta = self.etalow
        elif eta > self.etahigh:
            eta = self.etahigh
        with pyro.plate("data", len(y_obs)):
            y_values = self.mobilityFuncModel(E*10e10, x, rho=rho*846 ,eta=eta+0.01)
            y = pyro.sample("y", dist.Normal(y_values, 0.0001), obs=y_obs)
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
            y_values = self.mobilityFuncModel(E*10e10, x, eta=eta*0.01)
            y = pyro.sample("y", dist.Normal(y_values, 1.), obs=y_obs)
        return y

    def model_Density(self, x, y_obs):
        # Density definition
        rho_mean = pyro.param("rho_mean", dist.Normal(1, 3.))
        rho_var = pyro.param("rho_var", dist.Cauchy(1., 0.))
        rho = pyro.sample("rho", dist.InverseGamma(rho_mean, rho_var))
        with pyro.plate("data", y_obs.shape[1]):
            y_values = self.mobilityFuncModel(0.99*10e10, x, rho=rho*8.4e-3)
            y = pyro.sample("y", dist.Normal(y_values, 1.), obs=y_obs)
        return y

    def Model_Young(self, x, y_obs):
        E_mean = pyro.param("E_mean", dist.Normal(1, 3.))
        E_var = pyro.param("E_var", dist.Cauchy(1., 0.))
        E = pyro.sample("E", dist.Normal(E_mean, E_var))
        with pyro.plate("data", y_obs.shape[1]):
            y_values = self.mobilityFuncModel(E*10e10, x)
            y = pyro.sample("y", dist.Normal(y_values, 1.), obs=y_obs)
        return y
    
    def Model_Damping(self, x, y_obs):
        eta_mean = pyro.param("eta_mean", dist.Normal(1, 3))
        eta_var = pyro.param("eta_var", dist.Cauchy(1., 0.))
        eta = pyro.sample("eta", dist.Normal(eta_mean, eta_var))
        with pyro.plate("data", y_obs.shape[1]):
            y_values = self.mobilityFuncModel(0.993, x, eta=eta*0.01)
            y = pyro.sample("y", dist.Normal(y_values, 0.0001), obs=y_obs)
        return y


    def train(self):
        pyro.clear_param_store()
        y_obs = self.Y_exp # Suppose this was the vector of observed y's
        input_x = self.freq#[0:2000]
        input_x = torch.tensor(input_x)#[0:2000]
        pyro.render_model(self.model_YoungDampingDensity, model_args=(input_x, y_obs), render_distributions=True)
        
        nuts_kernel = NUTS(self.model_YoungDampingDensity)
        mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=1000, num_chains=1)      
        mcmc.run(input_x, y_obs)

        # Show summary of inference results
        mcmc.summary()
        posterior_samples = mcmc.get_samples()
        
        plt.figure(10)
        plt.plot(self.freq, 20*np.log10(self.Y_exp))

        mob = self.mobilityFuncModel(10e10*posterior_samples["E"].mean(), input_x, rho=846*posterior_samples["rho"].mean() ,eta=0.01*posterior_samples["eta"].mean())
        plt.plot(self.freq, 20*np.log10(mob))
        plt.show()
        sns.displot(posterior_samples["E"])
        plt.xlabel("Young's modulus values")
        plt.show()
                
        sns.displot(posterior_samples["rho"])
        plt.xlabel("density values")
        plt.show()
        return


if __name__ == "__main__":
    x = inferenceProcess()
    x.run()
