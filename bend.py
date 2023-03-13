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

    def run(self):
        self.beam = self.beamProperties()

        # Reading and processing input data
        
        files = ["centerFreqResponse", "center2FreqResponse", "randomFreqResponse"]
        self.Y_exp = []
        for file in files:
            experiment = pd.read_csv("./Data/bend/"+file+".csv")[20:]
            # Mobility value calculated from input data and converted to torch
            mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
            self.Y_exp.append(mobility)
            #self.Y_exp_norm = (self.Y_exp - self.Y_exp.mean()) / self.Y_exp.std() # Normalization
            self.freq = torch.tensor(experiment["freq"].values) # Freq values(x axis) converted to torch
        self.Y_exp = pd.read_csv("./Data/bend/"+file+".csv")[20:]
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
        # C
        """
        beam = self.beam
        l = beam["length"]/2

        # calculating the bending wave number
        w = 2*np.pi*freq # Angular frequency
        B = E*beam["I"] #
        complex_B = E*(1+1j*eta)*beam["I"]
        cb = torch.sqrt(w)*(B/beam["massPerUnit"])**(1/4) # bending wave velocity
        
        kl = w/cb*l # bending wave number
        complex_kl = kl*(1-1j*eta/4)
        
        N_l = torch.cos(complex_kl)*torch.cosh(complex_kl) + 1
        D_l = torch.cos(complex_kl)*torch.sinh(complex_kl) + torch.sin(complex_kl)*torch.cosh(complex_kl)

        #Y = -(0.25*eta+1j)*l/(2*kl*np.sqrt(B*beam["massPerUnit"])) * N_l/D_l
        # The mobility is a complex value but just the absolute value is shown 
        Y = abs(-1j*l/ (2*complex_kl*torch.sqrt(complex_B*beam["massPerUnit"])) * N_l/D_l)
        """
        beam = self.beam
        l = beam["length"]/2

        # calculating the bending wave number
        w = 2*torch.pi*freq # Angular frequency
        B = E*beam["I"] #
        complex_B = E*(1+1j*eta)*beam["I"]
        cb = torch.sqrt(w)*(B/beam["massPerUnit"])**(1/4) # bending wave velocity
        
        kl = w/(cb)*l # bending wave number
        complex_kl = kl*(1-1j*eta/4)
        N_l = torch.cos(kl)*torch.cosh(kl) + 1
        D_l = torch.cos(kl)*torch.sinh(kl) + torch.sin(kl)*torch.cosh(kl)

        Y = -(1j*l)/ (2*complex_kl*l*torch.sqrt(complex_B*beam["massPerUnit"])) * N_l/D_l
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

    def my_model(self, x, y_obs):
        beta = pyro.sample("beta", pyro.distributions.Normal(0., 1.))
        with pyro.plate("data", len(y_obs)):
            y = pyro.sample("y", pyro.distributions.Normal(beta*x, 1.), obs=y_obs)
        return y

    def my_model_Y(self, x, y_obs):
        E_mean = pyro.param("E_mean", dist.Normal(1, 3.))
        E_var = pyro.param("E_var", dist.Cauchy(1., 0.))
        E = pyro.sample("E", dist.Normal(E_mean, E_var))
        with pyro.plate("data", y_obs.shape[1]):
            y = pyro.sample("y", dist.Normal(self.mobilityFuncModel(E*10e10, x), 1.), obs=y_obs)
        return y


    def train(self, lr=0.01, n_steps=2000):
        pyro.clear_param_store()
        y_obs = self.Y_exp # Suppose this was the vector of observed y's
        input_x = self.freq
        pyro.render_model(self.my_model_Y, model_args=(input_x, y_obs), render_distributions=True)
        #sampled_y = self.my_model(input_x)
        #nuts_kernel = NUTS(model, jit_compile=args.jit)
        #mcmc = MCMC()
        # Run inference in Pyro
        
        nuts_kernel = NUTS(self.my_model_Y)
        mcmc = MCMC(nuts_kernel, num_samples=len(self.freq), warmup_steps=500, num_chains=1)
        
        mcmc.run(input_x, y_obs)

        # Show summary of inference results
        mcmc.summary()
        posterior_samples = mcmc.get_samples()
        
        sns.displot(posterior_samples["E"]*10e10)
        plt.xlabel("E values")
        plt.show()
        return

    def train2(self, lr=0.01, n_steps=2000):
        pyro.clear_param_store()
        # nuts_kernel = NUTS(model, jit_compile=args.jit)
        # mcmc = MCMC()
        # Run inference in Pyro
        """
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500, num_chains=1)
        mcmc.run(self.freq, self.Y_exp)

        # Show summary of inference results
        mcmc.summary()
        """
        adam_params = {"lr": lr}
        adam = pyro.optim.Adam(adam_params)
        svi = SVI(self.model, adam, loss=Trace_ELBO())
        for step in range(n_steps):
            loss = svi.step(Y_exp)
            if step % 100 == 0:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))
                print(pyro.param("youngs").item())

if __name__ == "__main__":
    x = inferenceProcess()
    x.run()
