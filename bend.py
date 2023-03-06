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


    def beamProperties(self):
        beam = {"length": 0.301,
            "width": 0.026,
            "thickness": 0.003,
            
            "E": 10e11,
            
            "mass": 0.1877
            }

        beam["massPerUnit"] = beam["mass"] / beam["length"]
        beam["volume"] = beam["length"] * beam["width"] * beam["thickness"]
        beam["I"] = beam["width"]*beam["thickness"]**3/12
        return beam

    def run(self):
        self.beam = self.beamProperties()

        self.experiment = pd.read_csv("./Data/bend/centerFreqResponse.csv")[10:]
        self.experiment["mobility"] = self.experiment["force"] + 1j*self.experiment["velocity"]

        """
        w = 2*np.pi*self.experiment["freq"]
        B = self.beam["E"]*self.beam["I"]
        cb = np.sqrt(w)*(B/self.beam["massPerUnit"])**(1/4)
        kl_data = w/cb * (1-1j*(0.0007/4))*self.beam["length"]
        """

        Y_exp = torch.tensor(abs(np.array(self.experiment["mobility"]))) # Suppose this was the vector of observed y's
        pyro.render_model(self.model, model_args=(Y_exp,), render_distributions=True, filename="model.pdf")

        self.train(self.model, self.guide, Y_exp)
        map_estimate = pyro.param("youngs").item()
        print("Our MAP estimate of the Young's modulus is {:.3f}".format(map_estimate)) 


    def mobilityFuncModel(self, E, eta=0.1):
        beam = self.beam
        freq = self.experiment["freq"].values
        l = beam["length"]/2

        E = E.detach().numpy()

        w = 2*np.pi*freq
        B = E*beam["I"]
        complex_B = E*(1+1j*eta)*beam["I"]
        cb = np.sqrt(w)*(B/beam["massPerUnit"])**(1/4)
        kl = w/cb*l

        complex_kl = kl*(1-1j*eta/4)
        
        N_l = np.cos(complex_kl)*np.cosh(complex_kl) + 1
        D_l = np.cos(complex_kl)*np.sinh(complex_kl) + np.sin(complex_kl)*np.cosh(complex_kl)

        #Y = -(0.25*eta+1j)*l/(2*kl*np.sqrt(B*beam["massPerUnit"])) * N_l/D_l
        Y = -1j*l/ (2*complex_kl*np.sqrt(complex_B*beam["massPerUnit"])) * N_l/D_l
        #Y = Y/max(Y)
        return torch.tensor(abs(Y))

    def model(self, Y_exp):
        E_theo = 10e10 # theoreticall Young's modulus of brass
        # define the hyperparameters that control the Beta prior
        alpha = torch.tensor(E_theo)
        beta = torch.tensor(3.)
        f = pyro.sample("latent_fairness", dist.Normal(alpha, beta)) 
        
        with pyro.plate("data", len(Y_exp)):
            Y = pyro.sample("Y", dist.Normal(self.mobilityFuncModel(f), 10.), obs=Y_exp)

    def guide(self, Y_exp):
        Youngs = pyro.param("youngs", torch.tensor(10e10),
                            constraint=constraints.real) #Theoretical value of Young's modulus
        pyro.sample("latent_fairness", dist.Normal(Youngs, 3.))

    def train(self, model, guide, Y_exp, lr=0.01, n_steps=2000):
        pyro.clear_param_store()
        adam_params = {"lr": lr}
        adam = pyro.optim.Adam(adam_params)
        svi = SVI(model, guide, adam, loss=Trace_ELBO())

        for step in range(n_steps):
            loss = svi.step(Y_exp)
            if step % 100 == 0:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))
                print(pyro.param("youngs").item())

if __name__ == "__main__":
    x = inferenceProcess()
    x.run()
