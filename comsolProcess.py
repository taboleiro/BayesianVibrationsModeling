import mph
import matplotlib.pyplot as plt

import pyro
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, HMC

import torch
from torch.distributions import constraints

import matplotlib.pyplot as plt
import seaborn as sns
import time 

import graphviz
import utils

class ComsolProcess(object):
    def __init__(self, comsolModelPath, inputFiles, freqValues):
        # initializing variables
        self.inputFiles = inputFiles
        self.freqValues = freqValues
        self.E_theo = 10e10
        self.rho_theo = 8.34e3
        self.eta_theo = 0.01
        # Starting comsol process
        client = mph.start()
        self.model = client.load(comsolModelPath)


    def updateParams(self, model, young=1, rho=1, eta=1):#, freq=10):
        model.parameter('Youngs', str(young*self.E_theo)+' [Pa]')
        model.parameter('density', str(rho*self.rho_theo)+' [kg/m^3]')
        model.parameter('damping', str(eta*self.eta_theo)+' [Pa]')
        #model.parameter('freq', str(freq)+' [Hz]')


    def model_YoungDampingDensity(self, x, y_obs):
        # Density definition
        rho_mean = pyro.param("rho_mean", dist.Normal(1., 0.5), constraint=constraints.real)
        rho_var = pyro.param("rho_var", dist.Cauchy(1., 0.), constraint=constraints.positive)
        #rho_var = pyro.param("rho_var", dist.Cauchy(1., 0.), constraint=constraints.positive)
        rho = pyro.sample("rho", dist.Normal(rho_mean, rho_var))
        # Damping loss factor definition
        eta_mean = pyro.param("eta_mean", dist.Normal(1., .01), constraint=constraints.positive)
        eta_var = pyro.param("eta_var", dist.Cauchy(1., .5))
        eta = pyro.sample("eta", dist.Normal(eta_mean, eta_var))
        # Young's modulus definition
        E_mean = pyro.param("E_mean", dist.Normal(1, .01), constraint=constraints.positive)
        E_var = pyro.param("E_var", dist.Cauchy(1., 0.), constraint=constraints.positive)
        E = pyro.sample("E", dist.Normal(E_mean, E_var))
        # Since the studio is done frequcuency by frequency, the loop can't be vectorized like: "with pyro.plate......"
        y = torch.zeros(len(y_obs))
        with pyro.plate("data", len(y_obs)):
            #updateParam(model, young=E.detach().numpy(), rho=rho.detach().numpy(), eta=eta.detach().numpy(), freq=x[i])
            self.updateParams(self.model, young=E.detach().numpy(), rho=rho.detach().numpy())
            self.model.solve("Study 3")
            comsolResults = abs(self.model.evaluate('comp1.point1'))
            y = pyro.sample("y", dist.Normal(torch.tensor(comsolResults), 0.5), obs=y_obs)
        return y

    def run(self):

        freqVal = utils.createComsolVector("lin", self.freqValues, param="step", display=False)
        # Reading input files
        Y_exp = np.array([])
        for file in self.inputFiles:
            experiment = pd.read_csv("./Data/bend/"+file+".csv")
            # Mobility value calculated from input data and converted to torch
            mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
            Y_exp = np.append(Y_exp, abs(mobility))
            freq = torch.tensor(experiment["freq"].values) # Freq values(x axis) converted to torch

        
        
        # BAYESIAN INFERENCE PROCESS
        # The experimental data has a resolution of 0.5 Hz. The values selected are integers
        # so the position of the freq. values in the array will be x*2
        input_x = torch.tensor(freq[(freqVal*2).astype(int)])
        y_obs = torch.tensor(Y_exp[(freqVal*2).astype(int)]) # Suppose this was the vector of observed y's

        pyro.clear_param_store()
        pyro.render_model(self.model_YoungDampingDensity, model_args=(input_x, y_obs), render_distributions=True)

        nuts_kernel = HMC(self.model_YoungDampingDensity, step_size=0.0855, num_steps=4)
        #NUTS(self.model_YoungDampingDensity)
        mcmc = MCMC(nuts_kernel, num_samples=150, num_chains=1, warmup_steps=40)        
        mcmc.run(input_x, y_obs)

        # Show summary of inference results
        mcmc.summary()
        posterior_samples = mcmc.get_samples()

        sns.displot(posterior_samples["E"]*10e10)
        plt.xlabel("Young's modulus values")
        plt.show()
                
        sns.displot(posterior_samples["rho"]*8.34e3)
        plt.xlabel("density values")
        plt.show()
        return

if __name__ == "__main__":
 
    files = ["centerFreqResponse"]#, "center2FreqResponse", "randomFreqResponse"]
    files = ["centerFreqResponse"]#, "center2FreqResponse", "randomFreqResponse"]
    freqValues = [[117.5, 122.5, 0.5],\
                  [645.5, 650.5, 0.5],\
                  [1597.5, 1600.5, 0.5],\
                  [2977.5, 2981.5, 0.5]] 
    obj = ComsolProcess("comsol/beam.mph", files, freqValues)
    obj.run()


    
