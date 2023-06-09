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
        self.rho_theo = 8976
        self.eta_theo = 0.01

        self.Ehigh = 10e11
        self.Elow = 10e9

        self.rhohigh = 8800
        self.rholow = 7300

        self.etahigh = 0.5
        self.etalow = 0.001  

        
        self.E_mean=10.0e10
        self.E_var_init = 5e9
        self.rho_mean=8050.0
        self.rho_var_init = 250
        self.eta_mean=0.00505
        self.eta_var_init = 0.002 
        # Starting comsol process
        client = mph.start()
        self.model = client.load(comsolModelPath)


    def updateParams(self, model, E=1, rho=1, eta=1):#, freq=10):
        model.parameter('Youngs', str(E)+' [Pa]')
        model.parameter('density', str(rho)+' [kg/m^3]')
        model.parameter('damping', str(eta)+' [Pa]')
        #model.parameter('freq', str(freq)+' [Hz]')


    def normalization(self, rho, eta, E, rho_var, eta_var, E_var):

        rho_var = rho_var*self.rho_var_init
        eta_var = eta_var*self.eta_var_init
        E_var = E_var*self.E_var_init

        rho_norm = rho*rho_var + self.rho_mean
        eta_norm = eta*eta_var + self.eta_mean
        E_norm = E*E_var + self.E_theo

        return rho_norm, eta_norm, E_norm

    def model_YoungDampingDensity(self, x, y_obs):
        # Density definition
        rho_mean = pyro.param("rho_mean", dist.Normal(0, 1))
        rho_std = pyro.param("rho_std", torch.tensor(1), constraint=constraints.positive)
        rho = pyro.sample("rho", dist.Normal(rho_mean, rho_std))
        # Damping loss factor definition
        eta_mean = pyro.param("eta_mean", dist.Normal(0, 1))
        eta_std = pyro.param("eta_std", torch.tensor(1), constraint=constraints.positive)
        eta = pyro.sample("eta", dist.Normal(eta_mean, eta_std))
        # Young's modulus definition
        E_mean = pyro.param("E_mean", dist.Normal(0, 1))
        E_std = pyro.param("E_std", torch.tensor(1), constraint=constraints.positive)
        E = pyro.sample("E", dist.Normal(E_mean, E_std))
        
        rho, eta, E = self.normalization(rho, eta, E, rho_std, eta_std, E_std)

        with pyro.plate("data", len(y_obs)):            
            #updateParam(model, young=E.detach().numpy(), rho=rho.detach().numpy(), eta=eta.detach().numpy(), freq=x[i])
            self.updateParams(self.model, young=E.detach().numpy(), rho=rho.detach().numpy(), eta=eta.detach().numpy())
            self.model.solve("Study 3")
            comsolResults = abs(self.model.evaluate('comp1.point1'))
            y = pyro.sample("y", dist.Normal(20*torch.log10(comsolResults), 0.001), obs=20*torch.log10(y_obs))
        return y

    def guide(self, x, y_obs):
        # Density guide
        rho_mean = pyro.param("rho_mean", dist.Normal(0, 1))
        rho_std = pyro.param("rho_std", torch.tensor(1), constraint=constraints.positive)
        pyro.sample("rho", dist.Normal(rho_mean, rho_std))

        # Damping loss factor guide
        eta_mean = pyro.param("eta_mean", dist.LogNormal(0, 1))
        eta_std = pyro.param("eta_std", torch.tensor(1), constraint=constraints.positive)
        pyro.sample("eta", dist.Normal(eta_mean, eta_std))

        # Damping loss factor guide
        E_mean = pyro.param("E_mean", dist.Normal(0, 1))
        E_std = pyro.param("E_std", torch.tensor(1), constraint=constraints.positive)
        pyro.sample("E", dist.Normal(E_mean, E_std))


    def run(self):

        #freqVal = utils.createComsolVector("lin", self.freqValues, param="step", display=False)
        freqVal = [115.0, 115.5, 116.0, 116.5, 117.0, 117.5, 118.0, 118.5, 119.0, 119.5, 120.0, 120.5, 121.0, 121.5, 122.0, 122.5, 123.0, 123.5, 124.0, 124.5, 642.0, 642.5, 643.0, 643.5, 644.0, 644.5, 645.0, 645.5, 646.0, 646.5, 647.0, 647.5, 648.0, 648.5, 649.0, 649.5, 650.0, 650.5, 651.0, 651.5, 1595.5, 1596.0, 1596.5, 1597.0, 1597.5, 1598.0, 1598.5, 1599.0, 1599.5, 1600.0, 1600.5, 1601.0, 1601.5, 1602.0, 1602.5, 1603.0, 1603.5, 1604.0, 1604.5, 1605.0, 2974.5, 2975.0, 2975.5, 2976.0, 2976.5, 2977.0, 2977.5, 2978.0, 2978.5, 2979.0, 2979.5, 2980.0, 2980.5, 2981.0, 2981.5, 2982.0, 2982.5, 2983.0, 2983.5, 2984.0]
        freqVal = np.array(freqVal)
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
        pyro.render_model(self.zzzmodel_YoungDampingDensity, model_args=(input_x, y_obs), render_distributions=True)

        #nuts_kernel = HMC(self.model_YoungDampingDensity, step_size=0.00001, num_steps=4)
        nuts_kernel = NUTS(self.zzzmodel_YoungDampingDensity)
        mcmc = MCMC(nuts_kernel, num_samples=200, num_chains=1, warmup_steps=50)        
        mcmc.run(input_x, y_obs)

        # Show summary of inference results
        mcmc.summary()
        posterior_samples = mcmc.get_samples()

        sns.displot(posterior_samples["E"]*10e10)
        plt.xlabel("Young's modulus values")
        plt.show()
                
        sns.displot(posterior_samples["rho"]*8976)
        plt.xlabel("density values")
        plt.show()

        sns.displot(posterior_samples["eta"]*0.01)
        plt.xlabel("eta / ")
        plt.show()
        return

if __name__ == "__main__":
 
    files = ["centerFreqResponse"]#, "center2FreqResponse", "randomFreqResponse"]
    files = ["centerFreqResponse"]#, "center2FreqResponse", "randomFreqResponse"]
    freqValues = [[117.5, 122.5, 0.5],\
                  [645.5, 650.5, 0.5],\
                  [1597.5, 1600.5, 0.5],\
                  [2977.5, 2981.5, 0.5]] 
    freqValues = [[60, 600, 1]]
    obj = ComsolProcess("comsol/beam.mph", files, freqValues)
    obj.run()


    
