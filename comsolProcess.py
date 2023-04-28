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
        # Starting comsol process
        client = mph.start()
        self.model = client.load(comsolModelPath)


    def updateParams(self, model, E=1, rho=1, eta=1):#, freq=10):
        E = E*self.E_theo
        rho = rho*self.rho_theo
        eta = eta*self.eta_theo# E lim

        if E < self.Elow: 
            E = self.Elow
        elif E > self.Ehigh:
            E = self.Ehigh
        
        # rho lim
        if rho < self.rholow: 
            rho = self.rholow
        elif rho > self.rhohigh:
            rho = self.rhohigh
        
        # eta lim
        if eta < self.etalow: 
            eta = self.etalow
        elif eta > self.etahigh:
            eta = self.etahigh

        print("E: "+str(E))
        print("rho: "+str(rho))
        print("eta: "+str(eta))
        model.parameter('Youngs', str(E)+' [Pa]')
        model.parameter('density', str(rho)+' [kg/m^3]')
        model.parameter('damping', str(eta)+' [Pa]')
        #model.parameter('freq', str(freq)+' [Hz]')


    def zzzmodel_YoungDampingDensity(self, x, y_obs):
        # Density definition
        #rho_mean = pyro.param("rho_mean", dist.Normal(1., 0.1), constraint=constraints.real)
        #rho_var = pyro.param("rho_var", torch.tensor(0.5))
        #rho_var = pyro.param("rho_var", dist.Cauchy(1., 0.5), constraint=constraints.positive)
        #rho_var = pyro.param("rho_var", dist.Cauchy(1., 0.), constraint=constraints.positive)
        #rho = pyro.sample("rho", dist.Normal(rho_mean, rho_var))
        rho = pyro.sample("rho", dist.Normal(1, 0.5))
        # Damping loss factor definition
        #eta_mean = pyro.param("eta_mean", dist.Normal(1., .5), constraint=constraints.positive)
        #eta_var = pyro.param("eta_var", torch.tensor(0.5))
        #eta_var = pyro.param("eta_var", dist.Cauchy(1., .5))
        #eta = pyro.sample("eta", dist.Normal(eta_mean, eta_var))
        eta = pyro.sample("eta", dist.Normal(1, 0.20))
        # Young's modulus definition
        #E_mean = pyro.param("E_mean", dist.Normal(1, .001), constraint=constraints.positive)
        #E_var = pyro.param("E_var", torch.tensor(0.2))
        #E_var = pyro.param("E_var", dist.Cauchy(1., 0.5), constraint=constraints.positive)
        #E = pyro.sample("E", dist.Normal(E_mean, E_var))
        E = pyro.sample("E", dist.Uniform(0.1, 0.2))
        # Since the studio is done frequcuency by frequency, the loop can't be vectorized like: "with pyro.plate......"
        y = torch.zeros(len(y_obs))

        

        with pyro.plate("data", len(y_obs)):
            self.updateParams(self.model, E=E.detach().numpy(), rho=rho.detach().numpy(), eta=eta.detach().numpy())
            self.model.solve("Study 3")
            comsolResults = abs(self.model.evaluate('comp1.point1'))
            y = pyro.sample("y", dist.Normal(torch.tensor(comsolResults), 1), obs=y_obs)
        return y



    def model_YoungDampingDensity(self, x, y_obs):
                # Density definition
        rho_mean = pyro.param("rho_mean", dist.Normal(1, .05), constraint=constraints.positive)
        rho_var = pyro.param("rho_var", dist.Cauchy(1., 0.))
        rho = pyro.sample("rho", dist.Normal(1, .05).mask((0.7 <= x) & (x <= 1.3)))
        # Damping loss factor definition
        eta_mean = pyro.param("eta_mean", dist.Normal(1, 3.), constraint=constraints.positive)
        eta_var = pyro.param("eta_var", dist.Cauchy(1., 0.))
        eta = pyro.sample("eta", dist.Normal(1, .5).mask((0.5 <= x) & (x <= 1.5)))
        # Young's modulus definition
        E_mean = pyro.param("E_mean", dist.Normal(0.99, .05), constraint=constraints.positive)
        E_var = pyro.param("E_var", dist.Cauchy(1., 0.))
        E = pyro.sample("E", dist.Normal(1, .05).mask((0.9 <= x) & (x <= 1.1)))
        with pyro.plate("data", len(y_obs)):
            #updateParam(model, young=E.detach().numpy(), rho=rho.detach().numpy(), eta=eta.detach().numpy(), freq=x[i])
            self.updateParams(self.model, young=E.detach().numpy(), rho=rho.detach().numpy(), eta=eta.detach().numpy())
            self.model.solve("Study 3")
            comsolResults = abs(self.model.evaluate('comp1.point1'))
            y = pyro.sample("y", dist.Normal(comsolResults, 1.), obs=y_obs)
            #plt.plot(x, y_obs)
            #plt.plot(x, y)
        return y

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


    
