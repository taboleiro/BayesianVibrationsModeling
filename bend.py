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

import mph
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
        client = mph.start()
        self.modelComsol = client.load("./comsol/TestComsol.mph")
        self.studyName = "Study 2"
        self.evalPoint = "comp1.point1"

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
            experiment = pd.read_csv("./Data/bend/"+file+".csv")[20:]
            # Mobility value calculated from input data and converted to torch
            mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
            self.Y_exp = mobility
            # self.Y_exp.append(abs(mobility))#[self.freqVal*2]))
            #self.Y_exp_norm = (self.Y_exp - self.Y_exp.mean()) / self.Y_exp.std() # Normalization
            self.freq = torch.tensor(experiment["freq"].values) # Freq values(x axis) converted to torch
        #self.Y_exp = pd.read_csv("./BayesianVibrationsModeling/Data/bend/"+file+".csv")[20:]
        #self.Y_exp = torch.tensor(self.Y_exp)
        
        #self.freq = torch.tensor(self.freq[mobility>0.1])
        #self.Y_exp = torch.tensor(mobility[mobility>0.1])
        freqVal = [115.0, 115.5, 116.0, 116.5, 117.0, 117.5, 118.0, 118.5, 119.0, 119.5, 120.0, 120.5, 121.0, 121.5, 122.0, 122.5, 123.0, 123.5, 124.0, 124.5, 642.0, 642.5, 643.0, 643.5, 644.0, 644.5, 645.0, 645.5, 646.0, 646.5, 647.0, 647.5, 648.0, 648.5, 649.0, 649.5, 650.0, 650.5, 651.0, 651.5, 1595.5, 1596.0, 1596.5, 1597.0, 1597.5, 1598.0, 1598.5, 1599.0, 1599.5, 1600.0, 1600.5, 1601.0, 1601.5, 1602.0, 1602.5, 1603.0, 1603.5, 1604.0, 1604.5, 1605.0, 2974.5, 2975.0, 2975.5, 2976.0, 2976.5, 2977.0, 2977.5, 2978.0, 2978.5, 2979.0, 2979.5, 2980.0, 2980.5, 2981.0, 2981.5, 2982.0, 2982.5, 2983.0, 2983.5, 2984.0]
        freqVal = np.array(freqVal)
        self.freq = torch.tensor(self.freq[(freqVal*2).astype(int)])
        self.Y_exp = torch.tensor(mobility[(freqVal*2).astype(int)]) # Suppose this was the vector of observed y's

        #y_obs = torch.tensor(Y_exp) # Suppose this was the vector of observed y's
        #y_obs = torch.tensor(mobility[np.logical_and(mobility>0.2, mobility<0.75)]) # Suppose this was the vector of observed y's
        self.train()
        #map_estimate = pyro.param("E").item()
        #print("Our MAP estimate of the Young's modulus is {:.3f}".format(map_estimate)) 

    def normalization(self, E, rho, eta):
        E_theo=torch.tensor(10e10)
        E_var_init =torch.tensor(5.0e9)
        rho_theo=torch.tensor(8000.0)
        rho_var_init =torch.tensor(250.0)
        eta_mean=torch.tensor(0.00505)
        eta_var_init =torch.tensor( 0.006)
        
        rho_norm = rho*rho_var_init + rho_theo
        eta_norm = eta*eta_var_init + eta_mean
        E_norm = E*E_var_init + E_theo

        return E_norm, rho_norm, eta_norm

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
        return torch.sgn(Y)

    def model_YoungDampingDensity(self, x, y_obs):
        # Density definition
        #rho_mean = pyro.param("rho_mean", dist.Normal(1, .25), constraint=constraints.positive)
        #rho_var = pyro.param("rho_var", dist.Cauchy(1., 0.))
        rho = pyro.sample("rho", dist.Normal(0, .1))
        #rho = pyro.sample("rho", dist.Uniform(self.rholow, self.rhohigh))
        #rho = pyro.sample("rho", dist.Beta(1, 1))
        # Damping loss factor definition
        #eta_mean = pyro.param("eta_mean", dist.Normal(1, 3.), constraint=constraints.positive)
        #eta_var = pyro.param("eta_var", dist.Cauchy(.1, 0.))
        eta = pyro.sample("eta", dist.Normal(0, .1))
        #eta = pyro.sample("eta", dist.Beta(1, 1))
        # Young's modulus definition
        #E_mean = pyro.param("E_mean", dist.Normal(0.99, .25), constraint=constraints.positive)
        #E_var = pyro.param("E_var", dist.Cauchy(.1, 0.))
        E = pyro.sample("E", dist.Normal(0, .1))
        #E = pyro.sample("E", dist.Beta(1, 1))

        # E lim
        """
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
        """
        E, rho, eta = self.normalization(E, rho, eta)
        y_values = self.mobilityFuncModel(E, x, rho=rho ,eta=eta)
        #y_values = self.solveComsol(self.modelComsol, E=E.detach().numpy(), rho=rho.detach().numpy(), eta=eta.detach().numpy())
        with pyro.plate("data", len(y_obs)):
            y = pyro.sample("y", dist.Normal(y_values, 0.01), obs=y_obs)
        return y


    def solveComsol(self, modelComsol, E=1, rho=1, eta=1):#, freq=10):
        # Update parameters
        modelComsol.parameter('youngs', str(E)+' [Pa]')
        modelComsol.parameter('density', str(rho)+' [kg/m^3]')
        modelComsol.parameter('eta', str(eta))

        # Solving comsol FEM
        print(modelComsol.parameters())
        modelComsol.solve(self.studyName)
        comsolResults = torch.tensor(abs(modelComsol.evaluate(self.evalPoint)))

        return comsolResults

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
            y = pyro.sample("y", dist.Normal(y_values, 0.01), obs=y_obs)
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
        
        E, rho, eta = self.normalization(posterior_samples["E"].mean(), posterior_samples["rho"].mean(), posterior_samples["eta"].mean())
        mob = self.mobilityFuncModel(E, input_x, rho=rho ,eta=eta)
        
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
