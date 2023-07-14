import mph
import matplotlib.pyplot as plt

import pyro
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO,TraceGraph_ELBO, MCMC, NUTS, HMC

import torch
from torch.distributions import constraints
import pyro.poutine as poutine
from pyro.poutine.runtime import effectful

import matplotlib.pyplot as plt
import seaborn as sns
import time 

from scipy import stats
import graphviz
import utils
from tqdm import tqdm


import yaml

class ComsolProcess(object):
    def __init__(self, configFile, freqValues):
        # Reading configuration file
        with open(configFile) as f:
            config = yaml.safe_load(f) 
        # initializing variables

        self.inputFiles = config["files"]["dataFile"]
        self.freqValues = freqValues
        
        # Param initialization
        configParam = config["parameters"]
        self.E_mean = configParam["young"]["init"]["mean"]
        self.E_std_init = configParam["young"]["init"]["var"]

        self.rho_mean = configParam["rho"]["init"]["mean"]
        self.rho_std_init = configParam["rho"]["init"]["var"]

        self.eta_mean = configParam["eta"]["init"]["mean"]
        self.eta_std_init = configParam["eta"]["init"]["var"]

        # Starting comsol process
        comsolFilePath = config["files"]["comsolModels"]
        self.studyName = config["comsol"]["studyName"]
        self.evalPoint = config["comsol"]["evaluation"]
        client = mph.start()
        self.modelComsol = client.load(comsolFilePath["training"])
        self.modelComsolFullRange = client.load(comsolFilePath["test"])

        # Bayesian optimizer
        SVIconfig = config["bayesian"]
        self.n_steps = SVIconfig["n_steps"]
        self.lr = SVIconfig["learningRate"]
        if SVIconfig["optimType"] == "ClippedAdam":
            adam_params = {
                "lr": .1, #0.001,
                "betas": (0.96, 0.999),
                "clip_norm": 20.0,
                "lrd": 0.99996,
                "weight_decay": 2.0}
            initial_lr = self.lr
            gamma = SVIconfig["gamma"]  # final learning rate will be gamma * initial_lr
            lrd = gamma ** (2 / SVIconfig["n_steps"])
            self.optim = pyro.optim.ClippedAdam(adam_params)

        elif SVIconfig["optimType"] == "adam":
            adam_params = {"lr": self.lr, 
                    "betas": (0.95, 0.999),
                    "eps": 1e-08}
            self.optim = pyro.optim.adam(adam_params)
        elif SVIconfig["optimType"] == "SGD":
            optimizer = pyro.optim.SGD()
            self.scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': {'lr': 0.01}, 'gamma': 0.1})

        else:
            print("ERROR: Optimizer not valid")
    
    def run(self):

        #freqVal = utils.createComsolVector("lin", self.freqValues, param="step", display=False)
        #freqVal = [105.0, 105.5, 106.0, 106.5, 107.0, 107.5, 108.0, 108.5, 109.0, 109.5, 110.0, 110.5, 111.0, 111.5, 112.0, 112.5, 113.0, 113.5, 114.0, 114.5, 115.0, 632.0, 632.5, 633.0, 633.5, 634.0, 634.5, 635.0, 635.5, 636.0, 636.5, 637.0, 637.5, 638.0, 638.5, 639.0, 639.5, 640.0, 640.5, 641.0, 641.5, 642.0, 1585.5, 1586.0, 1586.5, 1587.0, 1587.5, 1588.0, 1588.5, 1589.0, 1589.5, 1590.0, 1590.5, 1591.0, 1591.5, 1592.0, 1592.5, 1593.0, 1593.5, 1594.0, 1594.5, 1595.0, 1595.5]
        #freqVal = np.concatenate([np.arange(2730, 2770), np.arange(3190, 3210), np.arange(5370, 5400)])
        #freqVal = [1365, 1365.5, 1366, 1366.5, 1367, 1367.5, 1368, 1368.5, 1369, 1369.5, 1370, 1370.5, 1371, 1371.5, 1372, 1372.5, 1373, 1373.5, 1374, 1374.5, 1375, 1375.5, 1376, 1376.5, 1377, 1377.5, 1378, 1378.5, 1379, 1379.5, 1380, 1380.5, 1381, 1381.5, 1382, 1382.5, 1383, 1383.5, 1384, 1384.5, 1595, 1595.5, 1596, 1596.5, 1597, 1597.5, 1598, 1598.5, 1599, 1599.5, 1600, 1600.5, 1601, 1601.5, 1602, 1602.5, 1603, 1603.5, 1604, 1604.5, 2685, 2685.5, 2686, 2686.5, 2687, 2687.5, 2688, 2688.5, 2689, 2689.5, 2690, 2690.5, 2691, 2691.5, 2692, 2692.5, 2693, 2693.5, 2694, 2694.5, 2695, 2695.5, 2696, 2696.5, 2697, 2697.5, 2698, 2698.5, 2699, 2699.5]

    
        #freqindex = [np.array(freqVal*2).astype(int)]
        # Reading input files
        Y_exp = np.array([])
        if type(self.inputFiles) is str:
            self.inputFiles = [self.inputFiles]
        for file in self.inputFiles:
            experiment = pd.read_csv(file)[20:]
            # Mobility value calculated from input data and converted to torch
            mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
            Y_exp = np.append(Y_exp, abs(mobility))
            freq = torch.tensor(experiment["freq"].values)  # Freq values(x axis) converted to torch

        # BAYESIAN INFERENCE PROCESS
        # The experimental data has a resolution of 0.5 Hz. The values selected are integers
        # so the position of the freq. values in the array will be x*2
        #input_x = freq
        input_x = torch.tensor(freq[np.logical_and(mobility>0.2, mobility<0.75)])

        #y_obs = torch.tensor(Y_exp) # Suppose this was the vector of observed y's
        y_obs = torch.tensor(mobility[np.logical_and(mobility>0.2, mobility<0.75)]) # Suppose this was the vector of observed y's
        # the prior information is scaled to the physical values of the parameters
        self.rho_mean_scaled = self.rho_mean
        self.eta_mean_scaled = self.eta_mean
        self.E_mean_scaled = self.E_mean
        
        self.rho_std_scaled = self.rho_std_init*0.1
        self.eta_std_scaled = self.eta_std_init*0.1
        self.E_std_scaled = self.E_std_init*0.1

        pyro.clear_param_store()        
        svi = SVI(self.model, self.guide, self.optim, loss=TraceGraph_ELBO())

        self.losses = np.zeros(self.n_steps)*np.nan
        self.E_values = np.zeros(self.n_steps)
        self.eta_values = np.zeros(self.n_steps)
        self.rho_values = np.zeros(self.n_steps)
        self.step = 0

        
        for step in tqdm(range(self.n_steps)):    
            self.step = step
            loss = svi.step(input_x, y_obs)
            self.losses[step] = loss
            print("---------------")
            #print(self.E_values[step])
            #print(self.rho_values[step])
            print(self.losses[step])
            print(self.eta_values[step])
            print("---------------")
            # updating data values
        
            #print(".", end="")
            #print('[iter {}]  loss: {:.4f}'.format(step, loss))

        self.outputPlots(experiment)
        
        return

    def solveComsol(self, modelComsol, E=1, rho=1, eta=1):#, freq=10):
        # Update parameters
        modelComsol.parameter('Youngs', str(E)+' [Pa]')
        modelComsol.parameter('density', str(rho)+' [kg/m^3]')
        modelComsol.parameter('damping', str(eta))

        # Solving comsol FEM
        print(modelComsol.parameters())
        modelComsol.solve(self.studyName)
        comsolResults = torch.tensor(abs(modelComsol.evaluate(self.evalPoint)))

        return comsolResults

    """
    def physicalScaling(self, rho, eta, E, rho_std, eta_std, E_std):
        rho_var = rho_std*self.rho_std_init
        eta_var = eta_std*self.eta_std_init
        E_var = E_std*self.E_std_init
        rho_var = self.rho_std_init
        eta_var = self.eta_std_init
        E_var = self.E_std_init

        rho_norm = rho*rho_var + self.rho_mean
        eta_norm = eta*eta_var + self.eta_mean
        E_norm = E*E_var + self.E_mean

        return rho_norm, eta_norm, E_norm
    """    


    def physicalScaling(self, rho, eta, E):   

        rho_physical = rho*self.rho_std_scaled + self.rho_mean_scaled
        eta_physical = eta*self.eta_std_scaled + self.eta_mean_scaled
        E_physical = E*self.E_std_scaled + self.E_mean_scaled     

        return rho_physical, eta_physical, E_physical
       
    # @poutine.scale(scale=1.0/601)
    def model(self, x, y_obs):
        # Density definition
        rho_mean = torch.tensor(0.)
        rho_std = torch.tensor(.1)
        rho = pyro.sample("rho", dist.Normal(rho_mean, rho_std))
        # Damping loss factor definition
        eta_mean = torch.tensor(0.)
        eta_std = torch.tensor(.1)
        eta = pyro.sample("eta", dist.Normal(eta_mean, eta_std))
        # Young's modulus definition
        E_mean = torch.tensor(0.)
        E_std = torch.tensor(.1)
        E = pyro.sample("E", dist.Normal(E_mean, E_std))
        
        [rho, eta, E] = self.physicalScaling(rho, eta, E)
        self.E_values[self.step] = E
        self.eta_values[self.step] = eta
        self.rho_values[self.step] = rho
        y_values = self.solveComsol(self.modelComsol, E=E.detach().numpy(), rho=rho.detach().numpy(), eta=eta.detach().numpy())
        
        with pyro.plate("data", len(y_obs)):
            y = pyro.sample("y", dist.Normal(y_values, 0.13), obs=y_obs)
        return y

    # @poutine.scale(scale=1.0/601)
    def guide(self, x, y_obs):
        # Density guide
        rho_mean_q = pyro.param("rho_mean_guide", torch.tensor(0.))
        rho_std_q = pyro.param("rho_std_guide", torch.tensor(0.001), constraint=constraints.positive)
        pyro.sample("rho", dist.Normal(rho_mean_q, rho_std_q))
        # Damping loss factor guide
        eta_mean_q = pyro.param("eta_mean_guide", torch.tensor(0.))
        eta_std_q = pyro.param("eta_std_guide", torch.tensor(0.001), constraint=constraints.positive)
        pyro.sample("eta", dist.Normal(eta_mean_q, eta_std_q))

        # Damping loss factor guide
        E_mean_q = pyro.param("E_mean_guide", torch.tensor(0.))
        E_std_q = pyro.param("E_std_guide", torch.tensor(0.001), constraint=constraints.positive)
        pyro.sample("E", dist.Normal(E_mean_q, E_std_q))  
    """
    @poutine.scale(scale=1.0/20)
    def model(self, x, y_obs):
        # Damping loss factor definition
        eta_mean = pyro.param("eta_mean", torch.tensor(0.))
        eta_std = pyro.param("eta_std", torch.tensor(1.), constraint=constraints.positive)
        eta = pyro.sample("eta", dist.Normal(eta_mean, eta_std))
        [rho, eta, E] = self.physicalScaling(0, eta, 0)
        self.eta_values[self.step] = eta
        y_values = self.solveComsol(self.modelComsol, E=9.9e10, rho=8300, eta=eta.detach().numpy())
            
        with pyro.plate("data", len(y_obs)):
            y = pyro.sample("y", dist.Normal(20*torch.log10(y_values), 0.001), obs=20*torch.log10(y_obs))
        return y

    @poutine.scale(scale=1.0/20)
    def guide(self, x, y_obs):
        # Damping loss factor guide
        eta_mean_q = pyro.param("eta_mean_guide", torch.tensor(0.))
        eta_std_q = pyro.param("eta_std_guide", torch.tensor(0.05), constraint=constraints.positive)
        pyro.sample("eta", dist.Normal(eta_mean_q, eta_std_q))

    
    """   
    def outputPlots(self, experiment):
        E_est = pyro.param("E_mean_guide").item()
        E_std = pyro.param("E_std_guide").item()

        rho_est = pyro.param("rho_mean_guide").item()
        rho_std = pyro.param("rho_std_guide").item()

        eta_est = pyro.param("eta_mean_guide").item()
        eta_std = pyro.param("eta_std_guide").item()

        plt.figure(10)
        plt.plot(np.linspace(0, self.n_steps, len(self.losses)), self.losses, "*-")
        #plt.yscale("log")
        plt.xlabel("iterations ")
        plt.ylabel(" Error estimation")
        plt.savefig("./figuresResultsComsol/ErrorElboNO"+str(self.lr).split(".")[-1]+"_"+str(self.n_steps)+"_all_decay.png")
        
        print("Calculating mobility FULL RANGE")
        rho_est, eta_est, E_est = self.physicalScaling(rho_est, eta_est, E_est)
        comsolResults_dB = self.solveComsol(self.modelComsol, E=E_est, rho=rho_est, eta=eta_est)
        #self.modelComsolFullRange.solve("Study 1")
        #comsolResults_dB = 20*torch.log10(torch.tensor(abs(self.modelComsolFullRange.evaluate('comp1.point1'))))

        mobility = abs(experiment["disp"].values)
        mob_dB = 20*np.log10(mobility)

        plt.figure(11)
        freq = np.linspace(50, 300, 501)
        plt.plot(experiment["freq"].values, mob_dB, label="experiment")
        plt.plot(freq, comsolResults_dB, "--", label="estimated")
        plt.xlabel("Frequency / Hz")
        plt.ylabel("Mobility / dB")
        plt.legend()
        plt.savefig("./figuresResultsComsol/mobNO_"+str(lr).split(".")[-1]+"_"+str(self.n_steps)+"LAST_all_decay.png")


        
        # ETA DISTRIBUTION PLOT
        plt.figure(12)
        sigma = np.sqrt(variance)
        x_init = np.linspace(-3*sigma*self.eta_std_init, 3*sigma*self.eta_std_init, 100) + self.eta_mean
        plt.plot(x_init, stats.norm.pdf(x_init, self.eta_mean, sigma*self.eta_std_init), label="prior")

        eta_var = pyro.param("eta_std").item()
        sigma = np.sqrt(eta_var)
        x = np.linspace(-3*self.eta_std_init*eta_std, 3*self.eta_std_init*eta_std, 100)+ eta_est
        plt.plot(x, stats.norm.pdf(x, eta_est, self.eta_std_init*eta_std), label="posterior")
        plt.xlabel("Damping loss factors")
        plt.legend()
        plt.savefig("./figuresResultsComsol/ETA_"+str(lr).split(".")[-1]+"_"+str(n_steps)+"LAST_all_decay.png")

        # RHO DISTRIBUTION PLOT
        plt.figure(13)
        variance = 1
        sigma = np.sqrt(variance)
        x_init = np.linspace(-3*sigma*self.rho_std_init, 3*sigma*self.rho_std_init, 100) + self.rho_mean
        plt.plot(x_init, stats.norm.pdf(x_init, self.rho_mean, sigma*self.rho_std_init), label="prior")

        x = np.linspace(-3*self.rho_std_init*rho_std, 3*self.rho_std_init*rho_std, 100)+ rho_est
        plt.plot(x, stats.norm.pdf(x, rho_est, self.rho_std_init*rho_std), label="posterior")
        plt.xlabel("Density / Kg/m^3")
        plt.legend()
        plt.savefig("./figuresResultsComsol/RHO_"+str(lr).split(".")[-1]+"_"+str(n_steps)+"LAST_all_decay.png")

        # E DISTRIBUTION PLOT
        plt.figure(14)
        variance = 1
        sigma = np.sqrt(variance)
        x_init = np.linspace(-3*sigma*self.E_std_init, 3*sigma*self.E_std_init, 100) + self.E_mean
        plt.plot(x_init, stats.norm.pdf(x_init, self.E_mean, sigma*self.E_std_init), label="prior")

        sigma = np.sqrt(E_std)
        x = np.linspace(-3*self.E_std_init*E_std, 3*self.E_std_init*E_std, 100)+ E_est
        plt.plot(x, stats.norm.pdf(x, E_est, self.E_std_init*E_std), label="posterior")
        plt.xlabel("Young's modulus / Pa")
        plt.legend()
        plt.savefig("./figuresResultsComsol/E_"+str(lr).split(".")[-1]+"_"+str(n_steps)+"LAST_all_decay.png")
        plt.show()



if __name__ == "__main__":
 
    configFilePath = "./configTest.yaml"  

    freqValues = [1365, 1365.5, 1366, 1366.5, 1367, 1367.5, 1368, 1368.5, 1369, 1369.5, 1370, 1370.5, 1371, 1371.5, 1372, 1372.5, 1373, 1373.5, 1374, 1374.5, 1375, 1375.5, 1376, 1376.5, 1377, 1377.5, 1378, 1378.5, 1379, 1379.5, 1380, 1380.5, 1381, 1381.5, 1382, 1382.5, 1383, 1383.5, 1384, 1384.5, 1595, 1595.5, 1596, 1596.5, 1597, 1597.5, 1598, 1598.5, 1599, 1599.5, 1600, 1600.5, 1601, 1601.5, 1602, 1602.5, 1603, 1603.5, 1604, 1604.5, 2685, 2685.5, 2686, 2686.5, 2687, 2687.5, 2688, 2688.5, 2689, 2689.5, 2690, 2690.5, 2691, 2691.5, 2692, 2692.5, 2693, 2693.5, 2694, 2694.5, 2695, 2695.5, 2696, 2696.5, 2697, 2697.5, 2698, 2698.5, 2699, 2699.5]
    #freqValues = np.linspace(50, 350, 601)
    obj = ComsolProcess(configFilePath, freqValues)
    obj.run()


    
