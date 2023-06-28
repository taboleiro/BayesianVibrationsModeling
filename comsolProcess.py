import mph
import matplotlib.pyplot as plt

import pyro
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC, NUTS, HMC

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
        self.E_var_init = configParam["young"]["init"]["var"]

        self.rho_mean = configParam["rho"]["init"]["mean"]
        self.rho_var_init = configParam["rho"]["init"]["var"]

        self.eta_mean = configParam["eta"]["init"]["mean"]
        self.eta_var_init = configParam["eta"]["init"]["var"]

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
        if SVIconfig["optimType"] == "ClippedAdam":
            initial_lr = SVIconfig["learningRate"]
            gamma = SVIconfig["gamma"]  # final learning rate will be gamma * initial_lr
            lrd = gamma ** (1 / SVIconfig["n_steps"])
            self.optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})

        elif SVIconfig["optimType"] == "adam":
            adam_params = {"lr": SVIconfig["learningRate"], 
                    "betas": (0.95, 0.999),
                    "eps": 1e-08}
            self.optim = pyro.optim.adam(adam_params)
        else:
            print("ERROR: Optimizer not valid")
    
    def run(self):

        #freqVal = utils.createComsolVector("lin", self.freqValues, param="step", display=False)
        #freqVal = [105.0, 105.5, 106.0, 106.5, 107.0, 107.5, 108.0, 108.5, 109.0, 109.5, 110.0, 110.5, 111.0, 111.5, 112.0, 112.5, 113.0, 113.5, 114.0, 114.5, 115.0, 632.0, 632.5, 633.0, 633.5, 634.0, 634.5, 635.0, 635.5, 636.0, 636.5, 637.0, 637.5, 638.0, 638.5, 639.0, 639.5, 640.0, 640.5, 641.0, 641.5, 642.0, 1585.5, 1586.0, 1586.5, 1587.0, 1587.5, 1588.0, 1588.5, 1589.0, 1589.5, 1590.0, 1590.5, 1591.0, 1591.5, 1592.0, 1592.5, 1593.0, 1593.5, 1594.0, 1594.5, 1595.0, 1595.5]
        freqVal = np.concatenate([np.arange(2730, 2770), np.arange(3190, 3210), np.arange(5370, 5400)])
        #freqVal = np.array(freqVal)
        # Reading input files
        Y_exp = np.array([])
        if type(self.inputFiles) is str:
            self.inputFiles = [self.inputFiles]
        for file in self.inputFiles:
            experiment = pd.read_csv(file)
            # Mobility value calculated from input data and converted to torch
            mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
            Y_exp = np.append(Y_exp, abs(mobility))
            freq = torch.tensor(experiment["freq"].values) # Freq values(x axis) converted to torch

        # BAYESIAN INFERENCE PROCESS
        # The experimental data has a resolution of 0.5 Hz. The values selected are integers
        # so the position of the freq. values in the array will be x*2
        #input_x = torch.tensor(freq[(freqVal*2).astype(int)])
        input_x = torch.tensor(freq[(freqVal).astype(int)])

        #y_obs = torch.tensor(20*np.log10(Y_exp[(freqVal*2).astype(int)])) # Suppose this was the vector of observed y's
        y_obs = torch.tensor(20*np.log10(Y_exp[(freqVal).astype(int)])) # Suppose this was the vector of observed y's

        pyro.clear_param_store()        
        svi = SVI(self.model, self.guide, self.optim, loss=TraceMeanField_ELBO())

        losses = []
        for step in tqdm(range(self.n_steps)):
            loss = svi.step(input_x, y_obs)
            losses.append(loss)
                #print(".", end="")
                #print('[iter {}]  loss: {:.4f}'.format(step, loss))

        self.outputPlots(losses, experiment)
        
        return

    def solveComsol(self, modelComsol, E=1, rho=1, eta=1):#, freq=10):
        # Update parameters
        modelComsol.parameter('Youngs', str(E)+' [Pa]')
        modelComsol.parameter('density', str(rho)+' [kg/m^3]')
        modelComsol.parameter('damping', str(eta)+' [Pa]')

        # Solving comsol FEM
        self.modelComsol.solve(self.studyName)
        comsolResults = 20*torch.log10(torch.tensor(abs(self.modelComsol.evaluate(self.evalPoint))))

        return comsolResults


    def normalization(self, rho, eta, E, rho_var, eta_var, E_var):

        rho_var = rho_var*self.rho_var_init
        eta_var = eta_var*self.eta_var_init
        E_var = E_var*self.E_var_init

        rho_norm = rho*rho_var + self.rho_mean
        eta_norm = eta*eta_var + self.eta_mean
        E_norm = E*E_var + self.E_mean

        return rho_norm, eta_norm, E_norm

    @poutine.scale(scale=1.0/63)  
    def model(self, x, y_obs):
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
            comsolResults = self.solveComsol(self.modelComsol, E=E.detach().numpy(), rho=rho.detach().numpy(), eta=eta.detach().numpy())
            y = pyro.sample("y", dist.Normal(comsolResults, 0.001), obs=y_obs)
        return y


    @poutine.scale(scale=1.0/63)
    def guide(self, x, y_obs):
        # Density guide
        rho_mean = pyro.param("rho_mean", dist.Normal(0, 1))
        rho_std = pyro.param("rho_std", torch.tensor(0.05), constraint=constraints.positive)
        pyro.sample("rho", dist.Normal(rho_mean, rho_std))

        # Damping loss factor guide
        eta_mean = pyro.param("eta_mean", dist.LogNormal(0, 1))
        eta_std = pyro.param("eta_std", torch.tensor(0.05), constraint=constraints.positive)
        pyro.sample("eta", dist.Normal(eta_mean, eta_std))

        # Damping loss factor guide
        E_mean = pyro.param("E_mean", dist.Normal(0, 1))
        E_std = pyro.param("E_std", torch.tensor(0.05), constraint=constraints.positive)
        pyro.sample("E", dist.Normal(E_mean, E_std))

    def outputPlots(self, losses, experiment):
        E_est = pyro.param("E_mean").item()
        E_std = pyro.param("E_std").item()

        rho_est = pyro.param("rho_mean").item()
        rho_std = pyro.param("rho_std").item()

        eta_est = pyro.param("eta_mean").item()
        eta_std = pyro.param("eta_std").item()

        plt.figure(10)
        plt.plot(np.linspace(0, self.n_steps, len(losses)), np.log10(losses), "*-")
        #plt.yscale("log")
        plt.xlabel("iterations ")
        plt.ylabel(" Error estimation")
        plt.savefig("./figuresResultsComsol/ErrorElboNO"+str(self.lr).split(".")[-1]+"_"+str(self.n_steps)+"_all_decay.png")
        
        print("Calculating mobility FULL RANGE")
        rho_est, eta_est, E_est = self.normalization(rho_est, eta_est, E_est, rho_std, eta_std, E_std)
        comsolResults_dB = self.solveComsol(self.modelComsolFullRange, E=E_est, rho=rho_est, eta=eta_est)
        #self.modelComsolFullRange.solve("Study 1")
        #comsolResults_dB = 20*torch.log10(torch.tensor(abs(self.modelComsolFullRange.evaluate('comp1.point1'))))

        mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
        mob_dB = 20*np.log10(mobility)

        plt.figure(11)
        freq = np.arange(50, 3000, 1)
        plt.plot(experiment["freq"].values, mob_dB, label="experiment")
        plt.plot(freq, comsolResults_dB[:-1], "--", label="estimated")
        plt.xlabel("Frequency / Hz")
        plt.ylabel("Mobility / dB")
        plt.legend()
        plt.savefig("./figuresResultsComsol/mobNO_"+str(lr).split(".")[-1]+"_"+str(self.n_steps)+"LAST_all_decay.png")


        
        # ETA DISTRIBUTION PLOT
        plt.figure(12)
        sigma = np.sqrt(variance)
        x_init = np.linspace(-3*sigma*self.eta_var_init, 3*sigma*self.eta_var_init, 100) + self.eta_mean
        plt.plot(x_init, stats.norm.pdf(x_init, self.eta_mean, sigma*self.eta_var_init), label="prior")

        eta_var = pyro.param("eta_std").item()
        sigma = np.sqrt(eta_var)
        x = np.linspace(-3*self.eta_var_init*eta_std, 3*self.eta_var_init*eta_std, 100)+ eta_est
        plt.plot(x, stats.norm.pdf(x, eta_est, self.eta_var_init*eta_std), label="posterior")
        plt.xlabel("Damping loss factors")
        plt.legend()
        plt.savefig("./figuresResultsComsol/ETA_"+str(lr).split(".")[-1]+"_"+str(n_steps)+"LAST_all_decay.png")

        # RHO DISTRIBUTION PLOT
        plt.figure(13)
        variance = 1
        sigma = np.sqrt(variance)
        x_init = np.linspace(-3*sigma*self.rho_var_init, 3*sigma*self.rho_var_init, 100) + self.rho_mean
        plt.plot(x_init, stats.norm.pdf(x_init, self.rho_mean, sigma*self.rho_var_init), label="prior")

        x = np.linspace(-3*self.rho_var_init*rho_std, 3*self.rho_var_init*rho_std, 100)+ rho_est
        plt.plot(x, stats.norm.pdf(x, rho_est, self.rho_var_init*rho_std), label="posterior")
        plt.xlabel("Density / Kg/m^3")
        plt.legend()
        plt.savefig("./figuresResultsComsol/RHO_"+str(lr).split(".")[-1]+"_"+str(n_steps)+"LAST_all_decay.png")

        # E DISTRIBUTION PLOT
        plt.figure(14)
        variance = 1
        sigma = np.sqrt(variance)
        x_init = np.linspace(-3*sigma*self.E_var_init, 3*sigma*self.E_var_init, 100) + self.E_mean
        plt.plot(x_init, stats.norm.pdf(x_init, self.E_mean, sigma*self.E_var_init), label="prior")

        sigma = np.sqrt(E_std)
        x = np.linspace(-3*self.E_var_init*E_std, 3*self.E_var_init*E_std, 100)+ E_est
        plt.plot(x, stats.norm.pdf(x, E_est, self.E_var_init*E_std), label="posterior")
        plt.xlabel("Young's modulus / Pa")
        plt.legend()
        plt.savefig("./figuresResultsComsol/E_"+str(lr).split(".")[-1]+"_"+str(n_steps)+"LAST_all_decay.png")
        plt.show()



if __name__ == "__main__":
 
    configFilePath = "./config.yaml"  

    freqValues = [105.0, 105.5, 106.0, 106.5, 107.0, 107.5, 108.0, 108.5, 109.0, 109.5, 110.0, 110.5, 111.0, 111.5, 112.0, 112.5, 113.0, 113.5, 114.0, 114.5, 115.0, 632.0, 632.5, 633.0, 633.5, 634.0, 634.5, 635.0, 635.5, 636.0, 636.5, 637.0, 637.5, 638.0, 638.5, 639.0, 639.5, 640.0, 640.5, 641.0, 641.5, 642.0, 1585.5, 1586.0, 1586.5, 1587.0, 1587.5, 1588.0, 1588.5, 1589.0, 1589.5, 1590.0, 1590.5, 1591.0, 1591.5, 1592.0, 1592.5, 1593.0, 1593.5, 1594.0, 1594.5, 1595.0, 1595.5]
    obj = ComsolProcess(configFilePath, freqValues)
    obj.run()


    
