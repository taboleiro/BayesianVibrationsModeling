import numpy as np
import pandas as pd
from statistics import mean, stdev
import mph

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import scipy.io as sio
from pyro.poutine.runtime import effectful

import torch
from torch.distributions import constraints
from torch.autograd import Variable
import yaml
from pymcmcstat.MCMC  import MCMC 

import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import time 

from scipy import stats
import graphviz
from tqdm import tqdm

class ComsolProcess(object):
    def __init__(self, configFile):
        # initializing variables
        with open(configFile) as f:
            config = yaml.safe_load(f) 
        # initializing variables

        self.inputFiles = config["files"]["dataFile"]
        if len(self.inputFiles) > 1:
            self.data = sio.loadmat(self.inputFiles[0])
            self.ref = sio.loadmat(self.inputFiles[1])
        else:
            self.data = pd.read_csv(self.inputFiles)

        self.freqValues = config["freqValues"]
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
        # Starting comsol process
        client = mph.start()
        self.model = client.load(comsolFilePath["test"])
        self.comsolModelFullRange = client.load(comsolFilePath["training"])
        self.error = np.array([])


    def solveComsol(self, modelComsol, param):#, freq=10):
        
        # Update parameters
        E, rho, eta = param
        rho, eta, E = self.normalization(rho, eta, E)
        modelComsol.parameter('youngs', str(E)+' [Pa]')
        modelComsol.parameter('density', str(rho)+' [kg/m^3]')
        modelComsol.parameter('eta', str(eta))


        # Solving comsol FEM
        modelComsol.solve("Study 1")
        #comsolResults1 = torch.tensor(modelComsol.evaluate("comp1.point2"))
        meas = modelComsol.evaluate("comp1.point2")
        ref = modelComsol.evaluate("comp1.point1")
        comsolResults = meas/ref
        return abs(comsolResults)

    def normalization(self, rho, eta, E):

        rho_norm = rho*self.rho_std_init + self.rho_mean
        eta_norm = eta*self.eta_std_init + self.eta_mean
        E_norm = E*self.E_std_init + self.E_mean

        return rho_norm, eta_norm, E_norm

        # Define sum of squares function
    def ssfun(self, q, data):
        y = data.ydata[0].T
        y = y[0]
        # Evaluate model
        ymodel = self.solveComsol(self.model, q)
        res = ymodel - y
        self.error = np.append(self.error, (res ** 2).sum(axis=0))
        return (res ** 2).sum(axis=0)
        
    def run(self):
        # Reading input files
        expRef = self.ref
        expMeas = self.data
        tf_exp = expMeas["y_FRF_vel"][1] / expRef["y_FRF_vel"][0]
        tf_exp_dB = 20*np.log10(abs(tf_exp))
        # Mobility value calculated from input data and converted to torch
        freq = expMeas["x_FRF_vel"][1].T# Freq values(x axis) converted to torch
        
        self.freqValues = np.array(self.freqValues)
        # BAYESIAN INFERENCE PROCESS
        # The experimental data has a resolution of 0.5 Hz. The values selected are integers
        # so the position of the freq. values in the array will be x*2
        input_x = freq[(self.freqValues*2).astype(int)]
        y_obs = tf_exp[(self.freqValues*2).astype(int)] # Suppose this was the vector of observed y's

        mcstat = MCMC()
        mcstat.data.add_data_set(input_x, y_obs)
        mcstat.model_settings.define_model_settings(
            sos_function=self.ssfun)
        # Define simulation options
        mcstat.simulation_options.define_simulation_options(nsimu=100) # No. of MCMC simulations
        # Add model parameters
        mcstat.parameters.add_model_parameter(
            name='E',
            theta0=0., # initial value 
            minimum=-1.5, # lower limit 
            maximum=1.5)
        mcstat.parameters.add_model_parameter(
            name='rho',
            theta0=0., # initial value 
            minimum=-1.5, # lower limit 
            maximum=1.5) # upper limit
        mcstat.parameters.add_model_parameter(
            name='eta',
            theta0=0., # initial value 
            minimum=-1.5, # lower limit 
            maximum=1.5)
        # Run simulation
        start = time.time()
        mcstat.run_simulation()
        timeConsumed = time.time() - start
        results = mcstat.simulation_results.results
        chain = results['chain'] 
        names = results['names']
        # generate mcmc plots
        mcpl = mcstat.mcmcplot # initialize plotting methods
        mcpl.plot_chain_panel(chain[:10], names[:10])

        results = mcstat.simulation_results.results 
        chain = results['chain'].copy()
        burnin = 0 #int(chain.shape[0]/2)
        # display chain statistics 
        mcstat.chainstats(chain[burnin:, :], results)
        
        mode_E = stats.mode(chain[:,0], keepdims=True)[0][0]
        mode_rho = stats.mode(chain[:,1], keepdims=True)[0][0]
        mode_eta = stats.mode(chain[:,2], keepdims=True)[0][0]

        comsolResult = self.solveComsol(self.comsolModelFullRange, [mode_E, mode_rho, mode_eta])
        
        results = {"vel_est": comsolResult,
           "E_est": mode_E,  
           "E_dist": chain[:, 0],
           "rho_est": mode_rho,  
           "rho_dist": chain[:, 1],
           "eta_est": mode_eta,
           "eta_dist": chain[:, 2],
           "completedResults": mcstat.simulation_results.results,
           "error": self.error,
           "time": timeConsumed}
        print(results)
        return results

if __name__ == "__main__":
    configFilePath = sys.argv[1]
    configFilePath = "./configTest.yaml" 
    obj = ComsolProcess(configFilePath)
    results = obj.run()
    with open('../resultsTest.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)



    
