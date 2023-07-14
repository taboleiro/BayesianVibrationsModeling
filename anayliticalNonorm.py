import torch
import pandas as pd
import numpy as np
from torch.distributions import constraints
from scipy import stats
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.poutine.runtime import effectful
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm

class ComsolProcess(object):
    def __init__(self):
        self.E_theo=8e10
        self.E_var_init = 5e9
        self.rho_theo=8000.0
        self.rho_var_init = 250
        self.eta_mean=0.00505
        self.eta_var_init = 0.006

    def mobilityFuncModel(self, E_dist, rho_dist, eta_dist, freq, E_theo=10e10, rho_theo=8976, eta_theo=0.007):
        """
        Calculates the mobility value based on the Young's modulus(E) and the frequency
        Input: 
            E   : Young's modulus
            eta : loss factor
        Output: 
            Y   : Mobility value
        """
        beam = {"length": 0.301,
            "width": 0.026,
            "thickness": 0.003,
            
            "E": 10e10,
            
            "mass": 0.1877,
            
            }
        beam["massPerUnit"] = beam["mass"] / beam["length"]
        beam["volume"] = beam["length"] * beam["width"] * beam["thickness"]
        beam["I"] = beam["width"]*beam["thickness"]**3/12

        E = E_dist
        rho = rho_dist
        eta = eta_dist
        l = beam["length"]/2

        # calculating the bending wave number
        w = 2*torch.pi*freq # Angular frequency
        B = E*beam["I"] #
        complex_B = E*(1+1j*eta)*beam["I"]
        massPerUnit = rho*beam["thickness"]*beam["width"]
        cb = torch.sqrt(w)*(B/massPerUnit)**(1/4) # bending wave velocity
        
        kl = w/(cb)*l # bending wave number
        complex_kl = kl*(1-1j*eta/4)
        N_l = torch.cos(complex_kl)*torch.cosh(complex_kl) + 1
        D_l = torch.cos(complex_kl)*torch.sinh(complex_kl) + torch.sin(complex_kl)*torch.cosh(complex_kl)

        Y = -(1j*l)/ (2*complex_kl*torch.sqrt(complex_B *massPerUnit)) * N_l/D_l
        return abs(Y)

    # @poutine.scale(scale=1.0/180)
    @poutine.scale(scale=1.0/(10e10))
    def model_YoungDampingDensity(self, x, y_obs):
        # Density definition
        rho_mean = pyro.param("rho_mean", torch.tensor(8000.0))
        rho_std = pyro.param("rho_std", torch.tensor(250), constraint=constraints.positive)
        rho = pyro.sample("rho", dist.Normal(rho_mean, rho_std))
        # Damping loss factor definition
        eta_mean = pyro.param("eta_mean", torch.tensor(0.00505))
        eta_std = pyro.param("eta_std", torch.tensor(0.006), constraint=constraints.positive)
        eta = pyro.sample("eta", dist.Normal(eta_mean, eta_std))
        # Young's modulus definition
        E_mean = pyro.param("E_mean", torch.tensor(9.0e10))
        E_std = pyro.param("E_std", torch.tensor(5e9), constraint=constraints.positive)
        E = pyro.sample("E", dist.Normal(E_mean, E_std))
        
        #rho, eta, E = self.normalization(rho, eta, E, rho_std, eta_std, E_std)
        #rho, eta, E = self.normalization(rho*rho_std+rho_mean, eta*eta_std+eta_mean, E*E_std+E_mean, rho_std, eta_std, E_std)

        self.E_values[self.step] = E
        self.eta_values[self.step] = eta
        self.rho_values[self.step] = rho

        y_values = self.mobilityFuncModel(E, rho, eta, x)
        with pyro.plate("data", len(y_obs)):
            y = pyro.sample("y", dist.Normal(20*torch.log10(y_values), 0.001), obs=20*torch.log10(y_obs))
        return y

    # @poutine.scale(scale=1.0/180)
    @poutine.scale(scale=1.0/(10e10))
    def guide(self, x, y_obs):
        # Density guide
        rho_mean_q = pyro.param("rho_mean_guide", torch.tensor(8000.0))
        rho_std_q = pyro.param("rho_std_guide", torch.tensor(250*0.05), constraint=constraints.positive)
        pyro.sample("rho", dist.Normal(rho_mean_q, rho_std_q))
        # Damping loss factor guide
        eta_mean_q = pyro.param("eta_mean_guide", torch.tensor(0.00505))
        eta_std_q = pyro.param("eta_std_guide", torch.tensor(0.006*0.05), constraint=constraints.positive)
        pyro.sample("eta", dist.Normal(eta_mean_q, eta_std_q))

        # Damping loss factor guide
        E_mean_q = pyro.param("E_mean_guide", torch.tensor(9.0e10))
        E_std_q = pyro.param("E_std_guide", torch.tensor(5e9*0.05), constraint=constraints.positive)
        pyro.sample("E", dist.Normal(E_mean_q, E_std_q))
        

    def normalization(self, rho, eta, E, rho_var, eta_var, E_var):
        
        rho_norm = rho*self.rho_var_init + self.rho_theo
        eta_norm = eta*self.eta_var_init + self.eta_mean
        E_norm = E*self.E_var_init + self.E_theo

        return rho_norm, eta_norm, E_norm

    def train(self, freq, data, model, guide, lr=0.0001, n_steps=5000):
        pyro.clear_param_store()
        adam_params = {
            "lr": .01, #0.001,
            "betas": (0.96, 0.999),
            "clip_norm": 20.0,
            "lrd": 0.99996,
            "weight_decay": 2.0}
        initial_lr = lr
        gamma = 100 # final learning rate will be gamma * initial_lr
        lrd = gamma ** (1 / n_steps)

        optim = pyro.optim.ClippedAdam(adam_params)
        adam = pyro.optim.Adam(adam_params)
        
        svi = SVI(model, guide, optim, loss=TraceGraph_ELBO())

        losses = []
        self.E_values = np.zeros(n_steps)
        self.eta_values = np.zeros(n_steps)
        self.rho_values = np.zeros(n_steps)
        for step in tqdm(range(n_steps)):   
            self.step = step    
            loss = svi.step(freq, data)
            losses.append(loss)
            #if step % 50 == 0:
                #print(".", end="")
                #print('[iter {}]  loss: {:.4f}'.format(step, loss))
        return lr, n_steps, losses, self.E_values, self.eta_values, self.rho_values

if __name__ == "__main__":
    files = ["centerFreqResponse"]#, "center2FreqResponse", "randomFreqResponse"]
    Y_exp = []
    freqValues = [1365, 1365.5, 1366, 1366.5, 1367, 1367.5, 1368, 1368.5, 1369, 1369.5, 1370, 1370.5, 1371, 1371.5, 1372, 1372.5, 1373, 1373.5, 1374, 1374.5, 1375, 1375.5, 1376, 1376.5, 1377, 1377.5, 1378, 1378.5, 1379, 1379.5, 1380, 1380.5, 1381, 1381.5, 1382, 1382.5, 1383, 1383.5, 1384, 1384.5, 1595, 1595.5, 1596, 1596.5, 1597, 1597.5, 1598, 1598.5, 1599, 1599.5, 1600, 1600.5, 1601, 1601.5, 1602, 1602.5, 1603, 1603.5, 1604, 1604.5, 2685, 2685.5, 2686, 2686.5, 2687, 2687.5, 2688, 2688.5, 2689, 2689.5, 2690, 2690.5, 2691, 2691.5, 2692, 2692.5, 2693, 2693.5, 2694, 2694.5, 2695, 2695.5, 2696, 2696.5, 2697, 2697.5, 2698, 2698.5, 2699, 2699.5]
    #freqValues = np.array(np.arange(1365*2, 2700*2))
    #freqValues = np.arange(50, 350.5, 0.5)
    freqValues = 2*np.array(freqValues)
    freqIndex = freqValues.astype(int)
    for file in files:
        experiment = pd.read_csv("./Data/bend/"+file+".csv")[20:]
        # Mobility value calculated from input data and converted to torch
        mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
        #peaks = utils.resonances(mobility)
        Y_exp = mobility[freqIndex]
        freq = torch.tensor(experiment["freq"][freqIndex].values) # Freq values(x axis) converted to torch
        #freq = torch.tensor(experiment["freq"].values) # Freq values(x axis) converted to torch
    #Y_exp = pd.read_csv("./BayesianVibrationsModeling/Data/bend/"+file+".csv")[20:]
    Y_exp = torch.tensor(Y_exp)
    x = ComsolProcess()
    [lr, n_steps, losses, E_values, eta_values, rho_values] = x.train(freq, Y_exp, x.model_YoungDampingDensity, x.guide)
    E_est = pyro.param("E_mean_guide").item()
    eta_est = pyro.param("eta_mean_guide").item()
    rho_est = pyro.param("rho_mean_guide").item()
    E_std = pyro.param("E_std_guide").item()
    eta_std = pyro.param("eta_std_guide").item()
    rho_std = pyro.param("rho_std_guide").item()

    #rho_values, eta_values, E_values = x.normalization(rho_values, eta_values, E_values, rho_std, eta_std, E_std)
    plt.figure(30)
    plt.plot(rho_values)
    plt.figure(31)
    plt.plot(eta_values)
    plt.figure(32)
    plt.plot(E_values)
    plt.show()
    #rho_est, eta_est, E_est = x.normalization(rho_est, eta_est, E_est, rho_std, eta_std, E_std)  

    print("Our MAP estimate of E is {:.3f}".format((E_est)))
    print("Our MAP estimate of eta is {:.3f}".format((eta_est)))
    print("Our MAP estimate of rho is {:.3f}".format((rho_est)))

    plt.figure(10)
    plt.plot(np.linspace(0, n_steps, len(losses)), losses, "*-")
    plt.xlabel("iterations ")
    plt.ylabel(" Error estimation")
    plt.savefig("./figResults/ErrorElboNO"+str(lr).split(".")[-1]+"_"+str(n_steps)+"_all_decay.png")
    
    plt.figure(11)
    mob_dB = 20*np.log10(mobility)
    plt.plot(experiment["freq"].values, mob_dB, label="experiment")
    #Y_est = mobilityFuncModel(torch.tensor(E_est), torch.tensor(rho_est), torch.tensor(eta_est), torch.tensor(experiment["freq"].values))
    #plt.plot(experiment["freq"].values, 20*np.log10(Y_est), label="initial")
    Y_est = x.mobilityFuncModel(torch.tensor(E_est), torch.tensor(rho_est), torch.tensor(eta_est), torch.tensor(experiment["freq"].values))
    plt.plot(experiment["freq"].values, 20*np.log10(Y_est), label="estimated")
    plt.xlabel("Frequency / Hz")
    plt.ylabel("Mobility / dB")
    plt.xscale("log")
    plt.savefig("./figResults/mobNO_"+str(lr).split(".")[-1]+"_"+str(n_steps)+"LAST_all_decay.png")


    E_theo=8e10
    E_var_init = 5e9
    rho_theo=8000.0
    rho_var_init = 250
    eta_mean=0.01505
    eta_var_init = 0.006
    
    # ETA DISTRIBUTION PLOT
    plt.figure(12)
    eta_theo = 0.01
    variance = 1
    sigma = np.sqrt(variance)
    x_init = np.linspace(-3*sigma*eta_var_init, 3*sigma*eta_var_init, 100) + eta_theo
    plt.plot(x_init, stats.norm.pdf(x_init, eta_theo, sigma*eta_var_init), label="prior")

    eta_var = pyro.param("eta_std").item()
    sigma = np.sqrt(eta_var)
    x = np.linspace(-3*eta_std, 3*eta_std, 100)+ eta_est
    plt.plot(x, stats.norm.pdf(x, eta_est, eta_std), label="posterior")
    plt.xlabel("Damping loss factors")
    plt.legend()
    plt.savefig("./figResults/ETA_"+str(lr).split(".")[-1]+"_"+str(n_steps)+"LAST_all_decay.png")

    # RHO DISTRIBUTION PLOT
    plt.figure(13)
    eta_theo = 0.01
    variance = 1
    sigma = np.sqrt(variance)
    x_init = np.linspace(-3*sigma*rho_var_init, 3*sigma*rho_var_init, 100) + rho_theo
    plt.plot(x_init, stats.norm.pdf(x_init, rho_theo, rho_var_init), label="prior")

    rho_std = pyro.param("rho_std").item()
    sigma = np.sqrt(rho_std)
    x = np.linspace(-3*rho_var_init*rho_std, 3*rho_var_init*rho_std, 100)+ rho_est
    plt.plot(x, stats.norm.pdf(x, rho_est, rho_var_init*rho_std), label="posterior")
    plt.xlabel("Density / Kg/m^3")
    plt.legend()
    plt.savefig("./figResults/RHO_"+str(lr).split(".")[-1]+"_"+str(n_steps)+"LAST_all_decay.png")

    # E DISTRIBUTION PLOT
    plt.figure(14)
    variance = 1
    sigma = np.sqrt(variance)
    x_init = np.linspace(-3*sigma*E_var_init, 3*sigma*E_var_init, 100) + E_theo
    plt.plot(x_init, stats.norm.pdf(x_init, E_theo, sigma*E_var_init), label="prior")

    E_mean = pyro.param("E_mean").item()
    E_var = pyro.param("E_std").item()
    sigma = np.sqrt(E_std)
    x = np.linspace(-3*E_var_init*E_std, 3*E_var_init*E_std, 100)+ E_est
    plt.plot(x, stats.norm.pdf(x, E_est, E_var_init*E_std), label="posterior")
    plt.xlabel("Young's modulus / Pa")
    plt.legend()
    plt.savefig("./figResults/E_"+str(lr).split(".")[-1]+"_"+str(n_steps)+"LAST_all_decay.png")
    plt.show()
    print("finish")