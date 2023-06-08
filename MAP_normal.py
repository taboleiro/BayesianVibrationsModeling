import torch
import pandas as pd
import numpy as np
from torch.distributions import constraints
from scipy import stats
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm

def mobilityFuncModel(E_dist, rho_dist, eta_dist, freq, E_theo=10e10, rho_theo=8976, eta_theo=0.007):
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
        "freq": [[10, 470, 2],[470, 520, 1],[520, 600, 5], \
                [600,700, 2],[700,1350, 20],[1350, 1390,2], \
                [1390,1570,20],[1570,1630,2], [1630,2650,100], \
                [2650,2750,2],[2750,2950, 20], [2950, 3050, 2]]
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

def model_YoungDampingDensity(x, y_obs):
    # Density definition
    E_theo=10e10
    rho_theo=8976
    eta_theo=0.007
    rho_mean = pyro.param("rho_mean", dist.Normal(rho_theo, 1))
    rho = pyro.sample("rho", dist.Normal(rho_mean, 0.1))
    # Damping loss factor definition
    eta_mean = pyro.param("eta_mean", dist.Normal(eta_theo, 1))
    eta = pyro.sample("eta", dist.Normal(eta_mean, 0.1))
    # Young's modulus definition
    E_mean = pyro.param("E_mean", dist.Normal(E_theo, 2))
    E = pyro.sample("E", dist.Normal(E_mean, 0.1))
    
    with pyro.plate("data", len(y_obs)):
        y_values = mobilityFuncModel(E, rho, eta, x)
        y = pyro.sample("y", dist.Normal(20*torch.log10(y_values), 0.001), obs=20*torch.log10(y_obs))
    return y

def guide(x, y_obs):
    # Density guide
    E_theo=10.0e10
    rho_theo=8976.
    eta_theo=0.007
    rho_mean = pyro.param("rho_mean", dist.Normal(rho_theo, 1), constraint=constraints.positive)
    rho_std = pyro.param("rho_std", torch.tensor(0.3), constraint=constraints.positive)
    rho = pyro.sample("rho", dist.Normal(rho_mean, rho_std))

    # Damping loss factor guide
    eta_mean = pyro.param("eta_mean", dist.Normal(eta_theo, 0.5), constraint=constraints.positive)
    eta_std = pyro.param("eta_std", torch.tensor(.2), constraint=constraints.positive)
    eta = pyro.sample("eta", dist.Normal(eta_mean, eta_std))

    # Damping loss factor guide
    E_mean = pyro.param("E_mean", dist.Normal(E_theo, 2), constraint=constraints.positive)
    E_std = pyro.param("E_std", torch.tensor(1.), constraint=constraints.positive)
    E = pyro.sample("E", dist.Normal(E_mean, E_std))
 
    with pyro.plate("data", len(y_obs)):
        y_values = mobilityFuncModel(E, rho, eta, x)
        y = pyro.sample("y", dist.Normal(20*torch.log10(y_values), 0.01), obs=20*torch.log10(y_obs))
    return y

def train(freq, data, model, guide, lr=0.001, n_steps=30001):
    pyro.clear_param_store()
    adam_params = {"lr": lr, 
                   "betas": (0.9, 0.999),
                   "eps": 1e-08}
    adam = pyro.optim.Adam(adam_params)
    
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    losses = []
    for step in tqdm(range(n_steps)):
        loss = svi.step(freq, data)
        if step % 50 == 0:
            losses.append(loss)
            #print(".", end="")
            #print('[iter {}]  loss: {:.4f}'.format(step, loss))
    plt.figure(10)
    plt.plot(np.linspace(0, n_steps, len(losses)), losses, "*-")
    plt.savefig("./figuresResults/ErrorElboNO"+str(lr).split(".")[-1]+"_"+str(n_steps)+".png")
    return lr, n_steps

if __name__ == "__main__":
    files = ["centerFreqResponse"]#, "center2FreqResponse", "randomFreqResponse"]
    Y_exp = []
    for file in files:
        experiment = pd.read_csv("./Data/bend/"+file+".csv")[20:]
        # Mobility value calculated from input data and converted to torch
        mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
        peaks = utils.resonances(mobility)
        Y_exp = mobility[peaks]
        freq = torch.tensor(experiment["freq"][peaks].values) # Freq values(x axis) converted to torch
    #Y_exp = pd.read_csv("./BayesianVibrationsModeling/Data/bend/"+file+".csv")[20:]
    Y_exp = torch.tensor(Y_exp)

    [lr, n_steps] = train(freq, Y_exp, model_YoungDampingDensity, guide)
    E_est = pyro.param("E_mean").item()
    eta_est = pyro.param("eta_mean").item()
    rho_est = pyro.param("rho_mean").item()

    print("Our MAP estimate of E is {:.3f}".format((E_est)))
    print("Our MAP estimate of eta is {:.3f}".format((eta_est)))
    print("Our MAP estimate of rho is {:.3f}".format((rho_est)))

    plt.figure(11)
    mob_dB = 20*np.log10(mobility)
    plt.plot(experiment["freq"].values, mob_dB, label="experiment")
    Y_est = mobilityFuncModel(torch.tensor(E_est), torch.tensor(rho_est), torch.tensor(eta_est), torch.tensor(experiment["freq"].values))
    plt.plot(experiment["freq"].values, 20*np.log10(Y_est), label="initial")
    Y_est = mobilityFuncModel(torch.tensor(E_est), torch.tensor(rho_est), torch.tensor(eta_est), torch.tensor(experiment["freq"].values))
    plt.plot(experiment["freq"].values, 20*np.log10(Y_est), label="estimated")
    plt.savefig("./figuresResults/mobNO_"+str(lr).split(".")[-1]+"_"+str(n_steps)+".png")
    plt.show()


    plt.figure(12)
    eta_theo = 0.01
    variance = 1
    sigma = np.sqrt(variance)
    x_init = np.linspace(eta_theo - 3*sigma, eta_theo + 3*sigma, 100)
    plt.plot(x_init, stats.norm.pdf(x_init, eta_theo, sigma))


    eta_mean = pyro.param("eta_mean").item()
    eta_var = pyro.param("eta_std").item()
    sigma = np.sqrt(eta_var)
    x = np.linspace(eta_mean - 3*sigma, eta_mean + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, eta_mean, sigma))
    plt.show()