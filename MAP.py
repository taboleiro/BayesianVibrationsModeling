import torch
import pandas as pd
import numpy as np
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from scipy.stats import beta

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

    Ehigh = torch.tensor(12e10)
    Elow = torch.tensor(9e10)

    rhohigh = torch.tensor(8.8e3)
    rholow = torch.tensor(7.3e3)

    etahigh = torch.tensor(0.01)
    etalow = torch.tensor(0.0001)

    E = Elow + E_dist*(Ehigh-Elow)
    rho = rholow + rho_dist*(rhohigh-rholow)
    eta = etalow + E_dist*(etahigh-etalow)

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
    alpha_rho = pyro.param("alpha_rho", torch.tensor(1.))
    beta_rho = pyro.param("beta_rho", torch.tensor(1.))
    rho = pyro.sample("rho", dist.Beta(alpha_rho, beta_rho))
    # Damping loss factor definition
    alpha_eta = pyro.param("alpha_eta", torch.tensor(1.))
    beta_eta = pyro.param("beta_eta", torch.tensor(1.))
    eta = pyro.sample("eta", dist.Beta(alpha_eta, beta_eta))
    # Young's modulus definition
    alpha_E = pyro.param("alpha_E", torch.tensor(1.))
    beta_E = pyro.param("beta_E", torch.tensor(1.))
    E = pyro.sample("E", dist.Beta(alpha_E, beta_E))
    
    with pyro.plate("data", len(y_obs)):
        y_values = mobilityFuncModel(E, rho, eta, x)
        y = pyro.sample("y", dist.Normal(y_values, 0.001), obs=y_obs)
    return y

def guide(x, y_obs):
    # Density definition
    rho = pyro.sample("rho", dist.Beta(1, 1))
    # Damping loss factor definition
    eta = pyro.sample("eta", dist.Beta(1, 1))
    # Young's modulus definition
    E = pyro.sample("E", dist.Beta(1, 1))

    with pyro.plate("data", len(y_obs)):
        y_values = mobilityFuncModel(E, rho, eta, x)
        y = pyro.sample("y", dist.Beta(10, 10), obs=y_obs)
    return y

def train(freq, data, model, guide, lr=0.1, n_steps=3000):
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
    plt.plot(np.linspace(0, n_steps, len(losses)), np.array(losses), "*-")
    plt.yscale("log")
    plt.savefig("./figuresResults/ErrorElbo"+str(lr).split(".")[-1]+"_"+str(n_steps)+".png")
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

    r = beta.rvs(1, 1, size=1000)
    beta_E = pyro.param("beta_E").item()
    alpha_E = pyro.param("alpha_E").item()
    E_est, var, skew, kurt = beta.stats(alpha_E, beta_E , moments='mvsk')
    E_post = beta.rvs(alpha_E, beta_E, size=1000)

        
    r = beta.rvs(1, 1, size=1000)
    beta_rho = pyro.param("beta_rho").item()
    alpha_rho = pyro.param("alpha_rho").item()
    rho_post = beta.rvs(alpha_E, beta_E, size=1000)
    rho_est, var, skew, kurt = beta.stats(alpha_rho, beta_rho , moments='mvsk')

    r = beta.rvs(1, 1, size=1000)
    beta_eta = pyro.param("beta_eta").item()
    alpha_eta = pyro.param("alpha_eta").item()
    eta_post = beta.rvs(alpha_E, beta_E, size=1000)
    eta_est, var, skew, kurt = beta.stats(alpha_eta, beta_eta , moments='mvsk')
    
    Ehigh = 12e10
    Elow = 9e10

    rhohigh = 8.8e3
    rholow = 7.3e3

    etahigh = 0.01
    etalow = 0.0001
    print("Our MAP estimate of E is {:.3f}".format((Elow + E_est*(Ehigh-Elow))))
    print("Our MAP estimate of eta is {:.3f}".format((etalow + eta_est*(etahigh-etalow))))
    print("Our MAP estimate of rho is {:.3f}".format((rholow + rho_est*(rhohigh-rholow))))

    plt.figure(11)
    mob_dB = 20*np.log10(mobility)
    plt.plot(experiment["freq"].values, mob_dB)
    Y_est = mobilityFuncModel(torch.tensor(E_est), torch.tensor(rho_est), torch.tensor(eta_est), torch.tensor(experiment["freq"].values))
    plt.plot(experiment["freq"].values, 20*np.log10(Y_est), "--")
    plt.savefig("./figuresResults/mob_"+str(lr).split(".")[-1]+"_"+str(n_steps)+".png")
    plt.show()

    plt.figure(12)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, beta.pdf(x, alpha_E, beta_E),
        'r-', lw=1, alpha=0.6, label='beta pdf')
    plt.plot(x, beta.pdf(x, 1, 1),
        'r-', lw=1, alpha=0.6, label='beta init')

    plt.figure(13)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, beta.pdf(x, alpha_rho, beta_rho),
        'r-', lw=1, alpha=0.6, label='beta pdf')
    plt.plot(x, beta.pdf(x, 1, 1),
        'r-', lw=1, alpha=0.6, label='beta init')

    plt.figure(14)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, beta.pdf(x, alpha_eta, beta_eta),
        'r-', lw=1, alpha=0.6, label='beta pdf')
    plt.plot(x, beta.pdf(x, 1, 1),
        'r-', lw=1, alpha=0.6, label='beta init')
    plt.show()