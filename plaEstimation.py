# import and initialize
from pymcmcstat.MCMC import MCMC 
import pandas as pd
import numpy as np
import mph
import scipy.io as sio

data = sio.loadmat("../Data/laser/references/scanCurveBeamCornerMiddleLimit.mat")
ref = sio.loadmat("../Data/laser/references/curveBeamRefCenter.mat")

tf_center = data["y_FRF_disp"][1] / ref["y_FRF_disp"]
tf_corner = data["y_FRF_disp"][0] / ref["y_FRF_disp"]
tf_center_dB = 20*np.log10(abs(data["y_FRF_disp"][1] / ref["y_FRF_disp"]))
tf_corner_dB = 20*np.log10(abs(data["y_FRF_disp"][0] / ref["y_FRF_disp"]))
experiment = pd.read_csv("../Data/bend/centerFreqResponse.csv")[20:]
# Mobility value calculated from input data and converted to torch
mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
freq = experiment["freq"].values
goal = mobility[np.logical_and(mobility>0.1, mobility<0.75)]


client = mph.start()
#modelComsol = client.load("../comsol/beamSimpleTest1.mph")
modelComsol = client.load("../comsol/TestComsol.mph")
goalFreq = freq[np.logical_and(mobility>0.1, mobility<0.75)]

def solveComsol(modelComsol, param):#, freq=10):
    
    # Update parameters
    E, rho, eta = param
    rho, eta, E = normalization(rho, eta, E)
    modelComsol.parameter('youngs', str(E)+' [Pa]')
    modelComsol.parameter('density', str(rho)+' [kg/m^3]')
    modelComsol.parameter('eta', str(eta))


    # Solving comsol FEM
    modelComsol.solve("Study 2")
    #comsolResults1 = torch.tensor(modelComsol.evaluate("comp1.point2"))
    comsolResults = modelComsol.evaluate("comp1.point2")

    return abs(comsolResults)
def normalization(rho, eta, E):
    E_theo=9.7e10
    E_var_init =5.0e9
    rho_theo=8000.0
    rho_var_init =250.0
    eta_mean=0.00505
    eta_var_init = 0.006
    
    rho_norm = rho*rho_var_init + rho_theo
    eta_norm = eta*eta_var_init + eta_mean
    E_norm = E*E_var_init + E_theo

    return rho_norm, eta_norm, E_norm
    
def mobilityFuncModel(param):
    """
    Calculates the mobility value based on the Young's modulus(E) and the frequency
    Input: 
        E   : Young's modulus
        eta : loss factor
    Output: 
        Y   : Mobility value
    """
    global goalFreq
    beam = {"length": 0.301,
        "width": 0.026,
        "thickness": 0.003,
        
        "E": 10e10,
        
        "mass": 0.1877,
        
        }
    beam["massPerUnit"] = beam["mass"] / beam["length"]
    beam["volume"] = beam["length"] * beam["width"] * beam["thickness"]
    beam["I"] = beam["width"]*beam["thickness"]**3/12
    param
    E, rho, eta = param
    rho, eta, E = normalization(rho, eta, E)
    l = beam["length"]/2

    # calculating the bending wave number
    w = 2*np.pi*goalFreq # Angular frequency
    B = E*beam["I"] #
    complex_B = E*(1+1j*eta)*beam["I"]
    massPerUnit = rho*beam["thickness"]*beam["width"]
    cb = np.sqrt(w)*(B/massPerUnit)**(1/4) # bending wave velocity
    
    kl = w/(cb)*l # bending wave number
    complex_kl = kl*(1-1j*eta/4)
    N_l = np.cos(complex_kl)*np.cosh(complex_kl) + 1
    D_l = np.cos(complex_kl)*np.sinh(complex_kl) + np.sin(complex_kl)*np.cosh(complex_kl)

    Y = -(1j*l)/ (2*complex_kl*np.sqrt(complex_B *massPerUnit)) * N_l/D_l
    return abs(Y)


client = mph.start()
#modelComsol = client.load("../comsol/beamSimpleTest1.mph")
modelComsol = client.load("../comsol/TestComsol.mph")
goalFreq = freq[np.logical_and(mobility>0.1, mobility<0.75)]
def solveComsol(modelComsol, param):#, freq=10):
    
    # Update parameters
    E, rho, eta = param
    rho, eta, E = normalization(rho, eta, E)
    modelComsol.parameter('youngs', str(E)+' [Pa]')
    modelComsol.parameter('density', str(rho)+' [kg/m^3]')
    modelComsol.parameter('eta', str(eta))


    # Solving comsol FEM
    modelComsol.solve("Study 2")
    #comsolResults1 = torch.tensor(modelComsol.evaluate("comp1.point2"))
    comsolResults = modelComsol.evaluate("comp1.point2")

    return abs(comsolResults)
def normalization(rho, eta, E):
    E_theo=9.7e10
    E_var_init =5.0e9
    rho_theo=8000.0
    rho_var_init =250.0
    eta_mean=0.00505
    eta_var_init = 0.006
    
    rho_norm = rho*rho_var_init + rho_theo
    eta_norm = eta*eta_var_init + eta_mean
    E_norm = E*E_var_init + E_theo

    return rho_norm, eta_norm, E_norm
    
def mobilityFuncModel(param):
    """
    Calculates the mobility value based on the Young's modulus(E) and the frequency
    Input: 
        E   : Young's modulus
        eta : loss factor
    Output: 
        Y   : Mobility value
    """
    global goalFreq
    beam = {"length": 0.301,
        "width": 0.026,
        "thickness": 0.003,
        
        "E": 10e10,
        
        "mass": 0.1877,
        
        }
    beam["massPerUnit"] = beam["mass"] / beam["length"]
    beam["volume"] = beam["length"] * beam["width"] * beam["thickness"]
    beam["I"] = beam["width"]*beam["thickness"]**3/12
    param
    E, rho, eta = param
    rho, eta, E = normalization(rho, eta, E)
    l = beam["length"]/2

    # calculating the bending wave number
    w = 2*np.pi*goalFreq # Angular frequency
    B = E*beam["I"] #
    complex_B = E*(1+1j*eta)*beam["I"]
    massPerUnit = rho*beam["thickness"]*beam["width"]
    cb = np.sqrt(w)*(B/massPerUnit)**(1/4) # bending wave velocity
    
    kl = w/(cb)*l # bending wave number
    complex_kl = kl*(1-1j*eta/4)
    N_l = np.cos(complex_kl)*np.cosh(complex_kl) + 1
    D_l = np.cos(complex_kl)*np.sinh(complex_kl) + np.sin(complex_kl)*np.cosh(complex_kl)

    Y = -(1j*l)/ (2*complex_kl*np.sqrt(complex_B *massPerUnit)) * N_l/D_l
    return abs(Y)

results = mcstat.simulation_results.results
chain = results['chain'] 
names = results['names']
# generate mcmc plots
mcpl = mcstat.mcmcplot # initialize plotting methods
mcpl.plot_chain_panel(chain, names)
results = mcstat.simulation_results.results 
chain = results['chain']
burnin = int(chain.shape[0]/2)
# display chain statistics 
mcstat.chainstats(chain[burnin:, :], results)
mcpl.plot_density_panel(chain, names)
mcpl.plot_pairwise_correlation_panel(chain[burnin:,:], names)