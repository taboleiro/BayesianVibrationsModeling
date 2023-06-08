import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import subprocess

def createComsolVector(scale, sections, param="n_points", display=False):
    """
    CreateComsolVector - Copy directly to clipboard the comsol command to
    generate the desired range of values
    Given the start, end and step for every frequency section, and having
    decided the scale, the function return directly into the clipboard the
    command for defining a frequency array in comsol. The user just need to
    press ctrl+v to paste it in the program

    Args:
    - scale (str): scale step for the array
        - "lin"
        [[start_value, end_value, numper_of_points], ...]
        - "log"
    - sections (np.ndarray): array of vectors with the format: 
    - *args:
        - display (bool): if True, display a plot to visualize the resulting vector (default=True)

    Returns:
    - command (str): Contains the comsol command.
    """
    valList = []
    lastValue = 0

    sections = np.array(sections)

    if sections.shape[0] == 1:
        sections = np.reshape(sections, (-1, 3))
    
    val = []

    for n in range(sections.shape[0]):
        a = sections[n,:] # stracting individual zones a = (start, end ,step) 

        if scale == "lin":
            if param == "n_points":
                step = (a[1]-a[0])/(a[2]-1) #(end - start)/step
            else:
                step = a[2]

            # checking if a(1) is already in the final vector
            if lastValue == a[0]:
                tmp = [a[0]+step, step, a[1]]
            else:
                tmp = [a[0], step, a[1]]
            
            # val[n] := "range(start, step, end) "
            val.append(f'range({tmp[0]:.2f},{tmp[1]:.2f},{tmp[2]:.2f}) ')
            
            # Obtaining numerical values for graphical revision
            valList = np.concatenate((valList, np.arange(tmp[0], tmp[2]+tmp[1], tmp[1])), axis=None)

        elif scale == "log":
            if param == "n_points":
                step = (np.log10(a[1])-np.log10(a[0]))/(a[2]-1) # logaritmic step based on Comsol performance
            else:
                step = a[2]
            # checking if a(1) is already in the final vector
            if lastValue == a[0]:
                tmp = [10**(np.log10(a[0])+step), step, a[1]]
            else:
                tmp = [a[0], step, a[1]]

            # val[n] := "10^range(log10(start), step, log10(end)) "
            val.append(f'10^(range(log10({tmp[0]:.2f}),{tmp[1]:.2f},log10({tmp[2]:.2f}))) ')
            
            # Obtaining numerical values for graphical revision
            valList = np.concatenate((valList, 10**(np.arange(np.log10(tmp[0]), np.log10(tmp[2])+tmp[1], tmp[1]))), axis=None)

        else:
            print("Unknown spacing scale. valid: lin/log")
            return 

        # saving last value to compare it with the first of the next vector
        lastValue = a[1]
    
    command = ''.join(val)
    # copying the command
    subprocess.run("pbcopy", text=True, input=command)

    message = f"Command copied in the clipboard.\nTotal number of data points: {len(valList)}\nPress ctrl+v to paste it\n"
    print(message)

    if display:
        plt.figure(10)
        # logaritmic x axis if the scale is log. Linear otherwise
        if scale == "log":
            plt.semilogx(valList, range(1, len(valList)+1), "*-")
            plt.grid(True)
        else:
            plt.plot(valList, range(1, len(valList)+1), "*-")
            plt.grid(True)
        plt.ylabel("Data point number")
        plt.xlabel("Data value")
        plt.show()

    return valList


def resonances(data):
    # find the resonances
    peaks = find_peaks(data, prominence=0.3)[0]

    # take the index of the previous and next 10 values of each resonance
    freq = np.linspace(peaks-np.ones(len(peaks))*10, peaks+np.ones(len(peaks))*10, 21).T
    
    # Convert matrix into array
    return freq.flatten().astype(int)


if __name__ == "__main__":
    # freqVector = [[280, 1, 450], [101, 1000, 5]]
    # createComsolVector("lin",[[280, 1, 450],[101, 1000, 5]], param="step", display=True)
    experiment = pd.read_csv("./Data/bend/centerFreqResponse.csv")[20:]
    mobility = abs(experiment["force"].values + 1j*experiment["velocity"].values)
    peaks = resonances(mobility)
    plt.plot(experiment["freq"], mobility)
    plt.plot(experiment["freq"], peaks, "x")