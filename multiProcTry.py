import numpy as np
import pandas as pd

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, HMC

import torch
from torch.distributions import constraints

import matplotlib.pyplot as plt
import seaborn as sns

import graphviz
import utils

import mph
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import cpu_count
from queue import Empty



def boss():
    jobs = Queue()
    freq = [100, 200, 300, 400, 500, 600, 700, 800, 900, 50, 60, 70, 80, 90, 2000
    ]
    for f in freq:
        jobs.put(f)
    
    results = Queue()
    processes = []
    workers = cpu_count()
    #Initialize processes
    for i in range(3):
        process = myProcess()
        processes.append(process)
    for process in range(processes):
        #process = Process(target=worker, args=(jobs, results, model))
        #processes.append(process)
        process.start()
    for _ in freq:
        print(results.get())

def worker(jobs, results, model):
    while True:
        try:
            d = jobs.get(block=False)
        except Empty:
            break
        print(d)
        model.parameter('freq', str(d)+' [Hz]')
        model.solve("Study 3")
        Y = abs(model.evaluate('comp1.point1'))
        results.put((d, Y))


class myProcess(multiprocessing.Process):
    def __init__(self, name, queue):
        super(myProcess, self).__init__()
        self.name = name
        self.queue = queue
        client = mph.start(cores=1)
        self.model = client.load("./comsol/beam.mph")
    
    def run(self, freq):
        self.model.parameter('freq', str(freq)+' [Hz]')
        self.model.solve("Study 3")
        Y = abs(self.model.evaluate('comp1.point1'))
        return Y
            
if __name__ == "__main__":
    boss()