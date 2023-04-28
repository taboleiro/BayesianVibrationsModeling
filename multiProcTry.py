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
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import cpu_count
from queue import Empty


def boss():
    jobs = Queue()
    freq = [100, 200, 300, 400, 500, 600, 700, 800, 900, 50, 60, 70, 80, 90, 2000]
    for f in freq:
        jobs.put(f)
    
    results = Queue()
    processes = []
    workers = cpu_count()
    for _ in range(workers):
        process = Process(target=worker, args=(jobs, results))
        processes.append(process)
        process.start()
    for _ in freq:
        print(results.get())

def worker(jobs, results):
    client = mph.start(cores=1)
    model = client.load("./comsol/beam.mph")
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

if __name__ == "__main__":
    boss()