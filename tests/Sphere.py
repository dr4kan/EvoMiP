from evomip.Config import Config
from evomip.Parameter import *
from evomip.SearchSpace import SearchSpace
from evomip.Population import Population
from evomip.Functions import sphere
from evomip.WOA import WOA
from evomip.GWO import GWO
from evomip.MFO import MFO
from evomip.BAT import BAT
from evomip.ABC import ABC
from evomip.HS import HS
from evomip.IHS import IHS
from evomip.GSA import GSA
from evomip.CS import CS
from evomip.GA import GA
from evomip.SA import SA
from evomip.PS import PS


import numpy as np
import warnings
warnings.filterwarnings("error")

# configuration
config = Config(nmax_iter=100, nmax_iter_same_cost=0, seed=101, silent=True)

# search space
l = createListParameters(2, -5.12, 5.12)
sspace = SearchSpace(l)

# population
population = Population(100, sphere, sspace, config)

# running over all algorithms
results = {}
all_algos = [ABC, BAT, CS, GA, GSA, GWO, HS, IHS, MFO, PS, SA, WOA]
for i in all_algos:
    algo = i(population)
    algo.minimize()
    results[algo.result.algo] = algo.result.bestSolution.cost

# printing the results
print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in results.items()) + "}")