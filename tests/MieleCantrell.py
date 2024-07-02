from evomip.Config import Config
from evomip.Parameter import *
from evomip.SearchSpace import SearchSpace
from evomip.Population import Population
from evomip.Functions import sphere
from evomip.BAT import BAT
import numpy as np

# definition of the objective function
def miele_cantrell(x: np.array):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    return (np.exp(-x1) - x2)**4 + 100*(x2 - x3)**6 + (np.tan(x3 - x4))**4 + x1**8

# configuration of the optimization parameters
config = Config(nmax_iter=100, nmax_iter_same_cost=0)

# definition of the search space
l = createListParameters("x", 4, -2., 2.)
sspace = SearchSpace(l)

# defition of the population
population = Population(100, miele_cantrell, sspace, config)

# running the algorithm
algo = BAT(population)
algo.minimize()

# summary of the results
algo.summary()
