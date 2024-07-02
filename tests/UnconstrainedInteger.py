from evomip.Config import Config
from evomip.Parameter import *
from evomip.SearchSpace import SearchSpace
from evomip.Population import Population
from evomip.Functions import sphere
from evomip.WOA import WOA
import numpy as np

# definition of the objective function
def obFunc(x: np.array):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    x8 = x[7]
    x9 = x[8]
    x10 = x[9]
    y = x1**2 + x1*x2 - x2**2 + x3*x1 - x3**2 + 8*(x4**2) - 17*(x5**2) + 6*(x6**3) + x6*x5*x4*x7 + x8**3 + x9**4 - x10**5 - x10*x5 + 18*x3*x7*x6
    return -y

# configuration of the optimization parameters
config = Config(nmax_iter=500, nmax_iter_same_cost=0)

# definition of the search space
l = createListParameters("x", 10, 0, 99, True)
sspace = SearchSpace(l)

# defition of the population
population = Population(200, obFunc, sspace, config)

# running the algorithm
algo = WOA(population)
algo.minimize()

# summary of the results
algo.summary()
