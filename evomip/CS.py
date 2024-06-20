import numpy as np
from tqdm import tqdm
import math
import copy

from evomip.Algorithm import *
from evomip.Population import Population

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class CSPopulation(Population):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, population: Population, discovery_rate: float, 
                 step_size: float) -> None:
        super().__init__(population.size, population.objectiveFunction, 
                         population.searchSpace, population.config)
        self.pa = discovery_rate
        self.alpha = step_size
        
        
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def initRandom(self) -> None:
        self.n_valid_solutions = 0
        for i in range(0, self.size):
            self.solutions[i] = Individual(self.searchSpace.random())
            if (self.checkViolateConstraints(i) == False):
                self.n_valid_solutions += 1
                
        if (self.n_valid_solutions < self.config.min_valid_solutions):
            self.initRandom()

        self.evaluate()
        self.isInitialized = True
        
        
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def evaluate(self) -> None:
        for i in range(0, self.size):
            self.evalSolution(i)
            
            
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def evalSolution(self, i: int) -> None:
        self.evaluateCost(i)
        
        # check is the solution violates the constraints
        if (self.solutions[i].cost < self.bestSolution.cost):
            if (self.checkViolateConstraints(i) == False):
                self.bestSolution = copy.deepcopy(self.solutions[i])

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def generateCuckooEgg(self):
        # new solution by Lévy flights around the current best solutions
        beta = 1.5
        sigma = 0.6966

        # we add a temporary solution in first position
        self.solutions.insert(0, copy.deepcopy(self.solutions[0]))
        for j in range(0, self.searchSpace.dimension):
            self.solutions[0][j] += self.alpha * self.gauss(0., sigma) / pow(abs(self.gauss()), 1 / beta)

        # boundary check
        self.checkBoundary(0)

        # evaluate the new solution
        self.evalSolution(0)

        # choose a random nest, excluding the best (which is at index 1 now)
        k = self.randomInt(2, self.size)

        # if the new solution is better replace the selected nest
        if (self.solutions[0].cost < self.solutions[k].cost):
            self.solutions[k] = copy.deepcopy(self.solutions[0])

        # remove the temporary solution
        self.solutions.pop(0)

        # a pa fraction of the sub-optimal solution are replaced by new ones
        to_replace = round(self.pa * self.size)
        for i in range(1, to_replace):
            self.solutions[self.size-i] = Individual(self.searchSpace.random())

            # evaluate the new solutions
            self.evalSolution(self.size-i)


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class CS(Algorithm):

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, obj_function, population: Population,
                 discovery_rate: float = 0.25, step_size: float = 1.0) -> None:
        super().__init__(obj_function)
        self.population = CSPopulation(population, discovery_rate, step_size)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def minimize(self) -> None:
        maxIter = self.population.config.nmax_iter

        # if the population is not initialized, do it randomly
        if (self.population.isInitialized == False):
            if (self.population.config.silent == False):
                print("Generating the initial population...\n")
            self.population.initRandom() # also evaluate
        else:
            # Evaluate the cost for the population
            self.population.evaluate()
        
        # sort the population
        self.population.sort()
  
        # Update the cost history
        self.costHistory = np.append(self.costHistory, self.population.bestSolution.cost)

        # Update the population position history
        if (self.savePopPositions == True): 
            self.population.fillHistory()

        n_sc = 0
        for nIter in tqdm(range(0, maxIter)):

            # scale the penalty coefficient for
            # constrained optimization
            self.population.scalePenaltyCoeff()
            
            # generate a cuckoo egg
            self.population.generateCuckooEgg()

            # sort the population
            self.population.sort()

            # Update the cost history
            self.costHistory = np.append(self.costHistory, self.population.bestSolution.cost)

            # Update the population position history
            if (self.savePopPositions == True): 
                self.population.fillHistory()
            
            # Check on same cost iterations
            if (self.population.config.nmax_iter_same_cost > 0):
                if (nIter > 0 and self.population.bestSolution.cost < np.inf and
                    math.isclose(self.costHistory[nIter-1], self.costHistory[nIter], rel_tol=self.population.config.same_cost_rel_tol)):
                    n_sc += 1
                else:
                    n_sc = 0

                if (n_sc > self.population.config.nmax_iter_same_cost):
                    self.costHistory.resize(nIter+1)
                    break

        # write the results
        self.result = OptResult("CS", nIter, self.population.size, self.population.config,
                                self.population.bestSolution, self.population.searchSpace.parameters)
        

