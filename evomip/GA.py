import numpy as np
from tqdm import tqdm
import math
import copy

from evomip.Algorithm import *
from evomip.Population import Population
from evomip.Individual import Individual

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class GAIndividual(Individual):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, position: np.array) -> None:
        super().__init__(position)
        self.indicator: int = 0 # when 0 the cost needs to be revaluated
        
        
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def setIndicatorUp(self):
        self.indicator = 1


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def setIndicatorDown(self):
        self.indicator = 0
    
    
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class GAPopulation(Population):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, population: Population, keep_fraction: float, 
                 mutation_rate: float) -> None:
        super().__init__(population.size, population.objectiveFunction, 
                         population.searchSpace, population.config)
        self.solutions = [GAIndividual(np.empty(self.searchSpace.dimension))] * self.size
        self.keep_fraction = keep_fraction     # selection rate
        self.mutation_rate = mutation_rate     # mutation rate
                
        # number of chromosomes that survives to selection
        self.keep   = int(self.keep_fraction * self.size)
        
        # vector of probabilities used in the Roulette Wheel selection
        self.prob = []
        
        k = self.keep * (self.keep + 1) / 2
        self.prob.append(self.keep / k)
        for i in range(1, self.keep):
            self.prob.append((self.keep - i + 1) / k + self.prob[i - 2])
   
        
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def initRandom(self) -> None:
        self.n_valid_solutions = 0
        for i in range(0, self.size):
            self.solutions[i] = GAIndividual(self.searchSpace.random())
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
    def crossover(self):
        # generate offspring
        for i in range(0, self.size - self.keep, 2):
            self.solutions[self.size - 1 - i].setIndicatorDown()
            self.solutions[self.size - 2 - i].setIndicatorDown()
            ma = 0 
            pa = 0

            # mother and father
            ra1 = self.rand()
            ra2 = self.rand()
            for u in range(1, self.keep):
                if (ra1 > self.prob[u - 1] and ra1 <= self.prob[u]):
                    ma = u
                if (ra2 > self.prob[u - 1] and ra2 <= self.prob[u]): 
                    pa = u

            for k in range(0, self.searchSpace.dimension):
                beta = self.rand()
                self.solutions[self.size - 1 - i][k] = self.solutions[ma][k] - beta * (self.solutions[ma][k] - self.solutions[pa][k])
                self.solutions[self.size - 2 - i][k] = self.solutions[pa][k] + beta * (self.solutions[ma][k] - self.solutions[pa][k])


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def mutation(self):
        mutat = int(self.mutation_rate * self.size * self.searchSpace.dimension)

        for i in range(0, mutat):
            ra1 = self.randomInt(0, self.searchSpace.dimension)
            ra2 = self.randomInt(1, self.size)
            self.solutions[ra2][ra1] = self.searchSpace.randomParameter(ra1)
            self.solutions[ra2].setIndicatorDown()


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class GA(Algorithm):

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, obj_function, population: Population, 
                 keep_fraction: float = 0.4, mutation_rate: float = 0.1) -> None:
        super().__init__(obj_function)
        self.population = GAPopulation(population, keep_fraction, mutation_rate)


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

            # mating, crossover, evaluate and sort
            self.population.crossover()

            # mutation
            self.population.mutation()

            # evaluate the cost for the population
            self.population.evaluate()

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
        self.result = OptResult("GA", nIter, self.population.size, self.population.config,
                                self.population.bestSolution, self.population.searchSpace.parameters)
        

