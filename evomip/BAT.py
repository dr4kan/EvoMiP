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
class BATIndividual(Individual):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, position: np.array) -> None:
        super().__init__(position)
        self.freq: float = 0.
    
    
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class BATPopulation(Population):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, population: Population, initialLoudness: float, 
                 alpha: float, initialPulseRate: float, gamma: float,
                 fmin: float, fmax: float) -> None:
        super().__init__(population.size, population.objectiveFunction, 
                         population.searchSpace, population.config)
        self.solutions = [BATIndividual(np.empty(self.searchSpace.dimension))] * self.size
        self.initialLoudness = initialLoudness   # Initial loudness
        self.alpha = alpha                       # Parameter in [0, 1] to control how quickly the loudness changes
        self.initialPulseRate = initialPulseRate
        self.gamma = gamma
        self.fmin = fmin  # Minimum frequency 
        self.fmax = fmax  # Maximum frequency 
        self.loudness = self.initialLoudness
        self.pulseRate = self.initialPulseRate * (1 - math.exp(-self.gamma))
        
        
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def updateParameters(self, t: int) -> None:
        self.loudness = self.loudness * self.alpha
        self.pulseRate = self.initialPulseRate * (1 - math.exp(-self.gamma * (t+1)))
        
        
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def initRandom(self) -> None:
        self.n_valid_solutions = 0
        for i in range(0, self.size):
            self.solutions[i] = BATIndividual(self.searchSpace.random())
            self.solutions[i].freq = self.uniform(self.fmin, self.fmax)
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
    def moveBats(self) -> None:
        update_p: bool = False
        update_l: bool = False
        # we add a temporary solution in first position
        self.solutions.insert(0, BATIndividual(self.searchSpace.random()))

        # Loop on the population of bats
        for i in range(1, self.size):

            if (self.rand() < self.pulseRate):
                update_p = True
            if (self.rand() < self.loudness):
                update_l = True
   
            # update the frequency
            self.solutions[0].freq = self.uniform(self.fmin, self.fmax)
    
            # Loop on dimension
            for j in range(0, self.searchSpace.dimension):

                # update position and velocity
                self.solutions[0].velocity[j] = self.solutions[0].velocity[j] + (self.solutions[0][j] - self.bestSolution[j])*self.solutions[0].freq
                self.solutions[0][j] = self.solutions[0][j] + self.solutions[0].velocity[j]
                
                # improving the best solution
                if (update_p == True):
                    self.solutions[0][j] = self.bestSolution[j] + self.uniform(-1., 1.)*self.loudness
                
            # boundary check
            self.checkBoundary(0)
            
            # evaluate tmp
            self.evaluateCost(0)
            
            # conditionally save of the new solution
            if (update_l == True and self.solutions[0].cost < self.solutions[i].cost):
                self.solutions[i] = copy.deepcopy(self.solutions[0])

            # update the best solution
            # this is done automatically when evaluating tmp
        
        # remove the temporary solution
        self.solutions.pop(0)


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class BAT(Algorithm):

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, obj_function, population: Population, initialLoudness: float = 1.5, 
                 alpha: float = 0.95, initialPulseRate: float = 0.5, gamma: float = 0.9,
                 fmin: float = 0., fmax: float = 2.) -> None:
        super().__init__(obj_function)
        self.population = BATPopulation(population)
        self.population.initialLoudness = initialLoudness
        self.population.alpha = alpha
        self.population.initialPulseRate = initialPulseRate
        self.population.gamma = gamma
        self.population.fmin = fmin
        self.population.fmax = fmax


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

            # update the a parameter
            self.population.updateParameters(nIter)
            
            # move the whales
            self.population.moveBats()
            
            # Evaluate the cost for the population
            self.population.evaluate()
            
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
        self.result = OptResult("BAT", nIter, self.population.size, self.population.config,
                                self.population.bestSolution, self.population.searchSpace.parameters)
        

