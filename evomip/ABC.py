###############################################################################
# EvoMiP: Evolutionary minimization for Python                                #
# Copyright (C) 2024 Davide Pagano                                            #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# any later version.                                                          #
#                                                                             #
# This program is distributed in the hope that it will be useful, but         #
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY  #
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License    #
# for more details: <https://www.gnu.org/licenses/>.                          #
###############################################################################

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
class ABCIndividual(Individual):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, position: np.array) -> None:
        super().__init__(position)
        
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def getFitness(self) -> float:
        if (self.cost >= 0):
            return 1./(1. + self.cost)

        return 1. + abs(self.cost)
        
        
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class ABCPopulation(Population):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, population: Population, employedFraction: float, nScouters: int) -> None:
        super().__init__(population.size, population.objectiveFunction, 
                         population.searchSpace, population.config)
        
        self.employedFraction = employedFraction
        self.nScouters = nScouters
        f = int(employedFraction * self.size)
        self.onlookers = self.size - f - self.nScouters
        self.solutions = [ABCIndividual(np.empty(population.searchSpace.dimension))] * f
        self.prob = np.zeros(f)
        self.trial = np.zeros(f)
        self.fitness_sum: float = 0.
        self.limit_scout = int(0.5 * self.size * self.searchSpace.dimension)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def updateParameters(self, t: int, nmax_iter: int) -> None:
        self.a = 2 - t*(2./nmax_iter)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def initRandom(self) -> None:
        self.n_valid_solutions = 0
        for i in range(0, len(self.solutions)):
            self.solutions[i] = ABCIndividual(self.searchSpace.random())
            if (self.checkViolateConstraints(i) == False):
                self.n_valid_solutions += 1
                
        if (self.n_valid_solutions < self.config.min_valid_solutions):
            self.initRandom()

        self.evaluate()
        self.isInitialized = True
        

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def evaluate(self) -> None:
        for i in range(0, len(self.solutions)):
            self.evalSolution(i)
            
            
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def evalSolution(self, i: int) -> None:
        self.evaluateCost(i)
        
        # check is the solution violates the constraints
        if (self.solutions[i].cost < self.bestSolution.cost):
            if (self.checkViolateConstraints(i) == False):
                self.bestSolution = copy.deepcopy(self.solutions[i])
            
            
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def getRandomPopulationIndex(self, i: int):
        j = i
        while(j == i):
            j = self.randomInt(0, len(self.solutions))
            
        return j
            

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def generateNewSolution(self, i: int):
                
        # Random index in population with k != i
        k = self.getRandomPopulationIndex(i)

        # Random index in the dimension
        j = self.randomInt(0, self.searchSpace.dimension)
                           
        self.solutions[0][j] += self.uniform(-1., 1.) * (self.solutions[0][j] - self.solutions[k][j])

        # Boundary check
        self.checkBoundary(0)

        # Evaluate the new solution
        self.evalSolution(0)
        
    
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def computeProbabilities(self):
        # loop on population
        for i in range(0, len(self.solutions)):
            self.prob[i] = self.solutions[i].getFitness() / self.fitness_sum
        
                
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def employedBeesEvaluation(self):
        # Each employed bee xi generates a new candidate solution
        # in the neighborhood of its present position
        self.fitness_sum = 0.
        
        # loop on population
        for i in range(0, len(self.solutions)):
            # we add a temporary solution in first position
            self.solutions.insert(0, copy.deepcopy(self.solutions[i]))
        
            # Generate a new solution
            self.generateNewSolution(i)

            # Greedy selection: if the fitness value of the solu is better than m_individuals[i]
            # then update m_individuals[i] with tmp, otherwise keep m_individuals[i] unchanged
            if (self.solutions[0].getFitness() > self.solutions[i].getFitness()):
                self.solutions[i] = copy.deepcopy(self.solutions[0])
                self.trial[i] = 0
            else:
                self.trial[i] += 1

            self.solutions.pop(0)
            self.fitness_sum += self.solutions[i].getFitness()


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def onlookerBeesEvaluation(self):
        # After all employed bees complete the search process, they share the information of their
        # food sources with the onlooker bees. An onlooker bee evaluates the nectar information
        # taken from all employed bees and chooses a food source with a probability related to its
        # nectar amount. This probabilistic selection is a roulette wheel selection.
        self.computeProbabilities()

        discarded_solutions = False
        for k in range(0, self.onlookers):
            sel = self.randomInt(0, len(self.solutions))
                    
            # we add a temporary solution in first position
            self.solutions.insert(0, copy.deepcopy(self.solutions[sel]))

            # generate a new solution
            self.generateNewSolution(k)

            # Greedy selection: if the fitness value of the solu is better than m_individuals[i]
            # then update m_individuals[i] with tmp, otherwise keep m_individuals[i] unchanged
            if (self.solutions[0].getFitness() > self.solutions[sel].getFitness()):
                self.solutions[sel] = copy.deepcopy(self.solutions[0])
                self.trial[sel] = 0
            else:
                self.trial[sel] += 1

            # If a position cannot be improved over a predefined number (called limit)
            # of cycles, then the food source is abandoned
            if (self.trial[sel] > self.limit_scout):
                self.solutions[sel] = ABCIndividual(self.searchSpace.random())
                self.evalSolution(sel)
                self.trial[sel] = 0
                discarded_solutions = True
                
            self.solutions.pop(0)

        # if no solution was discarded, generate new random solutions
        if (discarded_solutions == True):
            return

        s: int = 0
        for k in range(0, self.nScouters):
            s = self.randomInt(0, len(self.solutions))
            self.solutions[s] = ABCIndividual(self.searchSpace.random())
            self.evalSolution(s)
    

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class ABC(Algorithm):

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, obj_function, population: Population, employedFraction: float = 0.5,
                 nScouters: int = 1) -> None:
        super().__init__(obj_function)
        self.population = ABCPopulation(population, employedFraction, nScouters)


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

            # Employed bees work
            self.population.employedBeesEvaluation()

            # Onlooker bees work + scout bees work
            self.population.onlookerBeesEvaluation()
            
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
        self.result = OptResult("ABC", nIter, self.population.size, self.population.config,
                                self.population.bestSolution, self.population.searchSpace.parameters)
        

