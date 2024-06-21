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
class SAIndividual(Individual):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, position: np.array) -> None:
        super().__init__(position)
        self.success = [0] * position.size
        self.position_best = np.copy(position)
        self.hasVelocity = True
        
    
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def upSuccess(self, j: int):
        self.success[j] += 1
        
    
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def resetSuccess(self):
        self.success = [0] * self.position.size

    
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class SAPopulation(Population):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, population: Population, T0: float, Ns: float, Nt: float, 
                 c_step: float, Rt: float, Wmin: float, Wmax: float) -> None:
        super().__init__(population.size, population.objectiveFunction, 
                         population.searchSpace, population.config)
        self.solutions = [SAIndividual(np.empty(self.searchSpace.dimension))] * self.size
        
        self.T0   = T0     # Initial temperature */
        self.Ns   = Ns     # Maximum number of cycles before step variation  */
        self.Nt   = Nt     # Maximum number of cycles before temperature variation  */
        self.C    = c_step # Step criterion  */
        self.Rt   = Rt     # Reduction coefficient for temperature  */
        self.Wmin = Wmin   # Minimum value of the weight employed in the formula for the starting point  */
        self.Wmax = Wmax   # Maximum value of the weight employed in the formula for the starting point  */
        self.Prob = []     # Vector of probabilities used in the Roulette Wheel selection */
        
        k = self.size * (self.size + 1) / 2
        self.Prob.append(self.size / k)
        for i in range(1, self.size):
            self.Prob.append((self.size - i + 1) / k + self.Prob[i - 2])
   
        
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def initRandom(self) -> None:
        self.n_valid_solutions = 0
        for i in range(0, self.size):
            self.solutions[i] = SAIndividual(self.searchSpace.random())
            self.solutions[i].velocity = self.searchSpace.randomVelocity()
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
    def move(self):
        # loop on population
        for i in range(0, self.size):
                
            # loop on dimension
            for h in range(0, self.searchSpace.dimension):
                # we add a temporary solution in first position
                self.solutions.insert(0, copy.deepcopy(self.solutions[i]))
            
                self.solutions[0][h] += self.uniform(-1., 1.) * self.solutions[i].velocity[h]

                # boundary check
                self.checkBoundary(0)

                # evaluate the new solution
                self.evalSolution(0)
        
                if (self.solutions[0].cost < self.solutions[i].cost or 
                    self.rand() < math.exp((self.solutions[i].cost - self.solutions[0].cost) / self.T0)):
                    self.solutions[i] = copy.deepcopy(self.solutions[0])
                    self.solutions[i].upSuccess(h)

                self.solutions.pop(0)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def setVelocity(self):
        # loop on population
        for i in range(0, self.size):
            
            for h in range(0, self.searchSpace.dimension):
                if (self.solutions[i].success[h] > 0.6 * self.Ns):
                    self.solutions[i].velocity[h] =  self.solutions[i].velocity[h] * (1. + self.C * ((self.solutions[i].success[h] / self.Ns - 0.6) / 0.4))
                elif (self.solutions[i].success[h] < 0.4 * self.Ns):
                    self.solutions[i].velocity[h] =  self.solutions[i].velocity[h] / (1. + self.C * (0.4 - (self.solutions[i].success[h] / self.Ns) / 0.4))
            
            self.solutions[i].resetSuccess()


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def setStartingPoint(self, t: int, nmax_iter: int):
        w = self.Wmax - (t / nmax_iter) * (self.Wmax - self.Wmin)

        # select elite solutions by roulette wheel
        # loop on population
        for i in range(0, self.size):
            
            elite1: int = 0
            elite2: int = 0
            ra1 = self.rand()
            ra2 = self.rand()
            for u in range(1, self.size):
                if (ra1 > self.Prob[u - 1] and ra1 <= self.Prob[u]):
                    elite1 = u
                if (ra2 > self.Prob[u - 1] and ra2 <= self.Prob[u]): 
                    elite2 = u

            # set initial point on the basis of the current position and elite best position
            for j in range(0, self.searchSpace.dimension):
                ra3 = self.rand()
                ra4 = self.rand()

                best_position = self.solutions[i].position_best[j]
                self.solutions[i][j] = best_position + w * (ra3 * (self.solutions[elite1].position_best[j] - best_position) + ra4 * (self.solutions[elite2].position_best[j] - best_position))

            # boundary check
            self.checkBoundary(i)


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class SA(Algorithm):

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, population: Population, T0: float = 50., 
                 Ns: float = 3., Nt: float = 3., c_step: float = 2., 
                 Rt: float = 0.85, Wmin: float = 0.25, Wmax: float = 1.25) -> None:
        super().__init__(population.objectiveFunction)
        self.population = SAPopulation(population, T0, Ns, Nt, c_step, 
                                       Rt, Wmin, Wmax)


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
            
        # set the starting point
        self.population.setStartingPoint(0, maxIter)
            
        # Update the cost history
        self.costHistory = np.append(self.costHistory, self.population.bestSolution.cost)

        # Update the population position history
        if (self.savePopPositions == True): 
            self.population.fillHistory()
        
        n_sc = 0
        for nIter in tqdm(range(0, maxIter), disable = self.population.config.silent):

            # scale the penalty coefficient for
            # constrained optimization
            self.population.scalePenaltyCoeff()

            for m in range(0, int(self.population.Nt)):
                for j in range(0, int(self.population.Ns)):
                    self.population.move()

                # Update the step vector
                self.population.setVelocity()

            # Update the temperature
            self.population.T0 *= self.population.Rt

            # sort the population
            self.population.sort()

            # set the starting point
            self.population.setStartingPoint(nIter, maxIter)
            
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
        self.result = OptResult("SA", nIter, self.population.size, self.population.config,
                                self.population.bestSolution, self.population.searchSpace.parameters)
        

