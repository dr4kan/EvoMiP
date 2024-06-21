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

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class GWOPopulation(Population):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, population: Population) -> None:
        super().__init__(population.size, population.objectiveFunction, 
                         population.searchSpace, population.config)
        self.a: float = 0.
        self.alpha = self.solutions[0]
        self.beta  = self.solutions[1]
        self.delta = self.solutions[2]


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def updateParameters(self, t: int, nmax_iter: int) -> None:
        self.a = 2 - t*(2./nmax_iter)


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
        
        # update the alpha, beta and delta wolves
        if (self.solutions[i].cost < self.alpha.cost):
            self.alpha = self.solutions[i]
        elif (self.solutions[i].cost < self.beta.cost):
            self.beta = self.solutions[i]
        elif (self.solutions[i].cost < self.delta.cost):
            self.delta = self.solutions[i]
        
        # check is the solution violates the constraints
        if (self.solutions[i].cost < self.bestSolution.cost):
            if (self.checkViolateConstraints(i) == False):
                self.bestSolution = copy.deepcopy(self.solutions[i])
                
                
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def moveWolves(self) -> None:
        r1, r2, A, C, D_alpha = 0., 0., 0., 0., 0.
        D_beta, D_delta, X1, X2, X3 = 0., 0., 0., 0., 0.

        # Loop on the population of wolves
        for i in range(0, self.size):

            # Loop on dimension
            for j in range(0, self.searchSpace.dimension):
                
                r1 = self.rand()
                r2 = self.rand()
                A  = 2*self.a*r1-self.a
                C  = 2*r2
                D_alpha = abs(C*self.alpha[j]-self.solutions[i][j])
                X1 = self.alpha[j]-A*D_alpha

                r1 = self.rand()
                r2 = self.rand()
                A  = 2*self.a*r1-self.a
                C  = 2*r2
                D_beta = abs(C*self.beta[j]-self.solutions[i][j])
                X2 = self.beta[j]-A*D_beta
                
                r1 = self.rand()
                r2 = self.rand()
                A  = 2*self.a*r1-self.a
                C  = 2*r2
                D_delta = abs(C*self.delta[j]-self.solutions[i][j])
                X3 = self.delta[j]-A*D_delta

                self.solutions[i][j] = (X1+X2+X3)/3.
                
            # boundary check
            self.checkBoundary(i)


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class GWO(Algorithm):

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, obj_function, population: Population) -> None:
        super().__init__(obj_function)
        self.population = GWOPopulation(population)


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
            self.population.updateParameters(nIter, maxIter)

            # move the whales
            self.population.moveWolves()

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
        self.result = OptResult("GWO", nIter, self.population.size, self.population.config,
                                self.population.bestSolution, self.population.searchSpace.parameters)
        