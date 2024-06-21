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
class GSAIndividual(Individual):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, position: np.array) -> None:
        super().__init__(position)
        self.hasVelocity = True
        self.mass: float = 0.
        
        
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class GSAPopulation(Population):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, population: Population, grav: float, 
                 grav_evolution: float) -> None:
        super().__init__(population.size, population.objectiveFunction, 
                         population.searchSpace, population.config)
        self.solutions = [GSAIndividual(np.empty(self.searchSpace.dimension))] * self.size
        self.grav = grav
        self.grav_evolution = grav_evolution
        
        
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def initRandom(self) -> None:
        self.n_valid_solutions = 0
        for i in range(0, self.size):
            self.solutions[i] = Individual(self.searchSpace.random())
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
    def changeMasses(self) -> None:
        sum_mass = 0.
        mass = []

        worst_cost = self.solutions[self.size-1].cost
        best_cost = self.solutions[0].cost
        delta_mass = worst_cost - best_cost

        # Loop on the population
        for i in range(0, self.size):
            mass.append((worst_cost - self.solutions[i].cost)/delta_mass)
            sum_mass += mass[i]
        
        # Loop on the population
        for i in range(0, self.size):
            self.solutions[i].mass = mass[i]/sum_mass
         
  
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def changeVelocities(self, t: int, nmax_iter: int):
        grav = self.grav * math.exp(-self.grav_evolution * t / nmax_iter)

        Kbest = self.size * (nmax_iter - t) / nmax_iter
        accel = 0.

        # compute the distances between planets
        distance = np.zeros((self.size, self.size))
        
        # Loop on the population
        for i in range(0, self.size):
            # Loop on the population
            for j in range(0, self.size):
            
                if (i < j):
                    for v in range(self.searchSpace.dimension):
                        distance[i][j] += np.power(self.solutions[j][v] - self.solutions[i][v], 2)
                    distance[i][j] = np.sqrt(distance[i][j])
                else:
                    distance [i][j] = distance [j][i]
            
        

        # compute the resulting acceleration of the i-planet due to the external gravitational forces
        for i in range(0, self.size):
            for k in range(self.searchSpace.dimension):
                ref_accel = 0.
                for j in range(0, self.size):
                    if (distance[i][j] > 0. and j < Kbest):
                        accel = grav * self.solutions[j].mass / (distance[i][j]) * (self.solutions[j][k] - self.solutions[i][k])
                        ref_accel += self.rand() * accel

                # compute the velocity taking into account the previous velocity and the forces applied
                vel = self.solutions[i].velocity[k] * self.rand() + ref_accel

                self.solutions[i].velocity[k] = vel
    

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def movePlanets(self, t: int, nmax_iter: int) -> None:
        # change the mass of the planets
        self.changeMasses()

        # change the velocity of planets
        self.changeVelocities(t, nmax_iter)

        # Loop on the population
        for i in range(0, self.size):
            # Loop on dimension
            for j in range(0, self.searchSpace.dimension):
                self.solutions[i][j] = self.solutions[i][j] + self.solutions[i].velocity[j]

            # boundary check
            self.checkBoundary(i)


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class GSA(Algorithm):

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, obj_function, population: Population, grav: float = 1000., 
                 grav_evolution: float = 20.) -> None:
        super().__init__(obj_function)
        self.population = GSAPopulation(population, grav, grav_evolution)


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

            # sort the population
            self.population.sort()

            # move the planets
            self.population.movePlanets(nIter, maxIter)

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
        self.result = OptResult("GSA", nIter, self.population.size, self.population.config,
                                self.population.bestSolution, self.population.searchSpace.parameters)
        

