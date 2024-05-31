import numpy as np
from tqdm import tqdm
import math

from evomip.Algorithm import *
from evomip.Population import Population

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class WOAPopulation(Population):
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, population: Population) -> None:
        super().__init__(population.size, population.objectiveFunction, 
                         population.searchSpace, population.config)
        self.a: float = 0.
        self.a2: float = 0. 

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def updateParameters(self, t: int, nmax_iter: int) -> None:
        self.a  = 2. - t*(2./nmax_iter)
        self.a2 = -1. + t*((-1.)/nmax_iter)

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def moveWhales(self) -> None:
        r1, r2, A, C, b, l, p, D_tmp, D_best, distance = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        rw = 0

        # Loop on the population of whales
        for i in range(0, self.size):

            r1 = self.rand()
            r2 = self.rand()
            A  = 2*self.a*r1-self.a
            C  = 2*r2
            b  = 1.
            l  = (self.a2-1)*self.rand()+1
            p  = self.rand()

            # Loop on dimension
            for j in range(0, self.searchSpace.dimension):

                if (p < 0.5):

                    if (abs(A) >= 1):
                        # random whale
                        rw             = self.randomInt(0, self.size)
                        tmp            = self.solutions[rw]
                        D_tmp          = abs(C*tmp[j] - self.solutions[i][j])
                        self.solutions[i][j] = tmp[j] - A*D_tmp

                    else:
                        # encircling prey
                        D_best         = abs(C*self.bestSolution[j] - self.solutions[i][j])
                        self.solutions[i][j] = self.bestSolution[j]-A*D_best

                else:
                    # distance whale to the prey
                    distance       = abs(self.bestSolution[j] - self.solutions[i][j])
                    self.solutions[i][j] = distance*math.exp(b*l)*math.cos(l*2*math.pi) + self.bestSolution[j]

            # boundary check
            self.checkBoundary(i)


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class WOA(Algorithm):

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, obj_function, population: Population) -> None:
        super().__init__(obj_function)
        self.population = WOAPopulation(population)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def minimize(self) -> None:
        maxIter = self.population.config.nmax_iter

        # if the population is not initialized, do it randomly
        if (self.population.isInitialized == False):
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
            self.population.moveWhales()

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

        self.population.sort()
        print(self.population)

        # write the results
        self.result = OptResult("WOA", nIter, self.population.size, self.population.config,
                                self.population.bestSolution, self.population.searchSpace.parameters)