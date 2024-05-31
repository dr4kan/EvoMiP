import numpy as np
import random
import copy

from evomip.SearchSpace import SearchSpace
from evomip.Individual import Individual
from evomip.Config import *

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class Population:

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, size: int, objective_function: 'function', 
                 searchSpace: SearchSpace, config: Config = Config()) -> None:

        self.objectiveFunction = objective_function

        # search space
        self.searchSpace = searchSpace
        
        # config
        self.config = config
        
        # population
        self.size = size
        self.solutions = [Individual(np.empty(searchSpace.dimension))] * size
        self.isInitialized = False
        self.bestSolution = self.solutions[0]
        self.history = []
        
        # seed
        self.seed = self.config.seed
        self.setSeed(self.seed)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __getitem__(self, key):
        return self.solutions[key]
    
    
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def scalePenaltyCoeff(self) -> None:
        self.config.penalty_coeff = min(self.config.max_penalty_param, self.config.penalty_coeff*self.config.penalty_scaling)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def fillHistory(self) -> None:
        self.history.append(np.copy(self.solutions))
    
    
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def printHistory(self) -> None:
        for i in range(0, len(self.history)):
            print("\nIteration #", i, sep="")
            print(self.history[i])


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def setSeed(self, t_seed: int) -> None:
        if (t_seed > 0):
            self.searchSpace.setSeed(t_seed)
            random.seed(t_seed)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def rand(self) -> float:
        return random.random()
    
    
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def randomInt(self, minVal: int, maxVal: int) -> int:
        return random.randint(minVal, maxVal-1)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def initRandom(self) -> None:
        #if (gSilent == False):
        print("Generating the initial population...\n")

        for i in range(0, self.size):
            self.solutions[i] = Individual(self.searchSpace.random())

        self.evaluate()
        self.isInitialized = True
           
           
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/  
    def sort(self) -> None:
        self.solutions.sort(key=lambda x: x.cost, reverse=False)
             
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def checkViolateConstraints(self, i: int) -> bool:
        for constraint in self.searchSpace.constraints:
            g = constraint.func
            inequality = constraint.inequality
            tmp_d = g(self.solutions[i].position)

            if (inequality == "<" and tmp_d >= 0):
                return True
            elif (inequality == "<=" and tmp_d > 0):
                return True
            elif (inequality == ">=" and tmp_d < 0):
                return True
            elif (inequality == ">" and tmp_d <= 0):
                return True

        return False
             
                
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def checkBoundary(self, i: int) -> None:
    
        if (len(self.searchSpace.constraints) > 0 and self.config.oobMethod == OOBMethod.DIS): 
        
            # Disregard the out-of-bound solution and generate new ones (DIS)
            self.solutions[i] = Individual(self.searchSpace.random())

        else:

            # loop on dimension
            for j in range(0, self.searchSpace.dimension):

                if (self.config.oobMethod == OOBMethod.PBC): # Periodic Boundary Condition (PBC)

                    if (self.solutions[i][j] < self.searchSpace[j].getMin()): 
                        self.solutions[i][j] = self.searchSpace[j].getMax() - abs(self.solutions[i][j] - self.searchSpace[j].getMin())
                    if (self.solutions[i][j] > self.searchSpace[j].getMax()): 
                        self.solutions[i][j] = self.searchSpace[j].getMin() + abs(self.searchSpace[j].getMax() - self.solutions[i][j])

                    if (self.solutions[i][j] < self.searchSpace[j].getMin() or self.solutions[i][j] > self.searchSpace[j].getMax()): 
                        self.solutions[i][j] = self.searchSpace.randomParameter(j)

                elif (self.config.oobMethod == OOBMethod.BAB): # Place out-of-bound solutions back at the boundaries (BAB)

                    if (self.solutions[i][j] < self.searchSpace[j].getMin()): 
                        self.solutions[i][j] = self.searchSpace[j].getMin()
                    if (self.solutions[i][j] > self.searchSpace[j].getMax()): 
                        self.solutions[i][j] = self.searchSpace[j].getMax()

                elif (self.config.oobMethod == OOBMethod.DIS): # Disregard the out-of-bound solution and generate new ones (DIS)

                    if (self.solutions[i][j] < self.searchSpace[j].getMin() or self.solutions[i][j] > self.searchSpace[j].getMax()): 
                        self.solutions[i][j] = self.searchSpace.randomParameter(j)

                elif (self.config.oobMethod == OOBMethod.RBC): # Reflective Boundary Condition (RBC)

                    if (self.solutions[i][j] < self.searchSpace[j].getMin()):
                        self.solutions[i][j] = 2 * self.searchSpace[j].getMin() - self.solutions[i][j]
                    if (self.solutions[i][j] > self.searchSpace[j].getMax()): 
                        self.solutions[i][j] = 2 * self.searchSpace[j].getMax() - self.solutions[i][j]

                    if (self.solutions[i][j] < self.searchSpace[j].getMin() or self.solutions[i][j] > self.searchSpace[j].getMax()): 
                        self.solutions[i][j] = self.searchSpace.randomParameter(j)

                    if (self.solutions[i].hasVelocity == True):
                        self.solutions[i].velocity[j] = -self.solutions[i].velocity[j]                         


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def constraintsPenaltyMethod(self, i: int) -> None:
        penalty: float = 0.

        for constraint in self.searchSpace.constraints:
            g = constraint.func
            inequality = constraint.inequality
            tmp_d = g(self.solutions[i].position)

            if ((inequality == "<" and tmp_d >= 0) or
                (inequality == "<=" and tmp_d > 0) or
                (inequality == ">=" and tmp_d < 0) or
                (inequality == ">" and tmp_d <= 0)):
                penalty += abs(tmp_d)

        self.solutions[i].cost = self.objectiveFunction(self.solutions[i].position) + self.config.penalty_coeff*penalty

            
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def constraintsBarrierMethod(self, i: int) -> None:
        if (self.checkViolateConstraints(i) == True):
            self.solutions[i].cost = np.inf
        else:
            self.solutions[i].cost = self.objectiveFunction(self.solutions[i].position)

    
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
    def evaluateCost(self, i: int) -> None:

        # In case of integer parameters
        # loop on dimension
        for j in range(0, self.searchSpace.dimension):
            if (self.searchSpace[j].getIsInteger() == True):
                self.solutions[i][j] = round(v[j]) # round the parameter
                if (self.solutions[i][j] < self.searchSpace[j].getMin()):
                    self.solutions[i][j] += 1
                elif (self.solutions[i][j] > self.searchSpace[j].getMax()):
                    self.solutions[i][j] -= 1
                
        if (self.config.constraintsMethod == ConstraintsMethod.PTY):
            self.constraintsPenaltyMethod(i)
        elif (self.config.constraintsMethod == ConstraintsMethod.BAR):
            self.constraintsBarrierMethod(i)
        else:
            # if violate any of the contraints, regenerate
            if (self.checkViolateConstraints(i)):
                self.solutions[i] = Individual(self.searchSpace.random())
                self.solutions[i].cost = self.objectiveFunction(self.solutions[i].position)
    
                                            
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __repr__(self) -> str:
        out = ""
        for i in range(0, self.size):
            out += str(self.solutions[i]) + "\n"
        return out