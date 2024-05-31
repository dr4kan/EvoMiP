from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy as np

from evomip.Population import Population
from evomip.Config import Config
from evomip.Individual import Individual
from evomip.Parameter import Parameter

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
@dataclass
class OptResult:
    algo: str
    nIter: int
    popSize: int
    config:  Config
    bestSolution: Individual
    parameters: list[Parameter]
    
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, algo: str = "", nIter: int = 0, popSize: int = 0, config: Config = None,
                 bestSolution: Individual = None, parameters: list[Parameter] = []) -> None:
        self.algo = algo
        self.nIter = nIter
        self.popSize = popSize
        self.config = config
        self.bestSolution = bestSolution
        self.parameters = parameters


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class Algorithm:

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, obj_function) -> None:
        self.obj_function = obj_function
        self.costHistory = np.empty(0)
        self.savePopPositions = False
        self.result = OptResult()


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def plotCostHistory(self) -> None:
        ys = self.costHistory
        xs = [x for x in range(len(ys))]
        plt.plot(xs, ys)
        plt.show()


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def setSavePopulationPositions(self, t: bool) -> None:
        self.save_population = t


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def summary(self) -> None:
        print("\n       EmiP Minimization Results")
        print("--------------------------------------------")
        print("         minimizer | ", self.result.algo, sep = "") 
        print("        iterations | ", self.result.nIter+1, sep = "") 
        print("   population size | ", self.result.popSize, sep = "")
        print("        OOB method | ", self.result.config.oobMethod, sep = "")
        print(" constraint method | ", self.result.config.constraintsMethod, sep = "")
        print("         best cost | ", self.result.bestSolution.cost, sep = "")
        print("     best solution | ", self.result.parameters[0].name, " = ", self.result.bestSolution[0], sep = "")
        for i in range(1, len(self.result.parameters)):
            print("                   | ", self.result.parameters[i].name, " = ", self.result.bestSolution[i], sep = "")
        print("--------------------------------------------")






