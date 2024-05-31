import random
import numpy

from evomip.Parameter import Parameter
from evomip.Constraint import Constraint

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
class SearchSpace: 

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, parameters: list[Parameter], constraints: list[Constraint] = []) -> None:
        self.dimension = len(parameters)
        self.parameters = parameters
        self.constraints = constraints
        self.gen_point = numpy.empty(self.dimension)
        self.constr_init_pop = False


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def setSeed(self, t_seed: int) -> None:
        random.seed(t_seed)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def setCostrInitPop(self, t: bool) -> None:
        self.constr_init_pop = t


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def ckeckConstraint(self) -> bool:
        for constraint in self.constraints:
            g = constraint.func
            inequality = constraint.inequality
            tmp_d = g(self.gen_point)

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
    def randomParameter(self, i: int) -> float:
        return random.uniform(self.parameters[i].getMin(), self.parameters[i].getMax())


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def random(self) -> numpy.array:
        for i in range(0, self.dimension):
            self.gen_point[i] = self.randomParameter(i)

        # in case of a constrained optimization, check is
        # the solution violates any constraint
        if (self.constr_init_pop == True):
            if (self.ckeckConstraint()): 
                return self.random()

        return self.gen_point


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __getitem__(self, index: int):
        return self.parameters[index]
