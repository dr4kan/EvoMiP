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

import random
import numpy as np

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
        

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def setSeed(self, t_seed: int) -> None:
        random.seed(t_seed)


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def violateConstraints(self, gen_point: np.array) -> bool:
        for constraint in self.constraints:
            g = constraint.func
            inequality = constraint.inequality
            tmp_d = g(gen_point)

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
    def random(self) -> np.array:
        gen_point = np.zeros(self.dimension)
        
        for i in range(0, self.dimension):
            gen_point[i] = self.randomParameter(i)

        return gen_point


    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def randomVelocity(self, alpha: float = 1.) -> np.array:
        gen_vel = np.zeros(self.dimension)
        
        for i in range(0, self.dimension):
            delta = self.parameters[i].max_val - self.parameters[i].min_val
            gen_vel[i] = random.uniform(-delta*alpha, delta*alpha)
            
        return gen_vel
        

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __getitem__(self, index: int):
        return self.parameters[index]
