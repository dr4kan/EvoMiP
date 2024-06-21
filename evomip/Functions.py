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

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# N-Dimensional SPHERE FUNCTION
#
# The function is usually evaluated in: 
# xi ∈ [-5.12, 5.12], for all i = 1, …, d
#
# xmin = (0, 0, ..., 0) and f(xmin) = 0
def sphere(x: np.array):    
    return np.sum(np.square(x))


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# N-Dimensional ACKLEY FUNCTION
#
# The function is usually evaluated in: 
# xi ∈ [-32.768, 32.768], for all i = 1, …, d
#
# xmin = (0, 0, ..., 0) and f(xmin) = 0
def ackley(x: np.array, a: float = 20., b: float = 0.2, c: float = 2*np.pi):
    d = x.size
    
    sum1 = np.sum(np.square(x))
    sum2 = np.sum(np.cos(c*x))
    
    term1 = -a * np.exp(-b*np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)
    
    return term1 + term2 + a + np.exp(1)
    
    
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# 2-Dimensional BUKIN FUNCTION N. 6
#
# The function is usually evaluated in: 
# x1 ∈ [-15, -5], x2 ∈ [-3, 3]
#
# xmin = (-10, 1) and f(xmin) = 0
def bukin6(x: np.array):
    x1 = x[0]
    x2 = x[1]
	
    term1 = 100 * np.sqrt(abs(x2 - 0.01*x1^2))
    term2 = 0.01 * abs(x1+10)
	
    return term1 + term2