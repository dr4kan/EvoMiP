import numpy as np

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