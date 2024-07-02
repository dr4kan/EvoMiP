# EvoMiP: Evolutionary minimizer for Python

Gradient-based algorithms, such as *Gradient descent*, *Back propagation*, *Conjugate gradient* and *Adaptive Moment Estimation (ADAM)*, despite the popularity present problems like the sensitivity to the choice of the initial point and step size, the ease of getting stuck in local optima, and the need of differentiable objective functions.

In last decade, the interest on metaheuristic nature inspired algorithms has been growing steadily, due to their flexibility and effectiveness. EvoMiP is a package for Python (based on *EmiR*, a popular R package from the same authors) which implements several methauristic algorithms for optimization problems:

* *Artificial Bee Colony Algorithm*
* *Bat Algorithm*
* *Cuckoo Search*
* *Genetic Algorithm*
* *Gravitationl Search Algorithm*
* *Grey Wolf Optimization*
* *Harmony Search*
* *Improved Harmony Search*
* *Moth-flame Optimization*
* *Particle Swarm Optimization*
* *Simulated Annealing*
* *Whale Optimization Algorithm*

Unlike other available tools, EvoMiP can be used not only for unconstrained problems, but also for problems subjected to inequality constraints and for integer or mixed-integer problems. 

## Installation

    pip3 install git+https://github.com/dr4kan/EvoMiP.git#egg=evomip

## Examples

### Unconstrained optimization of the 4-dimensional miele cantrell function

The function is defined and evaluated in $x_i \in [−2, 2]$, for $i = 1, . . . , 4$: 

$f (x) = (e^{−x_1} − x_2)^4 + 100(x_2 − x_3)^6 + \tan^4(x_3 − x_4) + x_1^8$.

The function has a global minimum: $f(0, 1, 1, 1) = 0$.

```
from evomip.Config import Config
from evomip.Parameter import *
from evomip.SearchSpace import SearchSpace
from evomip.Population import Population
from evomip.BAT import BAT
import numpy as np

# definition of the objective function
def miele_cantrell(x: np.array):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    return (np.exp(-x1) - x2)**4 + 100*(x2 - x3)**6 + (np.tan(x3 - x4))**4 + x1**8

# configuration of the optimization parameters
config = Config(nmax_iter=100, nmax_iter_same_cost=0)

# definition of the search space
l = createListParameters("x", 4, -2., 2.)
sspace = SearchSpace(l)

# defition of the population
population = Population(100, miele_cantrell, sspace, config)

# running the algorithm
algo = BAT(population)
algo.minimize()

# summary of the results
algo.summary()
```
