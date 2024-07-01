# EvoMiP: Evolutionary minimizer for Python

Gradient-based algorithms, such as *Gradient descent*, *Back propagation*, *Conjugate gradient* and *Adaptive Moment Estimation (ADAM)*, despite the popularity present problems like the sensitivity to the choice of the initial point and step size, the ease of getting stuck in local optima, and the need of differentiable objective functions.

In last decade, the interest on metaheuristic nature inspired algorithms has been growing steadily, due to their flexibility and effectiveness. EvoMiP is a package for Python (based on *EmiR*, a popular R package from the same authors) which implements several methauristic algorithms for optimization problems:

* *Artificial Bee Colony Algorithm*;
* *Bat Algorithm*;
* *Cuckoo Search*;
* *Genetic Algorithm*;
* *Gravitationl Search Algorithm*;
* *Grey Wolf Optimization*;
* *Harmony Search*;
* *Improved Harmony Search*;
* *Moth-flame Optimization*;
* *Particle Swarm Optimization*;
* *Simulated Annealing*;
* *Whale Optimization Algorithm*. 

Unlike other available tools, EvoMiP can be used not only for unconstrained problems, but also for problems subjected to inequality constraints and for integer or mixed-integer problems. 


### Installation

    pip3 install git+https://github.com/dr4kan/EvoMiP.git#egg=evomip
