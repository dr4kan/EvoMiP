from dataclasses import dataclass
from enum import Enum

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# Possible treatment of out-of-boundary solutions:
# - PBC: Periodic Boundary Condition
# - RBC: Reflective Boundary Condition
# - BAB: Back At boundaries
# - DIS: Disregard the solution and generate a new one
class OOBMethod(Enum):
    PBC = 1
    RBC = 2
    BAB = 3
    DIS = 4


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# Possible treatment of constraints:
# - PTY: A penalty is applied to the cost
# - BAR: The cost is set to inf
# - DIS: Disregard the solution and generate a new one
class ConstraintsMethod(Enum):
    PTY = 1
    BAR = 2
    DIS = 3
    

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
def OOBMethodFromStr(value: str) -> OOBMethod:
    if (value.upper() == "PBC"):
        return OOBMethod.PBC
    elif (value.upper() == "RBC"):
        return OOBMethod.RBC
    elif (value.upper() == "BAB"):
        return OOBMethod.BAB
    return OOBMethod.DIS


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
def ConstraintsMethodFromStr(value: str) -> ConstraintsMethod:
    if (value.upper() == "PTY" or value.upper() == "PENALTY"):
        return ConstraintsMethod.PTY
    elif (value.upper() == "BAR" or value.upper() == "BARRIER"):
        return ConstraintsMethod.BAR
    return ConstraintsMethod.DIS


#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
@dataclass
class Config:
    nmax_iter: int
    nmax_iter_same_cost: int
    absolute_tol: float
    silent: bool
    oobMethod: OOBMethod
    constraintsMethod: ConstraintsMethod
    penalty_coeff: float
    penalty_scaling: float
    max_penalty_param: float

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, nmax_iter: int = 100, nmax_iter_same_cost: int = 0, 
                 same_cost_rel_tol: float = 1.e-8, silent: bool = True,
                 oobMethod: str = "RBC", constraintsMethod: str = "BAR",
                 penalty_coeff: float = 2., penalty_scaling: float = 10.,
                 max_penalty_param: float = 1.e10, seed: int = 0) -> None:
        # general
        self.nmax_iter = nmax_iter
        self.nmax_iter_same_cost = nmax_iter_same_cost
        self.same_cost_rel_tol = same_cost_rel_tol
        
        # verbose
        self.silent = silent
        
        # oob
        self.oobMethod = OOBMethodFromStr(oobMethod)
        
        # constraints
        self.constraintsMethod = ConstraintsMethodFromStr(constraintsMethod)
        self.penalty_coeff = penalty_coeff
        self.penalty_scaling = penalty_scaling
        self.max_penalty_param = max_penalty_param
        
        # seed
        self.seed = seed


