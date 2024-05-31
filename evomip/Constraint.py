from dataclasses import dataclass

@dataclass
class Constraint:
    func: 'function'
    inequality: str

    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def __init__(self, func: 'function', inequality: str) -> None:
        self.func = func
        self.inequality = inequality