import numpy as np

class opt_problem:
    def __init__(self, function, gradient = 1):
        self.function = function
        if gradient != 1:
            self.gradient = gradient
        else:
            self.gradient = self.calc_grad(function)
        
    def calc_grad(function):
        return np.gradient(function)