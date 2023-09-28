# Chebyquad
import chebyquad_problem as ch
import opt_problem
import opt_method
import scipy.optimize as so
import numpy as np
import matplotlib.pyplot as plt

chebyproblem = opt_problem.opt_problem(ch.chebyquad, ch.gradchebyquad)

solver2 = opt_method.BFGS(chebyproblem, np.linspace(0,1,11), 1e-5, inexact = True)
solver2.optimize()

#x=np.linspace(0,1,4)
#xmin= so.fmin_bfgs(ch.chebyquad,x,ch.gradchebyquad)  # should converge after 18 iterations  
#print(xmin)