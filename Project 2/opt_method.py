import numpy as np
import opt_problem

class general_opt_method:
    def __init__(self, problem, initial_guess, tol, max_it=100000):
        self.problem = problem
        self.function = problem.function
        self.gradient = problem.gradient
        self.history = [np.array(initial_guess)]
        self.H = np.identity(len(initial_guess))
        self.tol = tol
        self.max_it = max_it

    def direction(self):
        return np.matmul(-self.H,self.gradient(self.history[-1]).T)

    def update_guess(self):
        # Calculate new guess

        print(f"direction: {self.line_search()*self.direction()}")
        print(f"history: {self.history[-1]}")
        
        self.history.append(
            np.add(self.history[-1], self.line_search()*self.direction())
        )


        print(f"Current guess: {self.history[-1]}")
        print(f"Direction: {self.direction()}")

        # Update inverse Hessian guess (H)
        self.update_hessian()

    def line_search(self):
        return 1e-2

    def update_hessian(self):
        raise NotImplementedError("Subclasses must implement this method")

    def check_convergence(self):
        if (np.linalg.norm(self.history[-1] - self.history[-2]) < self.tol):
            return True
        
        return False

    def optimize(self):
        for _ in range(self.max_it):
            self.update_guess()
            
            print(f"Current guess{self.history[-1]}")
            if self.check_convergence():
                print("Converged")
                break

        print(f"Number of iterations: {len(self.history)}")
        print(f"Solution: {self.history[-1]}")

        return self.history[-1]


class classicNewt(general_opt_method):
    def __init__(self, problem, initial_guess, tol, max_it=100000):
        super().__init__(problem, initial_guess, tol, max_it)

    def update_hessian(self, epsilon=1):
        n = len(self.history[-1])
        G = np.zeros((n, n)) # Hessian
        
        for i in range(n):
            for j in range(n):
                #print(f"looping update hessian i: {i}, j: {j}")

                x = self.history[-1].copy()
                #print(f"X values before mod: {x}")
                
                # x1 = np.zeros(len(self.history[-1]),1)
                # x1[i] = 1/np.sqrt(2)
                # x1[j] = 1/np.sqrt(2)

                x[i] += epsilon
                x[j] += epsilon
                g1 = self.gradient(x)[i]
                
                # x2 = np.zeros(len(self.history[-1]),1)
                # x2[i] = 1/np.sqrt(2)
                # x2[j] = -1/np.sqrt(2)        

                x[j] -= 2*epsilon
                g2 = self.gradient(x)[i]

                # x3 = np.zeros(len(self.history[-1]),1)
                # x3[i] = -1/np.sqrt(2)
                # x3[j] = -1/np.sqrt(2)
                
                x[i] -= 2*epsilon
                g3 = self.gradient(x)[i]

                # x4 = np.zeros(len(self.history[-1]),1)
                # x4[i] = -1/np.sqrt(2)
                # x4[j] = 1/np.sqrt(2)
                
                x[j] += 2*epsilon
                g4 = self.gradient(x)[i]

                #print(f"X values after mod: {x}")
                
                #print(f"g1: {g1}, g2: {g2}, g3: {g3}, g4: {g4}")
                
                G[i, j] = (g1 - g2 - g3 + g4) / (4 * epsilon)
        
        #print(f"G before symmetrized: {G}")
        G = 0.5*(G + G.T) # symmetrized
        #print(f"Symmetrized G: {G}")

        # Invert G

        self.H = np.linalg.inv(G)
        #print(f"Updated H: {self.H}")

def f1(x):
    #return (x[0]-3)**2 + x[1]**2
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2


def gf1(x):
  #  grad = 2*x[0] + 2*x[1]
    #print(f"return grad: {grad}")
    #return np.array([2*x[0]-6, 2*x[1]])

    df_dx1 = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
    df_dx2 = 200*(x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])


prob1 = opt_problem.opt_problem(f1, gf1)

solver = classicNewt(prob1, [1.01,1], 1e-6)
solver.optimize()