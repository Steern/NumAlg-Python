import numpy as np
import opt_problem

class general_opt_method:
    def __init__(self, problem, initial_guess, tol, max_it=1000, inexact = False):
        self.problem = problem
        self.function = problem.function
        self.gradient = problem.gradient
        self.history = [np.array(initial_guess)]
        self.H = np.identity(len(initial_guess))
        self.G = np.identity(len(initial_guess))
        self.tol = tol
        self.max_it = max_it
        self.inexact = inexact

    def direction(self):
        return np.matmul(-self.H,self.gradient(self.history[-1]).T)

    def update_guess(self):
        # Calculate new guess
        currentDirection = self.direction()
      #  print(f"direction: {self.line_search(currentDirection)*currentDirection}")
     #   print(f"history: {self.history[-1]}")
        
        self.history.append(
            np.add(self.history[-1], self.line_search(currentDirection)*currentDirection)
        )

      #  print(f"Current guess: {self.history[-1]}")
      #  print(f"Direction: {currentDirection}")

        # Update inverse Hessian guess (H)
        self.update_hessian()

    def line_search(self, currentDirection):
        if (self.inexact == True):
            return self.wolfe(self.function, self.gradient)
        alpha = 0.1
        tol = 1e-2

        def phi(alpha):
            return self.function(self.history[-1]+alpha*currentDirection)
        def phidot(alpha):
            return np.matmul(self.gradient(self.history[-1]+alpha*currentDirection).T,currentDirection)
        def phi_bis(alpha):
            epsilon = 1e-2
            return (phidot(alpha + epsilon) - phidot(alpha - epsilon))/(2*epsilon)
       
        max_it = 100
        for i in range(max_it):
            # Update alpha using Newton method
            alpha -= phidot(alpha) / phi_bis(alpha)
            # Check for convergence
            if (phidot(alpha) < tol):
                break
        return alpha    
        
    def update_hessian(self):
        raise NotImplementedError("Subclasses must implement this method")

    def check_convergence(self):
        if (np.linalg.norm(self.history[-1] - self.history[-2]) < self.tol):
            return True
        
        return False

    def optimize(self):
        for _ in range(self.max_it):
            self.update_guess()
            
        #    print(f"Current guess{self.history[-1]}")
            if self.check_convergence():
                print("Converged")
                break
            print(f"Iter NUmber: {_}")
        print(f"Function value = {self.function(self.history[-1])} at point {self.history[-1]}")
        print(f"Number of iterations: {len(self.history)}")
        # print(f"Solution: {self.history[-1]}")

        return self.history[-1]
    
    def wolfe(self, f, grad_f):

        x = self.history[-1]
        p = self.direction()

        phi = lambda alpha: f(x + alpha*p)
        phidot = lambda alpha: grad_f(x + alpha*p).dot(p)

        sigma = 10e-2
        rhu = 0.9
        alphamin = 2
        
        def condition1(phi, sigma, alpha, phidot):
            return phi(alpha) <= phi(0) + sigma*alpha*phidot(0)

        def condition2(phi, rhu, alpha, phidot):
            return phidot(alpha) >= rhu*phidot(0)

        while not condition1(phi, sigma, alphamin, phidot):
            alphamin = alphamin / 2

        alphaplus = alphamin

        while condition1(phi, sigma, alphaplus, phidot):
            alphaplus = alphaplus * 2

        while not condition2(phi, rhu, alphamin, phidot):
            alphanoll = (alphaplus + alphamin) / 2
            if condition1(phi, sigma, alphanoll, phidot):
                alphamin = alphanoll
            else:
                alphaplus = alphanoll
            
       # print(f"alpha: {alphamin}")
        return alphamin

class classicNewt(general_opt_method,):
    def __init__(self, problem, initial_guess, tol, max_it=1000, inexact = False):
        super().__init__(problem, initial_guess, tol, max_it, inexact)

    def update_hessian(self, epsilon=1):
        n = len(self.history[-1])
        G = np.zeros((n, n)) # Hessian
        
        for i in range(n):
            for j in range(n):
                #print(f"looping update hessian i: {i}, j: {j}")

                x = self.history[-1].copy()
                #print(f"X values before mod: {x}")
                
                x[i] += epsilon
                x[j] += epsilon
                g1 = self.gradient(x)[i]
                
                x[j] -= 2*epsilon
                g2 = self.gradient(x)[i]
                
                x[i] -= 2*epsilon
                g3 = self.gradient(x)[i]

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

    def line_search(self, currentDirection):
        if self.inexact:
            return self.wolfe(self.function, self.gradient)
        
        alpha = 1
        tol = 1e-2
        def phi(alpha):
            return self.function(self.history[-1]+alpha*currentDirection)
        def phidot(alpha):
            return np.matmul(self.gradient(self.history[-1]+alpha*currentDirection).T,currentDirection)
        def phi_bis(alpha):
            epsilon = 1.4e-8
            return (phidot(alpha + epsilon) - phidot(alpha - epsilon))/(2*epsilon)
       
        max_it = 10
        for i in range(max_it):
            # Update alpha using Newton method
            alpha -= phidot(alpha) / phi_bis(alpha)
            # Check for convergence
            if (phidot(alpha) < tol):
                break
        return alpha    


class goodBroyden(general_opt_method):
    def __init__(self, problem, initial_guess, tol, max_it=100000, inexact = False):
        super().__init__(problem, initial_guess, tol, max_it, inexact)

    def update_hessian(self, epsilon=1):
        delta = np.add(self.history[-1].T, -self.history[-2].T)
        gamma = np.add(self.gradient(self.history[-1]).T, -self.gradient(self.history[-2]).T)
      #  print(f"delta = {delta}, gamma = {gamma}")
        a = np.inner(np.matmul(delta.T, self.H), gamma)
        b1 = np.add(delta, -1*np.matmul(self.H, gamma))
        b = (1/a)*np.outer(b1, delta.T)
      #  print(f"Dim: Delta = {delta.shape}, Gamma = {gamma.shape}, dim a = {a.shape}, dim b = {b.shape} a = {a}, b = {b}, H = {self.H.shape}")
      #  print(f"Dim b1 = {b1.shape}")
        self.H = np.add(self.H,np.matmul(b, self.H)).T
      #  print(f"H = {self.H}")
        
class badBroyden(general_opt_method):
    def __init__(self, problem, initial_guess, tol, max_it=500, inexact = False):
        super().__init__(problem, initial_guess, tol, max_it, inexact)

    def update_hessian(self, epsilon=1):
        delta = np.add(self.history[-1].T, -self.history[-2].T)
        gamma = np.add(self.gradient(self.history[-1]).T, -self.gradient(self.history[-2]).T)
      #  print(f"delta = {delta}, gamma = {gamma}")
        a = np.inner(gamma.T, gamma)
        b1 = np.add(delta, -1*np.matmul(self.H, gamma))
        b = (1/a)*np.outer(b1, gamma.T)
      #  print(f"Dim: Delta = {delta.shape}, Gamma = {gamma.shape}, dim a = {a.shape}, dim b = {b.shape} a = {a}, b = {b}, H = {self.H.shape}")
      #  print(f"Dim b1 = {b1.shape}")
        self.H = np.add(self.H, b).T
     #   print(f"H = {self.H}")

class symmetricBroyden(general_opt_method):
    def __init__(self, problem, initial_guess, tol, max_it=100000, inexact = False):
        super().__init__(problem, initial_guess, tol, max_it, inexact)

    def update_hessian(self, epsilon=1):
        delta = np.add(self.history[-1].T, -self.history[-2].T)
        gamma = np.add(self.gradient(self.history[-1]).T, -self.gradient(self.history[-2]).T)
        u = np.add(delta, -np.matmul(self.H,gamma))
        a = 1/(np.inner(u.T, gamma))
        self.H = np.add(self.H, a*np.outer(u, u.T)).T

class DFP(general_opt_method):
    def __init__(self, problem, initial_guess, tol, max_it=100000, inexact = False):
        super().__init__(problem, initial_guess, tol, max_it, inexact)

    def update_hessian(self, epsilon=1):
        delta = np.add(self.history[-1].T, -self.history[-2].T)
        gamma = np.add(self.gradient(self.history[-1]).T, -self.gradient(self.history[-2]).T)
        a = (1/(np.inner(delta.T, gamma)))*np.outer(delta, delta.T)
        b1 = np.matmul(self.H,gamma)
        b2 = np.matmul(gamma.T, self.H)
        b4 = np.outer(b1, b2)
        b3 = np.inner(gamma.T, np.matmul(self.H, gamma))
        self.H = np.add(self.H, np.add(a, -b4/b3)).T

class BFGS(general_opt_method):
    def __init__(self, problem, initial_guess, tol, max_it=50, inexact = False):
        super().__init__(problem, initial_guess, tol, max_it, inexact)

    def update_hessian(self, epsilon=1):
        delta = np.add(self.history[-1].T, -self.history[-2].T)
        gamma = np.add(self.gradient(self.history[-1]).T, -self.gradient(self.history[-2]).T)
        a1 = np.inner(np.matmul(gamma.T, self.H), gamma)
        a2 = (1 + a1/(np.inner(delta.T, gamma)))
        a3 = np.outer(delta, delta.T) / (np.inner(delta.T, gamma))
        a = a2*a3
        b1 = np.outer(delta, np.matmul(gamma.T, self.H))
        b2 = np.matmul(self.H, np.outer(gamma, delta.T))
        b3 = np.add(b1, b2)
        b = b3 / np.inner(delta.T, gamma)
        self.H = np.add(self.H, np.add(a, -b)).T

        c1 = np.outer(gamma, gamma.T)/np.inner(gamma.T, delta)
        d1 = np.outer(np.matmul(self.G, delta), np.matmul(delta.T, self.G))
        d2 = np.inner(delta.T, np.matmul(self.G, delta))
        self.G = self.G + c1 - d1/d2

        HReal = np.linalg.inv(self.G)

        print(f"Norm of Approx and 'Real' H: {np.linalg.norm(self.H - HReal)}")