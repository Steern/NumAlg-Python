n = len()
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