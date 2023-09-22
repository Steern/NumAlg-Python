import opt_problem
import opt_method
import numpy as np
import matplotlib.pyplot as plt


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

# solver = opt_method.classicNewt(prob1, [-10,3], 1e-6, inexact = True)
# solver.optimize()

solver = opt_method.DFP(prob1, [3,-10], 1e-4, inexact = True)
solver.optimize()


# Create data for the contour plot
x = np.linspace(-2, 2, 1000)  # X-axis values
y = np.linspace(-2, 2, 1000)  # Y-axis values
X, Y = np.meshgrid(x, y)     # Create a grid of X and Y values
Z = 100*(Y - X**2)**2 + (1 - X)**2     # Z-values (example function, you can replace it with your own)


# Create the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=40, cmap='viridis')  # Adjust levels and cmap as needed
plt.clabel(contour, inline=1, fontsize=10)

# Plot the points on the contour plot
# Make vectors of x and y values from root guesses

x_coords = []
y_coords = []
for i in range(len(solver.history)):
    x_coords.append(solver.history[i][0])
    y_coords.append(solver.history[i][1])
    
plt.scatter(x_coords, y_coords, color='red', marker='x', label='Root guesses')

# Add labels and a color bar
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim()
plt.title('Contour Plot')
plt.colorbar(contour, label='Z-values')

# Show the plot
plt.show()
