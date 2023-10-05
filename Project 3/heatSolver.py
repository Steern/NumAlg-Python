from mpi4py import MPI
import numpy as np
from scipy.sparse import csr_matrix


def solve(T0, uNorm, uH, uWF, delta_x, relax_w, iterations):
    # Start threads
    """ Get a communicator:
    The most common communicator is the
    one that connects all available processes
    which is called COMM_WORLD.
    Clone the communicator to avoid interference
    with other libraries or applications
    """
    comm = MPI.Comm.Clone( MPI.COMM_WORLD )
        
    rank = comm.Get_rank()
    if rank == 0:
        # Room 2
        T0 = np.ones(2/delta_x, dtype='d')*T0
        its = 0
        while (its < iterations):
            # Edit matrix to adjust for bounderies 

            # Solve
            # Wait to receive Neumann conditions (dU values) for wall Gamma 1 from room 2
            y_len = 2/delta_x
            x_len = 1/delta_x
            A = matrix_A(delta_x,1/delta_x,2/delta_x)
            top_index = range(x_len)
            left_index = [i*x_len for i in range(y_len)]
            right_index = [i*x_len-1 for i in range(1,y_len+1)]
            bottom_index = range((y_len-1)*x_len, y_len*x_len)
            dirichlet_index = top_index + left_index + right_index + bottom_index
            A_mod = A
            for index in dirichlet_index:
                A_mod = A[:,index] = 0
            
            top_left = left_index[[i for i in range(int(y_len/2))]]
            bot_left = left_index[[i for i in range(int(y_len/2),y_len)]]
            top_right = right_index[[i for i in range(int(y_len/2))]]
            bot_right = right_index[[i for i in range(int(y_len/2),y_len)]]
            
            const_temp = np.array(y_len*x_len)
            const_temp[top_left + bot_right] = 15
            const_temp[bot_left] = data_1
            const_temp[top_right] = data_2 
            const_temp[bottom_index] = 5
            const_temp[top_index] = 40

            u_new = np.linalg.solve(A_mod,-np.matmul(A,const_temp)) 
            
            # Extract values from indices in u_new which corresponds to walls Gamma 1 and Gamma 1, to be
            # stored in U1 and U2 and sent away to other threads

            # send to room 1
            U1 = 0
            comm.Send([U1, MPI.DOUBLE], dest = 1, tag = 21)
            
            # send to room 3
            U2 = 0
            comm.Send([U2, MPI.DOUBLE], dest = 2, tag = 23)

            # wait to receive from room 1
            data1 = np.empty(1/delta_x, dtype='d')
            comm.Recv(data1, source=0, tag=12)
            
            # wait to receive from room 3
            data2 = np.empty(1/delta_x, dtype='d')
            comm.Recv(data2, source=0, tag=13)

            its = its + 1

        # Plot the data, either received from 1 and 3 or saved as attribute
        # plot(u) 

        # There is a prototype of that in plot.py! It can basically be moved here.

    if rank == 1:
        # Room 1
        while True:
            data = np.empty(1/delta_x, dtype='d')
            comm.Recv(data, source=0, tag=21 )
            # Edit matrix to adjust for bounderies 
            x_len = 1/delta_x
            y_len = 1/delta_x
            A = matrix_A(delta_x,x_len,y_len)
            top_index = range(x_len)
            left_index = [i*x_len for i in range(y_len)]
            right_index = [i*x_len-1 for i in range(1,y_len+1)]
            bottom_index = range((y_len-1)*x_len, y_len*x_len)
            dirichlet_index = top_index + left_index + bottom_index
            neumann_index = right_index
            A_mod = A
            for index in neumann_index:
                A_mod[index,:] = 

            # Solve

            # send to room 2
            dU2 = 0
            comm.Send([dU2, MPI.DOUBLE], dest = 0, tag = 12)
    
    if rank == 2:
        # Room 3
        while True:
            data = np.empty(1/delta_x, dtype='d')
            comm.Recv(data, source=0, tag = 23 )
            
            # Edit matrix to adjust for bounderies 
            x_len = 1/delta_x
            y_len = 1/delta_x
            A = matrix_A(delta_x,x_len,y_len)
            top_index = range(x_len)
            left_index = [i*x_len for i in range(y_len)]
            right_index = [i*x_len-1 for i in range(1,y_len+1)]
            bottom_index = range((y_len-1)*x_len, y_len*x_len)
            dirichlet_index = top_index + right_index + bottom_index
            neumann_index = left_index
            # Solve

            # send to room 2
            dU1 = 0
            comm.Send([dU1, MPI.DOUBLE], dest = 0, tag = 32)

def matrix_A(delta_x, size_x, size_y):
    N = size_x * size_y

    # Non-zero elements and their respective row and column indices
    data = []
    rows = []
    cols = []

    for i in range(Nx):
        for j in range(Ny):
            n = i + j * Nx
        
            # Set main diagonal
            rows.append(n)
            cols.append(n)
            data.append(-4)
        
            # West
            if i > 0:
                rows.append(n)
                cols.append(n-1)
                data.append(1)
            
            # East
            if i < Nx-1:
                rows.append(n)
                cols.append(n+1)
                data.append(1)
            
            # South
            if j > 0:
                rows.append(n)
                cols.append(n-Nx)
                data.append(1)
            
            # North
            if j < Ny-1:
                rows.append(n)
                cols.append(n+Nx)
                data.append(1)

    # Create the CSR matrix
    A = csr_matrix((data, (rows, cols)), shape=(N, N))

    return A/(dx**2)

def main():
    T0 = 40
    uNorm = uNorm
    uH = uH
    uWF = uWF
    delta_x = delta_x
    relax_w = relax_w
    iterations = iterations
    
    solve(T0, uNorm, uH, uWF, delta_x, relax_w, iterations)

if __name__ == "__main__":
    main()