from mpi4py import MPI
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from plot import plot

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
        data_1 = np.ones(int(1/delta_x)-2, dtype='d')*T0
        data_2 = np.ones(int(1/delta_x)-2, dtype='d')*T0
        its = 0
        
        y_len = int(2/delta_x)
        x_len = int(1/delta_x)

        u1 = np.ones(y_len*x_len, dtype='d')*uNorm
        
        while (its < iterations):
            # Edit matrix to adjust for bounderies 

            # Solve
            # Wait to receive Neumann conditions (dU values) for wall Gamma 1 from room 2


            A = matrix_A(int(1/delta_x) ,int(2/delta_x))
            top_index = list(range(x_len))
            left_index = list(edges(x_len,y_len)[3])
            right_index = list(edges(x_len,y_len)[1])
            bottom_index = list(range((y_len-1)*x_len, y_len*x_len))
            dirichlet_index = top_index + left_index + right_index + bottom_index
            A_mod = A.copy()
            for index in dirichlet_index:
                A_mod[:,index] = 0
                A_mod[index,index] = -1
            A_mod = csr_matrix(A_mod)

            half_y = int(y_len/2)
            top_left = left_index[:half_y]
            #bot_left = left_index[half_y-1:]
            bot_left = edges(x_len,x_len)[3] + int(x_len*half_y)
          #  top_right = right_index[:half_y-1]
            top_right = edges(x_len,x_len)[1]
            bot_right = right_index[half_y-2:]
            print(right_index)
            print(top_right)

            const_temp = np.zeros(y_len*x_len)
            const_temp[top_left] = uNorm
            const_temp[bot_right] = uNorm
            const_temp[bot_left] = data_1
            const_temp[top_right] = data_2 
            const_temp[bottom_index] = uWF
            const_temp[top_index] = uH

            u_new = spsolve(A_mod,-A@const_temp) # solves the system
            print(u_new)
            
            
            dU1 = -(u_new[bot_left] - u_new[bot_left + 1])/delta_x
            dU2 = (u_new[top_right] - u_new[top_right - 1])/delta_x

            u2 = u_new
            # u1 = np.ones(36)*15
            # u3 = np.ones(36)*15

            # plot_u2 = u2.reshape(   (y_len, x_len)  )
            # plot_u1 = u1.reshape(   (x_len, x_len)  ) 
            # plot_u3 = u3.reshape(   (x_len, x_len)  )
            # plot(plot_u1, plot_u2, plot_u3)
            
            # Extract values from indices in u_new which corresponds to walls Gamma 1 and Gamma 1, to be
            # stored in U1 and U2 and sent away to other threads

            # send to room 1
            comm.Send([dU1, MPI.DOUBLE], dest = 1, tag = 21)
            
            # send to room 3
            comm.Send([dU2, MPI.DOUBLE], dest = 2, tag = 23)

            u_new = relax_w*u_new + (1-relax_w)*u1

            u1 = u_new
            # wait to receive from room 1
            data_1 = np.empty(x_len**2, dtype='d')
            comm.Recv(data_1, source=1, tag=12)
            
            # wait to receive from room 3
            data_2 = np.empty(x_len**2, dtype='d')
            comm.Recv(data_2, source=2, tag=32)

            its = its + 1
            if its < iterations:
                #print(edges(x_len,x_len)[3])
                data_1 = data_1[(edges(x_len,x_len)[1])]
                data_2 = data_2[edges(x_len,x_len)[3]]

        # reshape data and then plot it
        u2 = u_new
        u1 = data_1
        u3 = data_2

        plot_u2 = u2.reshape(   (y_len, x_len)  )
        plot_u1 = u1.reshape(   (x_len, x_len)  ) 
        plot_u3 = u3.reshape(   (x_len, x_len)  )
        plot(plot_u1, plot_u2, plot_u3)

       

    if rank == 1:
        # Room 1
        u1 = np.ones(int(1/delta_x**2), dtype='d')
        while True:
           
            data = np.empty(int(1/delta_x)-2, dtype='d')
            comm.Recv(data, source=0, tag=21)
            # Edit matrix to adjust for bounderies 
            x_len = int(1/delta_x)
            y_len = int(1/delta_x)

            A = matrix_A(int(x_len),int(x_len))

            top_index = list(range(x_len))
            left_index = list(edges(x_len,x_len)[3])
            right_index = list(edges(x_len,x_len)[1])
            bottom_index = list(range((y_len-1)*x_len, y_len*x_len))
            dirichlet_index = top_index + left_index + bottom_index
            neumann_index = right_index
           
            A_mod = A.copy()

            for index in dirichlet_index:
                A_mod[:,index] = 0
                A_mod[index,index]=-1

            for index in neumann_index:
                A_mod[index,:]= 0
                A_mod[index,index] = -3
                A_mod[index,index+x_len] = 1
                A_mod[index, index-x_len] = 1
                A_mod[index, index-1] = 1

            const_temp = np.zeros(x_len*x_len)
            const_temp[left_index] = uH
            const_temp[top_index  + bottom_index] = uNorm
           
            neumann_vector = np.zeros(x_len*x_len)
            neumann_vector[neumann_index] = data

            # Solve
            u_new = spsolve(A_mod,-A@(const_temp) - neumann_vector)
            # send to room 2
            dU2 = u_new#[right_index]
            comm.Send([dU2, MPI.DOUBLE], dest = 0, tag = 12)
            
            u_new = relax_w*u_new + (1-relax_w)*u1

            u1 = u_new
    
    if rank == 2:
        # Room 3
        u1 = np.ones(int(1/delta_x**2), dtype='d')
        x_len = int(1/delta_x)
        y_len = int(1/delta_x)
        while True:
            print("hej")
            data = np.empty(x_len-2, dtype='d')
            comm.Recv(data, source=0, tag = 23 )
            
            # Edit matrix to adjust for bounderies 
            A = matrix_A(x_len,x_len)
            top_index = list(range(x_len))
            left_index = list(edges(x_len,x_len)[3])
            right_index = list(edges(x_len,x_len)[1])
            bottom_index = list(range((x_len-1)*x_len, x_len*x_len))
            dirichlet_index = top_index + right_index + bottom_index
            neumann_index = left_index

            A_mod = A.copy()

            for index in dirichlet_index:
                A_mod[:,index] = 0
                A_mod[index,index] = -1

            for index in neumann_index:
                A_mod[index,index] = -3
                A_mod[index,index+x_len] = 1
                A_mod[index, index-x_len] = 1
                A_mod[index, index+1] = 1

            const_temp = np.zeros(x_len*x_len)
            const_temp[right_index] = uH
            const_temp[top_index  + bottom_index] = uNorm
           
            neumann_vector = np.zeros(x_len*x_len)
            neumann_vector[neumann_index] = data
            # Solve
            u_new = spsolve(A_mod,-A@const_temp - neumann_vector)
            print(u_new)
            # send to room 2
            dU1 = u_new#[left_index]
            comm.Send([dU1, MPI.DOUBLE], dest = 0, tag = 32)
            u_new = relax_w*u_new + (1-relax_w)*u1

            u1 = u_new

def edges(N_x,N_y):
    edge1=np.array(range(N_x))
    edge2=np.array(range((N_x-1)+N_x,N_x*N_y-N_x,N_x))
    edge3=np.array(range(N_x*N_y-N_x,N_x*N_y))
    edge4=np.array(range(N_x,N_x*(N_y-1),N_x))
    
    return edge1,edge2,edge3,edge4

def inside(N_x,N_y):
    return np.setdiff1d(np.array(range(N_x*N_y-1)),np.concatenate(edges(N_x,N_y)))
    

def matrix_A(N_x, N_y):

    inner_index = inside(N_x,N_y)
    rowind = []
    colind = []
    values = []
    
    for k in inner_index:
        j=np.floor(k/N_x)
        i=np.remainder(k,N_x)
        rowind=np.append(rowind,[k,k,k,k,k])
        colind=np.append(colind,[j*N_x+i,(j+1)*N_x+i,(j-1)*N_x+i,j*N_x+i+1,j*N_x+i-1])
        values=np.append(values,[-4,1,1,1,1])
    for k in np.concatenate(edges(N_x,N_y)):
        rowind = np.append(rowind,k)
        colind = np.append(colind,k)
        values = np.append(values,1)
        
    return csr_matrix((values, (rowind, colind)), shape=(N_x*N_y, N_x*N_y))


def oldmatrix_A(size_x, size_y):
    
    N = size_x * size_y  # total number of unknowns
    A = np.zeros((N, N))
    
    # Fill the matrix A based on the finite difference discretization
    # This would also depend on the boundary conditions, which are not handled here
    
    for j in range(size_y):
        for i in range(size_x):
            idx = j * size_x + i  # map 2D indices to 1D

            A[idx, idx] = -4
            
            if i > 0:
                A[idx, idx-1] = 1
            if i < size_x-1:
                A[idx, idx+1] = 1
            if j > 0:
                A[idx, idx-size_x] = 1
            if j < size_y-1:
                A[idx, idx+size_x] = 1
    
    return csr_matrix(A*size_x**2)    


def main():
    T0 = 15
    uNorm = 15
    uH = 40
    uWF = 5
    delta_x = 1/6
    relax_w = 0.8
    iterations = 2
    
    solve(T0, uNorm, uH, uWF, delta_x, relax_w, iterations)

if __name__ == "__main__":
    main()