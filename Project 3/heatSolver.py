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

        #all temp from other rooms
        data_1_all = np.ones(int(1/delta_x**2))*T0 
        data_2_all = np.ones(int(1/delta_x)**2)*T0


        its = 0
        
        y_len = int(2/delta_x)
        x_len = int(1/delta_x)

        u1 = np.ones(y_len*x_len, dtype='d')*uNorm
        
        while (its < iterations):
            # Edit matrix to adjust for bounderies 

            
            # Wait to receive Neumann conditions (dU values) for wall Gamma 1 from room 2

            A = matrix_A(int(1/delta_x) ,int(2/delta_x))
            top_index, left_index, right_index, bottom_index = get_boundary_indices(x_len, y_len)
            dirichlet_index = top_index + left_index + right_index + bottom_index
            
            A = A/(delta_x**2)

            half_y = int(y_len/2)
            top_left = left_index[:half_y]
            bot_left = edges(x_len,x_len)[3] + int(x_len*half_y)
            top_right = edges(x_len,x_len)[1]
            bot_right = right_index[half_y-2:]

            const_temp = np.zeros(y_len*x_len)
            const_temp[top_left] = uNorm
            const_temp[bot_right] = uNorm
            const_temp[bot_left] = data_1
            const_temp[top_right] = data_2 
            const_temp[bottom_index] = uWF
            const_temp[top_index] = uH

            # Solve
            A = A.tocsr()
            u_new = spsolve(A,const_temp/delta_x**2) # solves the system
           
            
            dU1 = -(u_new[bot_left] - u_new[bot_left + 1])/delta_x
            dU2 = -(u_new[top_right] - u_new[top_right - 1])/delta_x
          
            u2 = u_new

            # send to room 1
            comm.Send([dU1, MPI.DOUBLE], dest = 1, tag = 21)
            
            # send to room 3
            comm.Send([dU2, MPI.DOUBLE], dest = 2, tag = 23)

            u_new = relax_w*u_new + (1-relax_w)*u1

            u1 = u_new
            # wait to receive from room 1
            data_1_new = np.empty(x_len**2, dtype='d')
            comm.Recv(data_1_new, source=1, tag=12)
            
            # wait to receive from room 3
            data_2_new = np.empty(x_len**2, dtype='d')
            comm.Recv(data_2_new, source=2, tag=32)

            its = its + 1
            
            data_1_all = relax_w*data_1_new + (1-relax_w)*data_1_all
            data_2_all = relax_w*data_2_new + (1-relax_w)*data_2_all
            data_1 = data_1_all[(edges(x_len,x_len)[1])]
            data_2 = data_2_all[edges(x_len,x_len)[3]]

        plot(data_1_all, u_new, data_2_all, x_len, y_len)

    if rank == 1:
        # Room 1
        u1 = np.ones(int(1/delta_x)**2, dtype='d')*T0
        while True:
           
            data = np.empty(int(1/delta_x)-2, dtype='d')
            comm.Recv(data, source=0, tag=21)
            # Edit matrix to adjust for bounderies 
            x_len = int(1/delta_x)
            y_len = int(1/delta_x)

            A = matrix_A(int(x_len),int(x_len))

            top_index, left_index, right_index, bottom_index = get_boundary_indices(x_len, y_len)

            dirichlet_index = top_index + left_index + bottom_index
            neumann_index = right_index
           
            A = set_Neumann(neumann_index, A, x_len, True)

            A = A/(delta_x**2)
            
            const_temp = np.zeros(x_len*x_len)
            const_temp[left_index] = uH
            const_temp[top_index  + bottom_index] = uNorm
           
            neumann_vector = np.zeros(x_len*x_len)
            neumann_vector[neumann_index] = data/delta_x

            # Solve
            
            A = A.tocsr()
            u_new = spsolve(A,(const_temp)/delta_x**2 - neumann_vector)
            # send to room 2
            dU2 = u_new
            comm.Send([dU2, MPI.DOUBLE], dest = 0, tag = 12)
            
            u_new = relax_w*u_new + (1-relax_w)*u1

            u1 = u_new
    
    if rank == 2:
        # Room 3
        u1 = np.ones(int(1/delta_x)**2, dtype='d')*T0
        x_len = int(1/delta_x)
        y_len = int(1/delta_x)
        while True:
            
            data = np.empty(x_len-2, dtype='d')
            comm.Recv(data, source=0, tag = 23 )
            
            # Edit matrix to adjust for bounderies 
            A = matrix_A(x_len,x_len)

            top_index, left_index, right_index, bottom_index = get_boundary_indices(x_len, y_len)

            dirichlet_index = top_index + right_index + bottom_index
            neumann_index = left_index

            A = set_Neumann(neumann_index, A, x_len, False)
           
            A = A/(delta_x**2)
           

            const_temp = np.zeros(x_len*x_len)
            const_temp[right_index] = uH
            const_temp[top_index  + bottom_index] = uNorm
           
            neumann_vector = np.zeros(x_len*x_len)
            neumann_vector[neumann_index] = data/delta_x
            # Solve
            
            A = A.tocsr()
            u_new = spsolve(A,const_temp/delta_x**2 - neumann_vector)


            # send to room 2
            dU1 = u_new
            comm.Send([dU1, MPI.DOUBLE], dest = 0, tag = 32)
            u_new = relax_w*u_new + (1-relax_w)*u1

            u1 = u_new
    
    if rank == 3:
        pass

def get_boundary_indices(x_len, y_len):
    top_index = list(range(x_len))
    left_index = list(edges(x_len, y_len)[3])
    right_index = list(edges(x_len, y_len)[1])
    bottom_index = list(range((y_len-1) * x_len, y_len * x_len))

    return top_index, left_index, right_index, bottom_index
        
def set_Dirichlet(dirichlet_index, A_mod):
    for index in dirichlet_index:
        A_mod[:,index] = 0
        A_mod[index,index]=-1

    return A_mod

def set_Neumann(neumann_index, A_mod, x_len, right):
    # we use right = True in room 1
    for index in neumann_index:
        A_mod[index,:]= 0
        A_mod[index,index] = -3
        A_mod[index,index+x_len] = 1
        A_mod[index, index-x_len] = 1
        if right:
            A_mod[index, index-1] = 1
        else:
            A_mod[index, index+1] = 1
    
    return A_mod

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
        
    A = csr_matrix((values, (rowind, colind)), shape=(N_x*N_y, N_x*N_y))
    return A.tolil()

def main():
    T0 = 15
    uNorm = 15
    uH = 40
    uWF = 5
    delta_x = 1/50
    relax_w = 0.8
    iterations = 10
    
    solve(T0, uNorm, uH, uWF, delta_x, relax_w, iterations)

if __name__ == "__main__":
    main()