# removed functions / old code
# just pasted here if we want to go back to something

def old_matrix_A(size_x, size_y):
    
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
    
    return csr_matrix(A/(delta_x ** 2))


def old_sparse_matrix_A():    
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