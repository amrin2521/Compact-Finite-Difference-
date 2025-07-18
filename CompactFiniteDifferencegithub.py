import numpy as np
import math

# 2D grid index to 1D index for matrix use
def node_index_xy_to_P(i: int, j: int, Nx: int):
    return ((j - 1) * Nx) + i - 1

#  1D index back to 2D 
def node_index_P_to_xy(l: int, Nx: int, Ny: int):
    i = ((l - 1) % Nx) + 1
    j = math.ceil(l / Nx)
    return i, j


"The Compact Finite Difference Formula : # (1/10) * u''[i-1] + u''[i] + (1/10) * u''[i+1] = (6/5) * (u[i+1] - 2*u[i] + u[i-1]) / h**2"
# one row of the A and B matrices using the compact finite difference method
def compact_finite_difference_row(var: str, i: int, j: int, Nx: int, Ny: int, hx, hy):
    N = Nx * Ny
    A_row = np.zeros(N)  # Matrix A comes from LHS of the compact scheme
    B_row = np.zeros(N)  # Matrix B from RHS of the compact scheme
    p = node_index_xy_to_P(i, j, Nx)  # 2D point to 1D index

    # step size and direction
    if var == "x":
        h = hx
        step = 1 #first step in x direction
        index = i
        Ndir = Nx
    elif var == "y":
        h = hy
        step = Nx #first step in y direction
        index = j
        Ndir = Ny
    else:
        raise ValueError("var must be 'x' or 'y'")

    #  values for A matrix from compact scheme's left-hand side
    A_row[p] = 1
    if index > 1:
        A_row[p - step] = 1 / 10
    if index < Ndir:
        A_row[p + step] = 1 / 10

    # values for B matrix from ompact scheme's right-hand side
    scale = 6 / 5
    B_row[p] = (-2 / h**2) * scale
    if index > 1:
        B_row[p - step] = (1 / h**2) * scale
    if index < Ndir:
        B_row[p + step] = (1 / h**2) * scale

    return A_row, B_row

# full Laplacian matrix using compact finite difference
def compact_laplacian_matrix(a, b, c, d, Nx, Ny):
    N = Nx * Ny
    A = np.zeros((N, N))  #  matrix A
    B = np.zeros((N, N))  # matrix B
    hx = (b - a) / (Nx + 1)  # Grid spacing in x
    hy = (d - c) / (Ny + 1)  # Grid spacing in y

    # Loopingg through every grid point 
    for j in range(1, Ny + 1):
        for i in range(1, Nx + 1):
            p = node_index_xy_to_P(i, j, Nx)

            # compact FD rows for x-direction
            A_row_x, B_row_x = compact_finite_difference_row("x", i, j, Nx, Ny, hx, hy)
            A[p, :] += A_row_x
            B[p, :] += B_row_x

            # compact FD rows for y-direction
            A_row_y, B_row_y = compact_finite_difference_row("y", i, j, Nx, Ny, hx, hy)
            A[p, :] += A_row_y
            B[p, :] += B_row_y

    #Now we have to Solve L= A⁻¹ B to get the final Laplacian operator matrix
    L = np.linalg.solve(A, B)
    return L

"""
Test for Example 3.1 from Zhang's paper (2018)
"""

def test_example_3_1(Nx, Ny, gamma=0.1, Dx=0.2, Dy=1.0):
    a, b, c, d = 0, 2 * np.pi, 0, 2 * np.pi
    hx = (b - a) / (Nx + 1)
    hy = (d - c) / (Ny + 1)
    N = Nx * Ny

    #spatial matrix: (Dx + Dy) * Laplacian + gamma * Identity
    L = compact_laplacian_matrix(a, b, c, d, Nx, Ny)
    A = (Dx + Dy) * L + gamma * np.eye(N)

    u = np.zeros(N)        #  u(x,y) = sin(x)sin(y)
    f_exact = np.zeros(N)  

    for k in range(N):
        i, j = node_index_P_to_xy(k + 1, Nx, Ny)
        x = a + i * hx
        y = c + j * hy
        u[k] = np.sin(x) * np.sin(y)
        f_exact[k] = -1.1 * u[k]  

   
    f_numeric = A @ u     #I'm applying the numerical operator (matrix A) to our known function u
                          # This gives us the predicted result based on our finite difference method
                       

    #  numerical and exact result
    error = np.linalg.norm(f_numeric - f_exact, ord=np.inf)
    return error

# test 
if __name__ == "__main__":
    with open("example3_1_output.txt", "w") as f:
        
        header = f"{'h':>10} {'Error':>15} {'Order':>10}"
        print(header)
        f.write(header + "\n")
        
        prev_error = None
        for N in [16, 32, 64, 128]:
            h = 2 * np.pi / (N + 1)
            error = test_example_3_1(N, N)
            if prev_error is None:
                line = f"{h:10.4f} {error:15.3e} {'-':>10}"
            else:
                order = np.log2(prev_error / error)
                line = f"{h:10.4f} {error:15.3e} {order:10.2f}"
            print(line)
            f.write(line + "\n")
            prev_error = error



    