# Imports
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np


# CG-function.
def OF_cg(u0 : np.ndarray, 
          v0 : np.ndarray,
          Ix : np.ndarray,
          Iy : np.ndarray,
          reg : float,
          rhsu : np.ndarray,
          rhsv : np.ndarray,
          tol = 1.e-8,
          maxit = 2000
          ) -> tuple[np.ndarray, np.ndarray]:
    """
    The CG method for the optical flow problem.

    Args:
    ---
    u0 - initial guess for u
    v0 - initial guess for v
    Ix - x-derivative of the first frame
    Iy - y-derivative of the first frame
    reg - regularisation parameter lambda
    rhsu - right-hand side in the equation for u
    rhsv - right-hand side in the equation for v
    tol - relative residual tolerance
    maxit - maximum number of iterations
    output:
    u - numerical solution for u
    v - numerical solution for v

    Returns:
    ---
     tuple[np.ndarray, np.ndarray]
        Numerical solution for u, v
    """

    # Dimensions, step-size, constants for diagonal (k1) and off-diagonal (k2) elements and cross derivatives
    n,m = Ix.shape
    n = n - 2               
    m = m - 2
    h = 1
    k1  = (4 * reg)/(h*h)  
    k2  = -reg/(h*h)        


    # Note: Implicitly imposing Dirichlet B.C. by only acting on interior nodes (n-2, m-2) and settung u0 = v0 = 0
    Ix = Ix[1:-1, 1:-1]
    Iy = Iy[1:-1, 1:-1]
    Ixy = Ix * Iy
    rhsu = rhsu[1:-1, 1:-1]
    rhsv = rhsv[1:-1, 1:-1]


    # Initialize x and b with row-wise numbering
    u = u0
    v = v0
    x = np.hstack((u.ravel(order='C'), v.ravel(order='C')))
    b = np.hstack((rhsu.ravel(order='C'), rhsv.ravel(order='C')))


    # 1D Laplacian
    Lx = sp.diags([k2, k1, k2], [-1, 0, 1], shape=(m, m))
    Ly = sp.diags([k2, 0, k2], [-1, 0, 1], shape=(n, n))


    # 2D Laplacian from Kroneckersum, https://stackoverflow.com/questions/34895970/buildin-a-sparse-2d-laplacian-matrix-using-scipy-modules
    ex = sp.eye(n)
    #plt.spy(sp.kron(ex,Lx))
    #plt.show()
    ey = sp.eye(m)
    #plt.spy(sp.kron(Ly,ey))
    #plt.show()
    L  = sp.kron(ex, Lx) + sp.kron(Ly, ey)

    #print(L.shape)
    #plt.spy(L)
    #plt.show()


    A_11 = sp.diags(Ix.ravel()**2) + L
    A_22 = sp.diags(Iy.ravel()**2) + L
    A_12 = sp.diags(Ixy.ravel())
    A_21 = A_12.copy()


    A = sp.block_array([[A_11, A_12],
                        [A_21, A_22]])


    #plt.spy(A_11, markersize=0.8, color = "black")
    #plt.show()
    #plt.spy(A, markersize=0.8, color = "black")
    #plt.show()


    # CG-method (p. 190 Saad)
    iter  = 0
    r0    = b - A@x
    r_old = r0.copy()
    p     = r_old.copy()


    while True:
        alpha = (r_old.T @ r_old) / ((A@p).T @ p)
        x += alpha * p
        r_new = r_old - alpha * (A @ p)
        beta = (r_new.T @ r_new) / (r_old.T @ r_old)
        p = r_new + beta * p
        r_old = r_new
        iter += 1

        if (np.linalg.norm(r_old) / np.linalg.norm(r0) < tol) or (iter >= maxit):
            break

    

    #x, info = sp.linalg.cg(A, b)


    # Unpack solution
    u = x[:(n*m)].reshape((n,m))
    v = x[(n*m):].reshape((n,m))
    #print(np.max(Ix**2), 4*reg/(h*h))


    return u, v
