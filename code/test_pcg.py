# Imports
from preprocessing import preprocessing
from OF_cg import cg
from helper_functions import mycomputeColor
import numpy as np
import matplotlib.pyplot as plt
from multigrid import V_cycle
from time import perf_counter
from PCG import pcg
from utils import F, norm


def test_pcg():
    """
    Main function for calling procedues to numerically solve the optical flow problem.

    Returns:
    ---
    None
    """

    # Unpack data from preprocessing (spatial derivatives, temporal derivatives, etc.)
    Ix, Iy, It, rhsu, rhsv, h = preprocessing(2, 1, 1, 1, 1)
    print("Ix: ", Ix.shape)
    print("Iy: ", Iy.shape)
    print("rhsu: ", rhsu.shape)
    print("rhsv: ", rhsv.shape)

    n, m = Ix.shape
    lam = 5

    # Call OF_cg to numerically solve for u anv using the CG-method.
    x, y = np.zeros((n, m)), np.zeros((n, m))

    # Initial residual
    u_init, v_init = np.zeros((n, m)), np.zeros((n, m))

    fu, fv = F(u_init, v_init, Ix, Iy, lam, h)
    ru = rhsu - fu
    rv = rhsv - fv
    print("Initial Residual: ", norm(ru, rv))

    # start = perf_counter()
    # u_cg, v_cg = cg(x, y, Ix, Iy, lam, rhsu, rhsv, tol=1e-8)
    # end = perf_counter()
    # print("Time: ", end - start)

    start = perf_counter()
    u, v = pcg(x, y, Ix, Iy, lam, rhsu, rhsv, s1=10, s2=10, max_level=5, tol=1e-8)
    end = perf_counter()
    print("Time: ", end - start)

    # fu, fv = F(u_cg, v_cg, Ix, Iy, lam, h)
    # ru = rhsu - fu
    # rv = rhsv - fv
    # print("Residual CG: ", norm(ru, rv))

    fu, fv = F(u, v, Ix, Iy, lam, h)
    ru = rhsu - fu
    rv = rhsv - fv
    print("Residual PCG: ", norm(ru, rv))

    result = mycomputeColor(u, v)
    plt.imshow(result)
    plt.show()

    # result = mycomputeColor(u_cg, v_cg)
    # plt.imshow(result)
    # plt.show()


test_pcg()
