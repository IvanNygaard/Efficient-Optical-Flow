# Imports
from preprocessing import preprocessing
from OF_cg import OF_cg
from helper_functions import mycomputeColor
import numpy as np
import matplotlib.pyplot as plt
from multigrid import V_cycle


def main():
    """
    Main function for calling procedues to numerically solve the optical flow problem.

    Returns:
    ---
    None
    """

    # Unpack data from preprocessing (spatial derivatives, temporal derivatives, etc.)
    Ix, Iy, It, rhsu, rhsv, h = preprocessing(1, 1, 1, 1, 1)
    print("Ix: ", Ix.shape)
    print("Iy: ", Iy.shape)
    print("rhsu: ", rhsu.shape)
    print("rhsv: ", rhsv.shape)

    n, m = Ix.shape

    # Call OF_cg to numerically solve for u anv using the CG-method.
    # u, v = OF_cg(np.zeros((n, m)), np.zeros((n, m)), Ix, Iy, 1, rhsu, rhsv)
    u, v = V_cycle(
        np.zeros((n, m)), np.zeros((n, m)), Ix, Iy, 1, rhsu, rhsv, 2, 1, 1, 6
    )

    # Plotting
    result = mycomputeColor(u, v)
    plt.imshow(result)
    plt.show()


main()
