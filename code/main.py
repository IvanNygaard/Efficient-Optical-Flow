# Imports
from preprocessing import preprocessing
from OF_cg import OF_cg
from helper_functions import mycomputeColor
import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    Main function for calling procedues to numerically solve the optical flow problem. 

    Returns:
    ---
    None
    """

    # Unpack data from preprocessing (spatial derivatives, temporal derivatives, etc.)
    Ix, Iy, It, rhsu, rhsv, h = preprocessing(2, 1, 1, 1, 1)

    n,m = Ix.shape
    N   = (n-2)*(m-2)
    

    # Call OF_cg to numerically solve for u anv using the CG-method.
    u, v = OF_cg(np.zeros(N), np.zeros(N), Ix, Iy, 0.05, rhsu, rhsv) 


    # Plotting
    result = mycomputeColor(u,v)
    plt.imshow(result)
    plt.show()

main()
