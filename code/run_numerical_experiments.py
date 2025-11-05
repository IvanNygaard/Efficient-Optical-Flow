import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocessing
from OF_cg import OF_cg

def run_numerical_experiments(sigma, dx, dy, dt):
    """
    Function used to create convergence plots from numerical experiments, with some nice styling,
    styling retrieved from https://github.com/AndreyChurkin/BeautifulFigures/blob/main/Python/scripts/beautiful_figure_example.py
    """
    k_vals = [6, 7, 8, 9]
 
    iteration_vals = []
    res_ratio_vals = []

 
    for i in range(len(k_vals)):
        Ix, Iy, It, rhsu, rhsv, h = preprocessing(sigma, dx, dy, dt, k_vals[i]) 

        n, m = Ix.shape
        u, v, iterations, res_ratio = OF_cg(np.zeros(n*m), np.zeros(n*m), Ix, Iy, 4 ** (k_vals[i]-4), rhsu, rhsv)

        iteration_vals.append(iterations)
        res_ratio_vals.append(res_ratio)


    plt.rcParams.update({
    'font.family': 'Courier New',
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
    }) 

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlabel(f"Iteration")
    ax.set_ylabel("r_0/r_k")
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.loglog(iteration_vals[0], res_ratio_vals[0], label = "k = 6", color = "red", linewidth = 1.5, zorder = 3) 
    ax.loglog(iteration_vals[1], res_ratio_vals[1], label = "k = 7", color = "blue", linewidth = 1.5, zorder = 3)
    ax.loglog(iteration_vals[2], res_ratio_vals[2], label = "k = 8", color = "green", linewidth = 1.5, zorder = 3)
    ax.loglog(iteration_vals[3], res_ratio_vals[3], label = "k = 9", color = "cyan", linewidth = 1.5, zorder = 3) 

    ax.legend()
    plt.show()

