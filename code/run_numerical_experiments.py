import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocessing
from OF_cg import OF_cg, cg
from time import perf_counter
from multigrid import V_cycle


def run_numerical_experiments(sigma, dx, dy, dt):
    """
    Function used to create convergence plots from numerical experiments, with some nice styling,
    styling retrieved from https://github.com/AndreyChurkin/BeautifulFigures/blob/main/Python/scripts/beautiful_figure_example.py
    """

    # k_vals for image size
    k_vals = [6, 7, 8, 9]
 
    iteration_vals = []
    res_ratio_vals = []


    # testing of sparse implementation of cg
    for i in range(len(k_vals)):
        Ix, Iy, It, rhsu, rhsv, h = preprocessing(sigma, dx, dy, dt, k_vals[i]) 

        n, m = Ix.shape
        x, y = np.zeros((n, m)), np.zeros((n, m))

        start = perf_counter()
        u, v, res_ratio = OF_cg(x, y, Ix, Iy, 4 ** (k_vals[i]-4), rhsu, rhsv)
        stop = perf_counter()
        print(f'Time for sparse OF_CG (no multigrid) = {stop - start}, k = {k_vals[i]}')

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
    ax.set_ylabel(r"$log_{10} \left( \frac{r_k}{r_0} \right)$")

    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.plot(np.arange(0, len(res_ratio_vals[0])), res_ratio_vals[0], label = "k = 6", color = "red", linewidth = 1.5, zorder = 3) 
    ax.plot(np.arange(0, len(res_ratio_vals[1])), res_ratio_vals[1], label = "k = 7", color = "blue", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[2])), res_ratio_vals[2], label = "k = 8", color = "green", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[3])), res_ratio_vals[3], label = "k = 9", color = "cyan", linewidth = 1.5, zorder = 3) 
  
    ax.set_yscale("log")
    
    plt.axhline(
    y=1e-8,         
    color="black",
    linestyle=":",
    linewidth=1.0
    )

    plt.title("Sparse CG (no multigrid)")
    ax.legend()
    plt.savefig("sparse_cg_no_multi_all_k.pdf", dpi=450, bbox_inches='tight')
    plt.show()


    

    # testing of working on the grid implementation of cg
    for i in range(len(k_vals)):

            Ix, Iy, It, rhsu, rhsv, h = preprocessing(sigma, dx, dy, dt, k_vals[i])    
            n, m = Ix.shape
            x, y = np.zeros((n, m)), np.zeros((n, m))


            start = perf_counter()
            cg_res = cg(x, y, Ix, Iy, 4 ** (k_vals[i]-4), rhsu, rhsv)
            stop = perf_counter()
            print(f'Time for on-the-grid OF_CG (no multigrid) = {stop - start}, k = {k_vals[i]}')
  
            u, v, res_ratio = cg_res[0], cg_res[1], cg_res[2]

            res_ratio_vals.append(res_ratio)


    fig, ax = plt.subplots(figsize=(10, 10))
 
    ax.set_xlabel(f"Iteration")
    ax.set_ylabel(r"$log_{10} \left( \frac{r_k}{r_0} \right)$")
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.plot(np.arange(0, len(res_ratio_vals[0])), res_ratio_vals[0], label = "k = 6", color = "red", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[1])), res_ratio_vals[1], label = "k = 7", color = "blue", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[2])), res_ratio_vals[2], label = "k = 8", color = "green", linewidth = 1.5, zorder = 3)
    ax.plot(np.arange(0, len(res_ratio_vals[3])), res_ratio_vals[3], label = "k = 9", color = "cyan", linewidth = 1.5, zorder = 3)
 
    ax.set_yscale("log")

    plt.axhline(
    y=1e-8,
    color="black",
    linestyle=":",
    linewidth=1.0
    )

    plt.title("On-the-grid CG (no multigrid)")
    ax.legend() 
    plt.savefig("grid_cg_no_multi_all_k.pdf", dpi=450, bbox_inches='tight')
    plt.show()


    """
    # on-the-grid CG + multigrid
    for i in range(len(k_vals)):
        Ix, Iy, It, rhsu, rhsv, h = preprocessing(sigma, dx, dy, dt, k_vals[i])

        n, m = Ix.shape
        start = perf_counter()
        vcyc_res = V_cycle(x, y, Ix, Iy, 1, rhsu, rhsv, s1=1, s2=1, level=1, max_level=2) 
        stop = perf_counter()
        print(f'Time for on-the-grid OF_CG (with multigrid) = {stop - start}, k = {k_vals[i]}')

        u, v, res_ratio = vcyc_res[0], vcyc_res[1], vcyc_res[2]
 
        res_ratio_vals.append(res_ratio)
 
    print(len(res_ratio_vals)) 
    print(len(res_ratio_vals[0][0]))
    print(len(res_ratio_vals[1][0]))
    print(len(res_ratio_vals[2][0]))
    fig, ax = plt.subplots(figsize=(10, 10))
 
    ax.set_xlabel(f"Iteration")
    ax.set_ylabel(r"$\frac{r_0}{r_k}$")
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.loglog(np.arange(0, len(res_ratio_vals[0][0])), res_ratio_vals[0][0], label = "k = 6", color = "red", linewidth = 1.5, zorder = 3)
    #ax.loglog(np.arange(0, len(res_ratio_vals[1][0])), res_ratio_vals[1][0], label = "k = 7", color = "blue", linewidth = 1.5, zorder = 3)
    #ax.loglog(np.arange(0, len(res_ratio_vals[2][0])), res_ratio_vals[2][0], label = "k = 8", color = "green", linewidth = 1.5, zorder = 3)
    #ax.loglog(np.arange(0, len(res_ratio_vals[3][0])), res_ratio_vals[3][0], label = "k = 9", color = "cyan", linewidth = 1.5, zorder = 3)


    plt.title("On-the-grid CG (with multigrid)")
    ax.legend()
    plt.show()
    """
