import numpy as np


def laplacian(w: np.ndarray, h: int) -> np.ndarray:
    """
    Calculate the discret laplacian of w, with steplength h

    Args:
    ---
    w : np.ndarray
        w
    h: int
        Stepsize corresponding to the level

    Returns:
    ---
    np.ndarray
        Discret laplacian of w
    """
    w_pad = np.pad(w, 1)
    n, m = w_pad.shape
    return (
        w_pad[0 : n - 2, 1 : m - 1]
        + w_pad[2:n, 1 : m - 1]
        + w_pad[1 : n - 1, 0 : m - 2]
        + w_pad[1 : n - 1, 2:m]
        - 4 * w_pad[1 : n - 1, 1 : m - 1]
    ) / h**2


def F(
    u: np.ndarray,
    v: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    lam: float,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    Fu = Ix**2 * u + Iy * Ix * v - lam * laplacian(u, h)
    Fv = Iy**2 * v + Iy * Ix * u - lam * laplacian(v, h)
    return Fu, Fv


def norm(ru: np.ndarray, rv: np.ndarray) -> float:
    """
    Calculate the norm of the would be vector of ru, rv if they were not represented on a grid
    """
    nru = np.linalg.norm(ru, ord="fro")
    nrv = np.linalg.norm(rv, ord="fro")
    return np.sqrt(nru**2 + nrv**2)
