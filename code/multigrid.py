import numpy as np
import scipy as sp


def V_cycle(
    u0: np.ndarray,
    v0: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    lam: float,
    rhs_u: np.ndarray,
    rhs_v: np.ndarray,
    s1: int,
    s2: int,
    level: int,
    max_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    V-cycle for the optical flow problem

    Args:
    ---
    u0 : np.ndarray
        Initial guess for u
    v0 : np.ndarray
        Initial guess for v
    Ix : np.ndarray
        x-derivative of the first frame
    Iy : np.ndarray
        y-derivative of the first frame
    lam : float
        Penalty term
    rhs_u : np.ndarray
        RHS in eq for u
    rhs_v : np.ndarray
        RHS in eq for v
    s1 : int
        Number of pre-smoothings
    s2 : int
        Number of post-smoothings
    level : int
        Current level
    max_level : int
        Total number of levels

    Returns:
    ---
    tuple[np.ndarray, np.ndarray]
        Numerical solution for u, v
    """
    # Stepsize
    h = level

    u, v = smoothing(u0, v0, Ix, Iy, lam, rhs_u, rhs_v, level, s1)
    ru_h, rv_h = residual(u, v, Ix, Iy, lam, rhs_u, rhs_v, h)
    ru_2h, rv_2h, Ix2h, Iy2h = restriction(ru_h, rv_h, Ix, Iy)
    if level == max_level - 1:
        eu_2h, ev_2h = OF_cg()
    else:
        eu_2h, ev_2h = V_cycle(
            np.zeros_like(ru_2h),
            np.zeros_like(rv_2h),
            Ix2h,
            Iy2h,
            lam,
            ru_2h,
            rv_2h,
            s1,
            s2,
            level + 1,
            max_level,
        )
    eu_h, ev_h = prolongation(eu_2h, ev_2h)
    u = u + eu_h
    v = v + ev_h
    u, v = smoothing(u, v, Ix, Iy, lam, rhs_u, rhs_v, h, s2)

    return u, v


def smoothing(
    u: np.ndarray,
    v: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    lam: float,
    rhsu: np.ndarray,
    rhsv: np.ndarray,
    s1: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smoothing using Red-Black Gauss-Seidel

    Args:
    ---
    u0 : np.ndarray
        Initial guess for u
    v0 : np.ndarray
        Initial guess for v
    Ix : np.ndarray
        x-derivative of the first frame
    Iy : np.ndarray
        y-derivative of the first frame
    lam : float
        Penalty term
    rhsu : np.ndarray
        RHS in eq for u
    rhsv : np.ndarray
        RHS in eq for v
    s1 : int
        Number of pre-smoothings
    h : int
        Stepsize corresponding to the level


    Returns:
    ---
    tuple[np.ndarray, np.ndarray]
        Smoothed u, v
    """
    # TEMP
    assert u.shape == v.shape

    # Pad u,v with zeros around (Dirichlet BC)
    u_pad = np.pad(u, 1)
    v_pad = np.pad(v, 1)

    n, m = v_pad.shape

    for _ in range(s1):
        # These two are invariant of each other
        black_update(u_pad, v_pad, Ix, Iy, lam, rhsu, h)
        red_update(v_pad, u_pad, Iy, Ix, lam, rhsv, h)

        # These last two depends on the first two
        black_update(v_pad, u_pad, Iy, Ix, lam, rhsv, h)
        red_update(u_pad, v_pad, Ix, Iy, lam, rhsu, h)

    return u_pad[1 : n - 1, 1 : m - 1], v_pad[1 : n - 1, 1 : m - 1]


def black_update(
    w: np.ndarray,
    p: np.ndarray,
    Iw: np.ndarray,
    Ip: np.ndarray,
    lam: float,
    rhs: np.ndarray,
    h: int,
) -> None:
    """
    Black update using Red-Black Gauss-Seidel.
    Note: This function mutates w

    Args:
    ---
    w : np.ndarray
        The padded array to update
    p : np.ndarray
        The padded other array
    Iw : np.ndarray
        If w=u -> Iw=Ix
        If w=v -> Iw=Iy
    Ip : np.ndarray
        If w=u -> Ip=Iy
        If w=v -> Ip=Ix
    lam : float
        Penalty term
    rhs : np.ndarray
        RHS in eq
    h : int
        Stepsize corresponding to the level

    Returns:
    ---
    None
    """
    n, m = w.shape
    k, d = Iw.shape
    # Lower update
    print(w[2 : n - 1 : 2, 1 : m - 1 : 2])
    print("LHS:")
    print(w[2 : n - 1 : 2, 0 : m - 2 : 2])
    print("RHS:")
    print(w[2 : n - 1 : 2, 2:m:2])
    print("Up:")
    print(w[1 : n - 2 : 2, 1 : m - 1 : 2])
    print("Down:")
    print(w[3:n:2, 1 : m - 1 : 2])

    print()

    w[2 : n - 1 : 2, 1 : m - 1 : 2] = (
        rhs[1:k:2, 0:d:2]
        - Ip[1:k:2, 0:d:2] * Iw[1:k:2, 0:d:2] * p[2 : n - 1 : 2, 1 : m - 1 : 2]
        + lam
        / h**2
        * (
            # LHS
            w[2 : n - 1 : 2, 0 : m - 2 : 2]
            # RHS
            + w[2 : n - 1 : 2, 2:m:2]
            # Up
            + w[1 : n - 2 : 2, 1 : m - 1 : 2]
            # Down
            + w[3:n:2, 1 : m - 1 : 2]
        )
    ) / (Ip[1:k:2, 0:d:2] ** 2 + 4 * lam)

    print(w[1 : n - 1 : 2, 2 : m - 1 : 2])
    print("LHS:")
    print(w[1 : n - 1 : 2, 1 : m - 2 : 2])
    print("RHS:")
    print(w[1 : n - 1 : 2, 3:m:2])
    print("Up:")
    print(w[0 : n - 2 : 2, 2 : m - 1 : 2])
    print("Down:")
    print(w[2:n:2, 2 : m - 1 : 2])

    # Side update
    w[1 : n - 1 : 2, 2 : m - 1 : 2] = (
        rhs[0:k:2, 1:d:2]
        - Ip[0:k:2, 1:d:2] * Iw[0:k:2, 1:d:2] * p[1 : n - 1 : 2, 2 : m - 1 : 2]
        + lam
        / h**2
        * (
            # LHS
            w[1 : n - 1 : 2, 1 : m - 2 : 2]
            # RHS
            + w[1 : n - 1 : 2, 3:m:2]
            # Up
            + w[0 : n - 2 : 2, 2 : m - 1 : 2]
            # Down
            + w[2:n:2, 2 : m - 1 : 2]
        )
    ) / (Ip[0:k:2, 1:d:2] ** 2 + 4 * lam)


def red_update(
    w: np.ndarray,
    p: np.ndarray,
    Iw: np.ndarray,
    Ip: np.ndarray,
    lam: float,
    rhs: np.ndarray,
    h: int,
) -> None:
    """
    Red update using Red-Black Gauss-Seidel
    Note: This function mutates w

    Args:
    ---
    w : np.ndarray
        The padded array to update
    p : np.ndarray
        The padded other array
    Iw : np.ndarray
        If w=u -> Iw=Ix
        If w=v -> Iw=Iy
    Ip : np.ndarray
        If w=u -> Ip=Iy
        If w=v -> Ip=Ix
    lam : float
        Penalty term
    rhs : np.ndarray
        RHS in eq
    h : int
        Stepsize corresponding to the level

    Returns:
    ---
    None
    """
    n, m = w.shape
    k, d = Iw.shape

    # Lower update
    w[1 : n - 1 : 2, 1 : m - 1 : 2] = (
        rhs[0:k:2, 0:d:2]
        - Ip[0:k:2, 0:d:2] * Iw[0:k:2, 0:d:2] * p[1 : n - 1 : 2, 1 : m - 1 : 2]
        + lam
        / h**2
        * (
            # LHS
            w[1 : n - 1 : 2, 0 : m - 2 : 2]
            # RHS
            + w[1 : n - 1 : 2, 2:m:2]
            # Up
            + w[0 : n - 2 : 2, 1 : m - 1 : 2]
            # Down
            + w[2:n:2, 1 : m - 1 : 2]
        )
    ) / (Ip[0:k:2, 0:d:2] ** 2 + 4 * lam)

    # Side update
    w[2 : n - 1 : 2, 2 : m - 1 : 2] = (
        rhs[1:k:2, 1:d:2]
        - Ip[1:k:2, 1:d:2] * Iw[1:k:2, 1:d:2] * p[2 : n - 1 : 2, 2 : m - 1 : 2]
        + lam
        / h**2
        * (
            # LHS
            w[2 : n - 1 : 2, 1 : m - 2 : 2]
            # RHS
            + w[2 : n - 1 : 2, 3:m:2]
            # Up
            + w[1 : n - 2 : 2, 2 : m - 1 : 2]
            # Down
            + w[3:n:2, 2 : m - 1 : 2]
        )
    ) / (Ip[1:k:2, 1:d:2] ** 2 + 4 * lam)


def residual(
    u: np.ndarray,
    v: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    lam: float,
    rhs_u: np.ndarray,
    rhs_v: np.ndarray,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the residual of the system

    Args:
    ---
    u : np.ndarray
        u
    v : np.ndarray
        v
    Ix : np.ndarray
        x-derivative of the first frame
    Iy : np.ndarray
        y-derivative of the first frame
    lam : float
        Penalty term
    rhs_u : np.ndarray
        RHS of eq for u
    rhs_v : np.ndarray
        RHS of eq for v
    h: int
        Stepsize corresponding to the level

    Returns:
    ---
    tuple[np.ndarray, np.ndarray]
        Residual of u, v
    """
    du = rhs_u - (Ix**2 * u + Iy * Ix * v - lam * laplacian(u, h))
    dv = rhs_v - (Ix**2 * v + Iy * Ix * u - lam * laplacian(v, h))
    return du, dv


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


def restriction(
    ru_h: np.ndarray,
    rv_h: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Restrict by power of 2

    Args:
    ---
    ru_h : np.ndarray
        Residual for u
    rv_h : np.ndarray
        Residual for v
    Ix : np.ndarray
        x-derivative of the first frame
    Iy : np.ndarray
        y-derivative of the first frame

    Returns:
    ---
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Restricted u, v, Ix, Iy
    """
    assert ru_h.shape == rv_h.shape == Ix.shape == Iy.shape

    n, m = ru_h.shape

    ru_2h = (
        ru_h[0 : n - 1 : 2, 0 : m - 1 : 2]  # upper left
        + ru_h[1:n:2, 0 : m - 1 : 2]  # lower left
        + ru_h[0 : n - 1 : 2, 1:m:2]  # upper right
        + ru_h[1:n:2, 1:m:2]  # lower right
    ) / 4
    rv_2h = (
        rv_h[0 : n - 1 : 2, 0 : m - 1 : 2]  # upper left
        + rv_h[1:n:2, 0 : m - 1 : 2]  # lower left
        + rv_h[0 : n - 1 : 2, 1:m:2]  # upper right
        + rv_h[1:n:2, 1:m:2]  # lower right
    ) / 4

    Ix2h = (
        Ix[0 : n - 1 : 2, 0 : m - 1 : 2]  # upper left
        + Ix[1:n:2, 0 : m - 1 : 2]  # lower left
        + Ix[0 : n - 1 : 2, 1:m:2]  # upper right
        + Ix[1:n:2, 1:m:2]  # lower right
    ) / 4

    Iy2h = (
        Iy[0 : n - 1 : 2, 0 : m - 1 : 2]  # upper left
        + Iy[1:n:2, 0 : m - 1 : 2]  # lower left
        + Iy[0 : n - 1 : 2, 1:m:2]  # upper right
        + Iy[1:n:2, 1:m:2]  # lower right
    ) / 4

    return ru_2h, rv_2h, Ix2h, Iy2h


def prolongation(eu_2h: np.ndarray, ev_2h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Prolongation of error on coarse grid

    Args:
    ---
    eu_2h : np.ndarray
        Residual for u (At coarse grid)
    ev_2h : np.ndarray
        Residual for v (At coarse grid)

    Returns:
    ---
    tuple[np.ndarray, np.ndarray]
        Prolongated eu_2h, ev_2h
    """
    pass


if __name__ == "__main__":
    N = 4
    M = 5
    u = np.linspace(1, N * M, N * M).reshape(N, M)
    u = np.pad(u, 1)
    v = u.copy()
    Ix, Iy = np.zeros((N, M)), np.zeros((N, M))
    rhsu, rhsv = np.zeros((N, M)), np.zeros((N, M))
    # smoothing(u, v, Ix, Iy, 1, rhsu, rhsv, 1, 1)
    print(u)
    black_update(u, v, Ix, Iy, 1, rhsu, 1)
    print(u)
