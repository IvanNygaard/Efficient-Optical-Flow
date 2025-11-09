import numpy as np
from utils import norm, F
from multigrid import V_cycle


def pcg(
    u0: np.ndarray,
    v0: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    lam: float,
    rhs_u: np.ndarray,
    rhs_v: np.ndarray,
    s1=2,
    s2=2,
    max_level=2,
    tol=1.0e-8,
    maxitr=2000,
    h: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    # Initialize
    Fu, Fv = F(u0, v0, Ix, Iy, lam, h)
    ru = rhs_u - Fu
    rv = rhs_v - Fv
    u = u0
    v = v0

    # Calculate the norm
    r0 = norm(ru, rv)
    r = r0  # To be updated in the iterations

    assert ru.shape == rv.shape

    # M*z0 = r0
    # zu, zv = cg(
    #     np.zeros_like(ru), np.zeros_like(rv), Ix, Iy, lam, ru, rv, 1e-8, 3, h
    # )
    zu, zv = V_cycle(
        ru,
        rv,
        Ix,
        Iy,
        lam,
        ru,
        rv,
        s1,
        s2,
        level=1,
        max_level=max_level,
    )
    pu = zu
    pv = zv

    # Define it here so the iterations work
    rk1_zk1 = np.sum(ru * zu) + np.sum(rv * zv)

    it = 0
    while it < maxitr:
        it += 1

        # Calculate alpha
        # r.T @ z
        rk_zk = rk1_zk1
        Fpu, Fpv = F(pu, pv, Ix, Iy, lam, h)
        pAp = np.sum(Fpu * pu) + np.sum(Fpv * pv)

        alpha = rk_zk / pAp

        u = u + alpha * pu
        v = v + alpha * pv

        ru = ru - alpha * Fpu
        rv = rv - alpha * Fpv

        # Break condition
        r = norm(ru, rv)
        print("Residual: ", r / r0)
        if r / r0 < tol:
            break

        # Solve Mz=r
        zu, zv = V_cycle(
            zu, zv, Ix, Iy, lam, ru, rv, s1, s2, level=1, max_level=max_level
        )
        # zu, zv = cg(zu, zv, Ix, Iy, lam, ru, rv, 1e-8, 3, h)

        rk1_zk1 = np.sum(ru * zu) + np.sum(rv * zv)
        beta = rk1_zk1 / rk_zk

        pu = zu + beta * pu
        pv = zv + beta * pv

    return u, v
