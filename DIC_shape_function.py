import numpy as np

def shape_functions_8node(xi, eta):
    N = np.zeros(8)
    dN_dxi = np.zeros(8)
    dN_deta = np.zeros(8)
    xi2 = xi * xi
    eta2 = eta * eta
    # ---------- Shape functions N_i ----------
    N[0] = -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta)
    N[1] = -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta)
    N[2] = -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta)
    N[3] = -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta)
    N[4] = 0.5 * (1 - xi2) * (1 - eta)
    N[5] = 0.5 * (1 + xi) * (1 - eta2)
    N[6] = 0.5 * (1 - xi2) * (1 + eta)
    N[7] = 0.5 * (1 - xi) * (1 - eta2)
    # ---------- dN/dxi ----------
    dN_dxi[0] = -0.25 * ((eta - 1) * (2 * xi + eta))
    dN_dxi[1] =  0.25 * ((1 - eta) * (2 * xi - eta))
    dN_dxi[2] =  0.25 * ((1 + eta) * (2 * xi + eta))
    dN_dxi[3] = -0.25 * ((1 + eta) * (2 * xi - eta))
    dN_dxi[4] = -xi * (1 - eta)
    dN_dxi[5] =  0.5 * (1 - eta2)
    dN_dxi[6] = -xi * (1 + eta)
    dN_dxi[7] = -0.5 * (1 - eta2)
    # ---------- dN/deta ----------
    dN_deta[0] = -0.25 * ((xi - 1) * (xi + 2 * eta))
    dN_deta[1] = -0.25 * ((xi + 1) * (xi - 2 * eta))
    dN_deta[2] =  0.25 * ((xi + 1) * (xi + 2 * eta))
    dN_deta[3] =  0.25 * ((xi - 1) * (xi - 2 * eta))
    dN_deta[4] = -0.5 * (1 - xi2)
    dN_deta[5] = -eta * (1 + xi)
    dN_deta[6] =  0.5 * (1 - xi2)
    dN_deta[7] = -eta * (1 - xi)
    return N, dN_dxi, dN_deta

def shape_functions_8node_batch(xi, eta):
    xi = np.asarray(xi)
    eta = np.asarray(eta)

    Npts = xi.shape[0]

    N = np.zeros((Npts, 8))
    dN_dxi = np.zeros((Npts, 8))
    dN_deta = np.zeros((Npts, 8))

    xi2 = xi * xi
    eta2 = eta * eta

    # ---------- N_i ----------
    N[:, 0] = -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta)
    N[:, 1] = -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta)
    N[:, 2] = -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta)
    N[:, 3] = -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta)
    N[:, 4] = 0.5 * (1 - xi2) * (1 - eta)
    N[:, 5] = 0.5 * (1 + xi) * (1 - eta2)
    N[:, 6] = 0.5 * (1 - xi2) * (1 + eta)
    N[:, 7] = 0.5 * (1 - xi) * (1 - eta2)

    # ---------- dN/dxi ----------
    dN_dxi[:, 0] = -0.25 * ((eta - 1) * (2 * xi + eta))
    dN_dxi[:, 1] =  0.25 * ((1 - eta) * (2 * xi - eta))
    dN_dxi[:, 2] =  0.25 * ((1 + eta) * (2 * xi + eta))
    dN_dxi[:, 3] = -0.25 * ((1 + eta) * (2 * xi - eta))
    dN_dxi[:, 4] = -xi * (1 - eta)
    dN_dxi[:, 5] =  0.5 * (1 - eta2)
    dN_dxi[:, 6] = -xi * (1 + eta)
    dN_dxi[:, 7] = -0.5 * (1 - eta2)

    # ---------- dN/deta ----------
    dN_deta[:, 0] = -0.25 * ((xi - 1) * (xi + 2 * eta))
    dN_deta[:, 1] = -0.25 * ((xi + 1) * (xi - 2 * eta))
    dN_deta[:, 2] =  0.25 * ((xi + 1) * (xi + 2 * eta))
    dN_deta[:, 3] =  0.25 * ((xi - 1) * (xi - 2 * eta))
    dN_deta[:, 4] = -0.5 * (1 - xi2)
    dN_deta[:, 5] = -eta * (1 + xi)
    dN_deta[:, 6] =  0.5 * (1 - xi2)
    dN_deta[:, 7] = -eta * (1 - xi)

    return N, dN_dxi, dN_deta
