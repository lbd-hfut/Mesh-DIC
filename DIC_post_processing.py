import numpy as np
from scipy.ndimage import gaussian_filter

def DIC_smooth_Displacement(u, v, flag, smooth_sigma=1.0):
    """
    Smooth the DIC displacement field with a mask.

    Parameters
    ----------
    u, v : 2D np.ndarray
        Original displacement fields in x and y directions.
    flag : 2D np.ndarray (bool)
        Boolean mask indicating valid points (True = valid).
    smooth_sigma : float
        Standard deviation of the Gaussian kernel for smoothing.

    Returns
    -------
    u_smooth, v_smooth : 2D np.ndarray
        Smoothed displacement fields in x and y directions.
    """
    u = np.array(u, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    flag = np.array(flag, dtype=bool)

    # 创建掩码
    mask = flag.astype(np.float64)

    # 对 u、v 做掩码加权高斯滤波
    u_weighted = gaussian_filter(u * mask, sigma=smooth_sigma)
    v_weighted = gaussian_filter(v * mask, sigma=smooth_sigma)

    mask_smooth = gaussian_filter(mask, sigma=smooth_sigma)
    # 防止除零
    mask_smooth[mask_smooth == 0] = 1.0

    u_smooth = u_weighted / mask_smooth
    v_smooth = v_weighted / mask_smooth

    # 对无效点恢复原值（可选）
    u_smooth[~flag] = u[~flag]
    v_smooth[~flag] = v[~flag]

    return u_smooth, v_smooth

def DIC_Strain_from_Displacement(u, v, flag, step, SmoothLen):
    """
    Calculate strain fields (Ex, Ey, Exy) from displacement fields (u, v) using local least squares fitting.
    
    Parameters:
        u, v     : 2D np.array, displacement fields (shape: H x W)
        flag     : 2D np.array, binary mask (1 for valid points, 0 for invalid)
        step     : float, physical spacing between grid points
        SmoothLen: int, size of smoothing window (must be odd)
    
    Returns:
        Ex, Ey, Exy: strain components (same shape as input)
    """
    u = u.T
    v = v.T
    flag = flag.T

    m = SmoothLen
    if m % 2 == 0:
        m += 1  # Ensure odd
    hfm = (m - 1) // 2

    flag0 = flag.copy()

    ny, nx = u.shape
    Ex = np.full_like(u, np.nan, dtype=np.float64)
    Ey = np.full_like(u, np.nan, dtype=np.float64)
    Exy = np.full_like(u, np.nan, dtype=np.float64)

    # Pad displacement and flag arrays
    pad_shape = ((hfm, hfm), (hfm, hfm))
    u = np.pad(u, pad_shape, constant_values=np.nan)
    v = np.pad(v, pad_shape, constant_values=np.nan)
    flag = np.pad(flag, pad_shape, constant_values=0)

    for i in range(nx):
        for j in range(ny):
            if flag0[j, i] == 0:
                continue

            # Extract local window
            uu = u[j:j + m, i:i + m]
            vv = v[j:j + m, i:i + m]
            F = flag[j:j + m, i:i + m]

            U = uu.flatten()
            V = vv.flatten()
            F_flat = F.flatten()

            # Coordinate grid for least-squares fit
            xx, yy = np.meshgrid(np.arange(-hfm, hfm + 1), np.arange(-hfm, hfm + 1))
            xx = xx.flatten() * step
            yy = yy.flatten() * step
            X = np.stack([np.ones_like(xx), yy, xx], axis=1)  # shape: (m^2, 3)

            # Mask invalid points
            valid = F_flat == 1
            if np.sum(valid) > 3:
                X_valid = X[valid]
                U_valid = U[valid]
                V_valid = V[valid]

                # Least squares using numpy (more stable than manual inversion)
                a = np.linalg.lstsq(X_valid, U_valid, rcond=None)[0]
                b = np.linalg.lstsq(X_valid, V_valid, rcond=None)[0]

                Ex[j, i] = a[1]  # dU/dy
                Ey[j, i] = b[2]  # dV/dx
                Exy[j, i] = 0.5 * (a[2] + b[1])  # (dU/dx + dV/dy)/2
            else:
                # Keep as NaN
                continue
    # Transpose back to original shape
    return Ex.T, Ey.T, Exy.T
