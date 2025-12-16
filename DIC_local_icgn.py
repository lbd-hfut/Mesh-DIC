import numpy as np
from typing import Tuple
from DIC_read_image import BufferManager

SUCCESS = 1
FAILED = 0

# -------------------------
# B-spline interp
# -------------------------
def interpqbs_local(xs, ys, REF_FLAG=False, DEF_FLAG=False):
    # 1. 预构建 integer 坐标 (N,2)
    xs_floor = np.floor(xs).astype(int)
    ys_floor = np.floor(ys).astype(int)
    num_pts = len(xs)
    coords_arr = np.stack([ys_floor, xs_floor], axis=1)  # (N,2)
    
    # 2. 构建 QK_B_QKT_arr  (N,6,6)
    QK_B_QKT_arr = np.zeros((num_pts, 6, 6))
    for i in range(num_pts):
        y, x = coords_arr[i]
        if REF_FLAG:
            QK_B_QKT_arr[i] = BufferManager.QKBQKT_ref[(y, x)]
        if DEF_FLAG:
            QK_B_QKT_arr[i] = BufferManager.QKBQKT_def[(y, x)]
    
    # 3. 构造 x_vec、y_vec  (N,6)
    xd = xs - xs_floor  # (N,)
    yd = ys - ys_floor  # (N,)
    x_powers = np.stack([xd**i for i in range(6)], axis=1)  # (N,6)
    y_powers = np.stack([yd**i for i in range(6)], axis=1)  # (N,6)
    
    # 4. 做 y_vec @ M  → shape (N,6)
    tmp = np.einsum("ni,nij->nj", y_powers, QK_B_QKT_arr)
    values = np.einsum("ni,ni->n", tmp, x_powers)
    return values


# -------------------------
# Inverse compositional update 
# -------------------------
def inverse_compositional_update_2nd_order(defvector_old, delta_def):
    # Translation
    U, V = defvector_old[0], defvector_old[1]
    du, dv = delta_def[0], delta_def[1]

    # First-order derivatives (affine terms)
    dudx, dvdx, dudy, dvdy = defvector_old[2], defvector_old[3], defvector_old[4], defvector_old[5]
    d_dudx, d_dvdx, d_dudy, d_dvdy = delta_def[2], delta_def[3], delta_def[4], delta_def[5]

    # # Second-order derivatives
    # dudxx, d_dudxx = defvector_old[6] , delta_def[6]
    # dvdxx, d_dvdxx = defvector_old[7] , delta_def[7]
    # dudxy, d_dudxy = defvector_old[8] , delta_def[8]
    # dvdxy, d_dvdxy = defvector_old[9] , delta_def[9]
    # dudyy, d_dudyy = defvector_old[10], delta_def[10]
    # dvdyy, d_dvdyy = defvector_old[11], delta_def[11]
    
    # First-order affine matrices
    M_old = np.array([
        [1 + dudx, dudy, U],
        [dvdx, 1 + dvdy, V],
        [0, 0, 1]
    ])
    
    M_delta = np.array([
        [1 + d_dudx, d_dudy, du],
        [d_dvdx, 1 + d_dvdy, dv],
        [0, 0, 1]
    ])
    
    # Inverse compositional for first-order
    M_new = M_old @ np.linalg.inv(M_delta)
    
    # Updated first-order parameters
    dudx_new = M_new[0,0] - 1
    dudy_new = M_new[0,1]
    U_new = M_new[0,2]
    
    dvdx_new = M_new[1,0]
    dvdy_new = M_new[1,1] - 1
    V_new = M_new[1,2]
    
    # # Second-order parameters (直接做增量减法)
    # # 逆合成对二阶项通常采用线性近似
    # dudxx_new = dudxx - d_dudxx
    # dudxy_new = dudxy - d_dudxy
    # dudyy_new = dudyy - d_dudyy
    
    # dvdxx_new = dvdxx - d_dvdxx
    # dvdxy_new = dvdxy - d_dvdxy
    # dvdyy_new = dvdyy - d_dvdyy
    
    defvector_new = np.array([
        U_new, V_new, dudx_new, dvdx_new, dudy_new, dvdy_new
    ])
    
    # defvector_new = np.array([
    #     U_new, V_new, dudx_new, dvdx_new, dudy_new, dvdy_new,
    #     dudxx_new, dvdxx_new, dudxy_new, dvdxy_new, dudyy_new, dvdyy_new
    # ])
    
    return defvector_new

# -------------------------
# Newton (IC-GN) implementation
# -------------------------
def iterativesearch_local(
    defvector_init: np.ndarray,
    xc: np.int32,
    yc: np.int32,
    dx: np.ndarray,
    dy: np.ndarray,
    max_iter: int = 20,
    cutoff_diffnorm: float = 1e-6,
    lambda_reg: float = 1e-3
) -> Tuple[int, np.ndarray]:
    
    X_valid_corrd = xc + dx
    Y_valid_corrd = yc + dy
    
    f_buffer = BufferManager.refImg[Y_valid_corrd, X_valid_corrd]
    fm = np.sum(f_buffer) / len(f_buffer)
    deltaf_inv = np.sqrt(np.sum(((f_buffer - fm)**2)))
    if deltaf_inv < lambda_reg:
        return FAILED, defvector_init, -1
    else:
        deltaf_inv = 1 / deltaf_inv
        df_dp_buffer = np.zeros((len(f_buffer), 6), dtype=np.float32)
        df_dx = BufferManager.fx_ref[Y_valid_corrd, X_valid_corrd]
        df_dy = BufferManager.fy_ref[Y_valid_corrd, X_valid_corrd]
        df_dp_buffer[:, 0] = df_dx
        df_dp_buffer[:, 1] = df_dy
        # for products, use np.multiply with out to avoid creating X*gx temporaries
        np.multiply(dx, df_dx, out=df_dp_buffer[:, 2])   # X * fx
        np.multiply(dx, df_dy, out=df_dp_buffer[:, 3])   # X * fy
        np.multiply(dy, df_dx, out=df_dp_buffer[:, 4])   # Y * fx
        np.multiply(dy, df_dy, out=df_dp_buffer[:, 5])   # Y * fy
        # # squared terms
        # np.multiply(dx * dx * 0.5, df_dx, out=df_dp_buffer[:, 6])  # 0.5 X^2 * fx
        # np.multiply(dx * dx * 0.5, df_dy, out=df_dp_buffer[:, 7])  # 0.5 X^2 * fy
        # np.multiply(dx * dy, df_dx, out=df_dp_buffer[:, 8])        # X * Y * fx
        # np.multiply(dx * dy, df_dy, out=df_dp_buffer[:, 9])        # X * Y * fy
        # np.multiply(dy * dy * 0.5, df_dx, out=df_dp_buffer[:, 10]) # 0.5 Y^2 * fx
        # np.multiply(dy * dy * 0.5, df_dy, out=df_dp_buffer[:, 11]) # 0.5 Y^2 * fy
        # compute Hessian
        hessian_gn = df_dp_buffer.T @ df_dp_buffer
        hessian_gn = hessian_gn * 2 * (deltaf_inv**2)
        
        try:
            cholesky_G = np.linalg.cholesky(hessian_gn)
            positivedef = True
        except np.linalg.LinAlgError:
            positivedef = False  # 非正定
            
        if positivedef:
            for iter in range(max_iter):
                flag, defvector_init, diffnorm, corrcoef = newton_local(
                    defvector_init=defvector_init,
                    xc=xc, yc=yc, dx=dx, dy=dy,
                    df_dp_buffer=df_dp_buffer,
                    f_buffer=f_buffer,
                    fm=fm, deltaf_inv=deltaf_inv,
                    lambda_reg=lambda_reg,
                    cutoff_diffnorm=cutoff_diffnorm,
                    cholesky_G=cholesky_G
                    )
                if diffnorm < cutoff_diffnorm:
                    return SUCCESS, defvector_init, corrcoef
                if flag == FAILED:
                    return FAILED, defvector_init, corrcoef
            return SUCCESS, defvector_init, corrcoef
        else:
            return FAILED, defvector_init, corrcoef

# -------------------------
# Newton (IC-GN) implementation
# -------------------------
def newton_local(
    defvector_init: np.ndarray, 
    xc: np.int32,
    yc: np.int32,
    dx: np.ndarray,
    dy: np.ndarray,
    df_dp_buffer: np.ndarray,
    f_buffer: np.ndarray,
    fm: float,
    deltaf_inv: float,
    lambda_reg: float,
    cutoff_diffnorm: float,
    cholesky_G: np.ndarray
) -> Tuple[int, np.ndarray, float]:
    
    u = (defvector_init[0] +
         defvector_init[2] * dx +
         defvector_init[4] * dy )
        #  0.5 * defvector_init[6] * dx * dx +
        #  defvector_init[8] * dx * dy +
        #  0.5 * defvector_init[10] * dy * dy)

    v = (defvector_init[1] +
         defvector_init[3] * dx +
         defvector_init[5] * dy )
        #  0.5 * defvector_init[7] * dx * dx +
        #  defvector_init[9] * dx * dy +
        #  0.5 * defvector_init[11] * dy * dy)
    
    X_tilda_corrd = xc + dx + u
    Y_tilda_corrd = yc + dy + v
    try:
        g_buffer = interpqbs_local(X_tilda_corrd, Y_tilda_corrd, REF_FLAG=False, DEF_FLAG=True)
    except:
        return FAILED, defvector_init, cutoff_diffnorm*10, 10
    gm = np.sum(g_buffer) / len(g_buffer)
    deltag_inv = np.sqrt(np.sum(((g_buffer - gm)**2)))
    if deltag_inv < lambda_reg:
        return FAILED, defvector_init, cutoff_diffnorm*10, 10
    else:
        deltag_inv = 1 / deltag_inv
        normalized_diff = (f_buffer-fm)*deltaf_inv - (g_buffer-gm)*deltag_inv
        corrcoef = np.sum(normalized_diff**2)
        gradient_buffer = 2 * deltaf_inv * np.einsum('i,ij->j', normalized_diff, df_dp_buffer)
        # gradient_buffer = np.sum(gradient_buffer, axis=0)
        y = np.linalg.solve(cholesky_G, gradient_buffer)       # forward solve
        x = np.linalg.solve(cholesky_G.T, y)     # backward solve
        delta = -x
        # 计算差分范数
        diffnorm = np.linalg.norm(delta) 
        if diffnorm < cutoff_diffnorm:
            return SUCCESS, defvector_init, diffnorm, corrcoef
        else:
            defvector_new = inverse_compositional_update_2nd_order(
                defvector_init, delta)
            # print("defvector_new:")
            # print(defvector_new)
            # print("corrcoef:")
            # print(corrcoef)
            return SUCCESS, defvector_new, diffnorm, corrcoef
        
