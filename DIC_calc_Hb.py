import os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from DIC_read_image import BufferManager
from DIC_g2l_DL import Global2Local_buffer 
from DIC_nodeuv_init import NodeUVInit_buffer
from DIC_shape_function import shape_functions_8node_batch

class StiffnessMatrixBuffer:
        Nmat_list_elem = []
        DN_list_elem = []
        A_global = None

def interpqbs(xs, ys, REF_FLAG=False, DEF_FLAG=False):
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

def load_StiffnessMatrixBuffer(FEM_file):
    npzfile = np.load(FEM_file, allow_pickle=True)
    StiffnessMatrixBuffer.A_global = \
        csr_matrix(
            (npzfile['A_data'], npzfile['A_indices'], npzfile['A_indptr']), 
            shape=npzfile['A_shape']
            )
    StiffnessMatrixBuffer.DN_list_elem = npzfile['DN']
    StiffnessMatrixBuffer.Nmat_list_elem = npzfile['Nmat']


def assemble_global_stiffness_Q8(alpha, output_dir):
    num_nodes  = Global2Local_buffer.nodes_coord.shape[0]
    DIM = 2
    FEMSize = DIM * num_nodes
    # Initialize global stiffness matrix in COO format
    INDEXAI, INDEXAJ, INDEXAVAL = [], [], []
    # Loop over elements to assemble global stiffness matrix
    for i, ele in enumerate(Global2Local_buffer.elements, start=1):
        # 取单元节点坐标
        conn = ele[:8]
        coords = np.array(
            [Global2Local_buffer.nodes_coord[Global2Local_buffer.id2idx[nid]] for nid in conn]
            ) # shape (8, 2)
        row_list, col_list = np.where(Global2Local_buffer.threaddiagram == i)
        valid_flag = Global2Local_buffer.plot_validpoints[row_list, col_list]
        if np.sum(valid_flag) == 0:
            StiffnessMatrixBuffer.Nmat_list_elem.append([])
            StiffnessMatrixBuffer.DN_list_elem.append([])
            continue
        row_list = row_list[valid_flag]
        col_list = col_list[valid_flag]
        # 获取像素坐标及对应的单元内局部坐标
        global_coords = Global2Local_buffer.plot_global_coords[row_list, col_list] 
        x_px_list, y_px_list = global_coords[:,0], global_coords[:,1]
        local_coords = Global2Local_buffer.plot_local_coords[row_list, col_list]
        xi_list, eta_list = local_coords[:,0], local_coords[:,1]
        # 计算形函数及其导数
        N_list, dN_dxi_list, dN_deta_list = shape_functions_8node_batch(xi_list, eta_list)
        # 获取参考图像梯度
        DfDx_list = BufferManager.fx_ref[row_list, col_list]
        DfDy_list = BufferManager.fy_ref[row_list, col_list]
        # 计算雅可比矩阵 J 批量
        J11 = np.sum(dN_dxi_list  * coords[:, 0], axis=1)
        J12 = np.sum(dN_dxi_list  * coords[:, 1], axis=1)
        J21 = np.sum(dN_deta_list * coords[:, 0], axis=1)
        J22 = np.sum(dN_deta_list * coords[:, 1], axis=1)
        detJ = J11*J22 - J12*J21
        invJ = np.zeros((len(xi_list), 2, 2))
        invJ[:,0,0] =  J22 / detJ
        invJ[:,0,1] = -J12 / detJ
        invJ[:,1,0] = -J21 / detJ
        invJ[:,1,1] =  J11 / detJ
        # 构建局部形函数矩阵 N (2 × 16) 批量
        num_pix = len(xi_list)
        Nmat_list = np.zeros((num_pix, 2, 16))
        for k in range(8):
            u_col = 2 * k
            v_col = u_col + 1
            Nmat_list[:, 0, u_col] = N_list[:, k]   # N_k 对 u
            Nmat_list[:, 1, v_col] = N_list[:, k]   # N_k 对 v
        StiffnessMatrixBuffer.Nmat_list_elem.append(Nmat_list)
        # 构建局部 DN (4 × 16) 批量
        DN_local = np.zeros((num_pix, 4, 16))
        for k in range(8):
            u_col = 2 * k
            v_col = u_col + 1
            # 对 u 自由度
            DN_local[:, 0, u_col] = dN_dxi_list[:, k]    # dN/dξ
            DN_local[:, 1, u_col] = dN_deta_list[:, k]   # dN/dη
            # 对 v 自由度
            DN_local[:, 2, v_col] = dN_dxi_list[:, k]    
            DN_local[:, 3, v_col] = dN_deta_list[:, k]
        # 将 DN_local 从 (ξ,η) 转换到 (x,y)：DN_global = T * DN_local
        # T: (P,4,4),  DN_local: (P,4,16) → DN_global: (P,4,16)
        DN_global = np.zeros_like(DN_local)
        T = np.zeros((num_pix, 4, 4), dtype=detJ.dtype)
        T[:, :2, :2] = invJ
        T[:, 2:, 2:] = invJ
        DN_global = np.einsum('pab,pbc->pac', T, DN_local)
        StiffnessMatrixBuffer.DN_list_elem.append(DN_global)
        # 参考图像梯度：Df = [df/dx, df/dy]^T
        Df = np.stack([DfDx_list, DfDy_list], axis=1)   # (num_pix, 2)
        # 计算单元局部刚度矩阵 A_e (16 × 16)
        # --- 图像项 A_img ---
        # g = Np^T df  (P, 2, 16).T@(P, 2)  → (P, 16, 1)
        g = np.einsum('pik,pi->pk', Nmat_list, Df)[:, :, None]  # (P, 16, 1)
        # g g^T shape (P, 16, 1)@(P, 16, 1).T → (P, 16, 16)
        A_img_all = np.einsum('pik,pjk->pij', g, g)
        # --- 正则项 A_reg ---
        # DN_global^T DN_global → shape (P, 16, 16)
        A_reg_all = alpha * np.einsum('pki,pkj->pij', DN_global, DN_global)
        # --- 逐点求和，得到最终 A_e ---
        A_e = A_img_all.sum(axis=0) + A_reg_all.sum(axis=0)   # (16,16)
        # 装配到全局矩阵：自由度映射
        tempIndexU = np.zeros(16, dtype=int)
        for j, node in enumerate(conn):
            global_idx = Global2Local_buffer.id2idx[node]
            tempIndexU[2*j]   = DIM * global_idx     # u dof
            tempIndexU[2*j+1] = DIM * global_idx + 1 # v dof
        
        idx_row, idx_col = np.meshgrid(tempIndexU, tempIndexU)
        INDEXAI.append(idx_row.ravel())
        INDEXAJ.append(idx_col.ravel())
        INDEXAVAL.append(A_e.ravel())
    # 合并所有单元并返回 CSR 稀疏矩阵
    INDEXAI = np.concatenate(INDEXAI)
    INDEXAJ = np.concatenate(INDEXAJ)
    INDEXAVAL = np.concatenate(INDEXAVAL)
    A_global = coo_matrix((INDEXAVAL, (INDEXAI, INDEXAJ)), shape=(FEMSize, FEMSize))
    A_global = A_global.tocsr()
    StiffnessMatrixBuffer.A_global = A_global
    # save_path_npy = os.path.join(output_dir, f"FEM_system.npz")
    # np.savez(save_path_npy, A_data=A_global.tocsr().data,
    #      A_indices=A_global.tocsr().indices,
    #      A_indptr=A_global.tocsr().indptr,
    #      A_shape = A_global.shape,
    #      DN = StiffnessMatrixBuffer.DN_list_elem,
    #      Nmat = StiffnessMatrixBuffer.Nmat_list_elem)

        
def assemble_global_residual_Q8(node_uv, alpha):
    num_nodes  = Global2Local_buffer.nodes_coord.shape[0]
    DIM = 2
    FEMSize = DIM * num_nodes
    # Initialize global residual vector
    INDEXBI, INDEXBVAL = [], []
    # Loop over elements to assemble global residual vector
    for i, ele in enumerate(Global2Local_buffer.elements, start=1):
        # 取单元节点坐标
        conn = ele[:8]
        ele_node_uv = np.array([node_uv[Global2Local_buffer.id2idx[nid]] for nid in conn]) # shape (8, 2)
        ele_node_u = ele_node_uv[:, 0]
        ele_node_v = ele_node_uv[:, 1]
        row_list, col_list = np.where(Global2Local_buffer.threaddiagram == i)
        valid_flag = Global2Local_buffer.plot_validpoints[row_list, col_list]
        if np.sum(valid_flag) == 0:
            continue
        row_list = row_list[valid_flag]
        col_list = col_list[valid_flag]
        # 获取像素坐标及对应的单元内局部坐标
        global_coords = Global2Local_buffer.plot_global_coords[row_list, col_list] 
        x_px_list, y_px_list = global_coords[:,0], global_coords[:,1]
        local_coords = Global2Local_buffer.plot_local_coords[row_list, col_list]
        xi_list, eta_list = local_coords[:,0], local_coords[:,1]
        # 获取形函数
        Nmat_list = StiffnessMatrixBuffer.Nmat_list_elem[i-1]
        N_list, _, _ = shape_functions_8node_batch(xi_list, eta_list)
        # 计算当前变形后像素坐标
        u_px_list = np.einsum('pk,k->p', N_list, ele_node_u)
        v_px_list = np.einsum('pk,k->p', N_list, ele_node_v)
        # 灰度残差
        r_img  = (BufferManager.refImg[row_list, col_list] - interpqbs(
            x_px_list + u_px_list, y_px_list + v_px_list, REF_FLAG=False, DEF_FLAG=True)
        )[:, None] # (P,1)
        # 获取参考图像梯度
        DfDx_list = BufferManager.fx_ref[row_list, col_list]
        DfDy_list = BufferManager.fy_ref[row_list, col_list]
        Df = np.stack([DfDx_list, DfDy_list], axis=1)
        # 获取全局DN (P,4,16) 批量
        DN_global = StiffnessMatrixBuffer.DN_list_elem[i-1]
        # 计算 g = Nᵀ Df   (P,16)
        # N_list: (P,2,16),  Df:(P,2)
        g = np.einsum('pik,pi->pk', Nmat_list, Df)[:, :, None]  # (P, 16, 1)
        # 图像残差项: g * r_img -> (P,16,1)
        b_img = g * r_img[:, None, :]  # (P,16,1)
        
        # 计算正则化项
        # DN_global: (P,4,16), U_e: (16,)
        U_e = np.zeros(16)
        for j, nid in enumerate(conn):
            idx = Global2Local_buffer.id2idx[nid]
            U_e[2*j]   = node_uv[idx, 0]
            U_e[2*j+1] = node_uv[idx, 1]
        U_e = U_e[:, None]  # (16,1)
        # DN_global^T DN_global * U_e -> (P,16,16)@(16,1) -> (P,16,1)
        reg_term = np.einsum('pki,pkj->pij', DN_global, DN_global)  # (P,16,16)
        reg_term = np.einsum('pij,jk->pik', reg_term, U_e)           # (P,16,1)
        reg_term = alpha * reg_term                                   # (P,16,1)
        # 单元残差: 按像素累加
        b_e = np.sum(b_img - reg_term, axis=0)  # (16,1)
        # 装配到全局残差向量
        tempIndexU = np.zeros(16, dtype=int)
        for j, nid in enumerate(conn):
            idx = Global2Local_buffer.id2idx[nid]
            tempIndexU[2*j]   = 2 * idx
            tempIndexU[2*j+1] = 2 * idx + 1

        INDEXBI.append(tempIndexU)
        INDEXBVAL.append(b_e[:, 0])
        
    # 合并所有元素
    INDEXBI = np.concatenate(INDEXBI)
    INDEXBVAL = np.concatenate(INDEXBVAL)
    b_global = np.zeros(FEMSize, dtype=float)
    b_global[INDEXBI] = INDEXBVAL
    return b_global
    
    
def global_ICGN(alpha, tol=1e-6, maxIter=100):
    """
    全局 ICGN 迭代求解
    参数:
        U_init: 初始位移 (num_nodes, 2)
        alpha: 正则化系数
        tol: 收敛阈值
        maxIter: 最大迭代次数
    返回:
        U: 收敛后的位移 (num_nodes, 2)
        norm_of_W_list: 每次迭代位移增量的规范化列表
    """
    # 将位移展开为一维向量
    U_init = NodeUVInit_buffer.nodes_coord_uv.copy()
    num_nodes = U_init.shape[0]
    DIM = 2
    U = U_init.reshape(-1)  # shape (2*num_nodes,)
    # 记录每次迭代位移增量 norm
    norm_of_W_list = []
    A = StiffnessMatrixBuffer.A_global.copy()
    # -----------------------------------------------------
    # 自动检测是否存在全 0 行（或无对角线），并构建可用自由度
    # -----------------------------------------------------
    n = A.shape[0]
    # 每行元素数量：indptr[k+1] - indptr[k]
    row_nnz = A.indptr[1:] - A.indptr[:-1]
    row_not_zero = row_nnz > 0
    # 对角线，若 CSR 中查不到 diag，则对角线对应位置为 0
    diag = A.diagonal()
    diag_not_zero = (np.abs(diag) > 1e-14)
    # 自由度有效：行非空 且 对角线非零
    valid_dof_mask = row_not_zero & diag_not_zero
    ind_fem_not_zero = np.where(valid_dof_mask)[0]
    print(f"[ICGN] 有效 DOF 数量: {len(ind_fem_not_zero)} / {n}")
    if len(ind_fem_not_zero) == 0:
        raise RuntimeError("全部自由度均无效（全 0 行或无对角线）！")
    
    # -----------------------------------------------------
    # 迭代
    # -----------------------------------------------------
    for step in range(maxIter):
        # 1. 计算全局残差向量 b
        b_global = assemble_global_residual_Q8(U.reshape(num_nodes, 2), alpha)  # shape (2*num_nodes,)
        
        # 2. 求解自由度增量 W
        # 只取有效自由度
        W = spsolve(StiffnessMatrixBuffer.A_global[ind_fem_not_zero, :][:, ind_fem_not_zero], b_global[ind_fem_not_zero])
        
        # 3. 计算规范化位移增量
        normW = np.linalg.norm(W)/np.sqrt(len(W))
        norm_of_W_list.append(normW)
        
        # 5. 更新位移
        U[ind_fem_not_zero] += W
        # if (step+1) % (maxIter//10) == 0:
        print(f"Step {step+1}/{maxIter}, normW: {normW:.5e}")
        
        # 6. 收敛判断
        if normW < tol:
            print(f"ICGN 收敛 at iter {step+1}, normW = {normW:.3e}")
            break
        elif normW > 1e6:  # 简单发散判定，可调整
            print(f"ICGN 发散 at iter {step+1}, normW = {normW:.3e}")
            break
        else:
            print(f"ICGN iter {step+1}, normW = {normW:.3e}")
    
    # 返回位移矩阵 (num_nodes, 2)
    return U.reshape(num_nodes, DIM), norm_of_W_list
    
def interp_uv_strain(node_uv):
    num_nodes  = Global2Local_buffer.nodes_coord.shape[0]
    plot_u = np.zeros_like(BufferManager.refImg, dtype=np.float32)
    plot_v = np.zeros_like(BufferManager.refImg, dtype=np.float32)
    plot_ex = np.zeros_like(BufferManager.refImg, dtype=np.float32)
    plot_ey = np.zeros_like(BufferManager.refImg, dtype=np.float32)
    plot_rxy = np.zeros_like(BufferManager.refImg, dtype=np.float32)
    
    # Loop over elements to assemble deformation matrix 
    for i, ele in enumerate(Global2Local_buffer.elements, start=1):
        # -------------------------
        # 1. 单元节点坐标和位移
        # -------------------------
        conn = ele[:8]
        ele_node_xy = np.array(
            [Global2Local_buffer.nodes_coord[Global2Local_buffer.id2idx[nid]] for nid in conn]
            ) # shape (8, 2)
        ele_node_uv = np.array([node_uv[Global2Local_buffer.id2idx[nid]] for nid in conn]) # shape (8, 2)
        ele_node_u = ele_node_uv[:, 0]
        ele_node_v = ele_node_uv[:, 1]
        # -------------------------
        # 2. 单元像素提取
        # -------------------------
        row_list, col_list = np.where(Global2Local_buffer.threaddiagram == i)
        valid_flag = Global2Local_buffer.plot_validpoints[row_list, col_list]
        if np.sum(valid_flag) == 0:
            continue
        row_list = row_list[valid_flag]
        col_list = col_list[valid_flag]
        local_coords = Global2Local_buffer.plot_local_coords[row_list, col_list]
        xi_list, eta_list = local_coords[:,0], local_coords[:,1]
        # -------------------------
        # 3. 形函数与导数（局部导数）
        # -------------------------
        N_list, dN_dxi_list, dN_deta_list = shape_functions_8node_batch(xi_list, eta_list)
        # -------------------------
        # 4. 计算像素位移
        # -------------------------
        u_px_list = np.einsum('pk,k->p', N_list, ele_node_u)
        v_px_list = np.einsum('pk,k->p', N_list, ele_node_v)
        plot_u[row_list, col_list] = u_px_list
        plot_v[row_list, col_list] = v_px_list
        # -------------------------
        # 5. 计算 Jacobian (x,y ← ξ,η)
        # -------------------------
        # dx/dξ = Σ dN/dξ * x_k
        dx_dxi  = np.einsum('pk,k->p', dN_dxi_list,  ele_node_xy[:,0])
        dx_deta = np.einsum('pk,k->p', dN_deta_list, ele_node_xy[:,0])
        dy_dxi  = np.einsum('pk,k->p', dN_dxi_list,  ele_node_xy[:,1])
        dy_deta = np.einsum('pk,k->p', dN_deta_list, ele_node_xy[:,1])
        # Jacobian determinant
        detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
        invJ_11 =  dy_deta / detJ
        invJ_12 = -dx_deta / detJ
        invJ_21 = -dy_dxi  / detJ
        invJ_22 =  dx_dxi  / detJ
        # -------------------------
        # 6. 计算局部导数：du/dξ, du/dη, dv/dξ, dv/dη
        # -------------------------
        du_dxi  = np.einsum('pk,k->p', dN_dxi_list, ele_node_u)
        du_deta = np.einsum('pk,k->p', dN_deta_list, ele_node_u)
        dv_dxi  = np.einsum('pk,k->p', dN_dxi_list, ele_node_v)
        dv_deta = np.einsum('pk,k->p', dN_deta_list, ele_node_v)
        # -------------------------
        # 7. 转换到全局坐标系
        # -------------------------
        du_dx = invJ_11 * du_dxi + invJ_12 * du_deta
        du_dy = invJ_21 * du_dxi + invJ_22 * du_deta
        dv_dx = invJ_11 * dv_dxi + invJ_12 * dv_deta
        dv_dy = invJ_21 * dv_dxi + invJ_22 * dv_deta
        # -------------------------
        # 8. 计算像素应变
        # -------------------------
        ex_list  = du_dx
        ey_list  = dv_dy
        rxy_list = (du_dy + dv_dx) / 2
        plot_ex[row_list, col_list] = ex_list
        plot_ey[row_list, col_list] = ey_list
        plot_rxy[row_list, col_list] = rxy_list
        
    return plot_u, plot_v, plot_ex, plot_ey, plot_rxy
        
        
        
        

        
        
        
        
        
    