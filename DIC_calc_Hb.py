import os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from DIC_read_image import BufferManager
from DIC_g2l_DL import Global2Local_buffer 
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
        row_list, col_list = np.where(Global2Local_buffer.threaddiagrams == i)
        valid_flag = Global2Local_buffer.valid_mask[row_list, col_list]
        if np.sum(valid_flag) == 0:
            StiffnessMatrixBuffer.Nmat_list_elem.append([])
            StiffnessMatrixBuffer.DN_list_elem.append([])
            continue
        row_list = row_list[valid_flag]
        col_list = col_list[valid_flag]
        # 获取像素坐标及对应的单元内局部坐标
        x_px_list, y_px_list = Global2Local_buffer.plot_global_coords[row_list, col_list] 
        xi_list, eta_list = Global2Local_buffer.plot_local_coords[row_list, col_list]
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
    StiffnessMatrixBuffer.DN_list_elem = np.array(StiffnessMatrixBuffer.DN_list_elem)
    StiffnessMatrixBuffer.Nmat_list_elem = np.array(StiffnessMatrixBuffer.DN_list_elem)
    save_path_npy = os.path.join(output_dir, f"FEM_system.npz")
    np.savez(save_path_npy, A_data=A_global.tocsr().data,
         A_indices=A_global.tocsr().indices,
         A_indptr=A_global.tocsr().indptr,
         A_shape = A_global.shape,
         DN = StiffnessMatrixBuffer.DN_list_elem,
         Nmat = StiffnessMatrixBuffer.Nmat_list_elem)

        
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
        row_list, col_list = np.where(Global2Local_buffer.threaddiagrams == i)
        valid_flag = Global2Local_buffer.valid_mask[row_list, col_list]
        if np.sum(valid_flag) == 0:
            continue
        row_list = row_list[valid_flag]
        col_list = col_list[valid_flag]
        # 获取像素坐标及对应的单元内局部坐标
        x_px_list, y_px_list = Global2Local_buffer.plot_global_coords[row_list, col_list] 
        xi_list, eta_list = Global2Local_buffer.plot_local_coords[row_list, col_list]
        # 获取形函数
        N_idx = [0,2,4,6,8,10,12,14]
        Nmat_list = StiffnessMatrixBuffer.Nmat_list_elem[i-1]
        N_list =  Nmat_list[0,N_idx]
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
    
    
def global_ICGN(U_init, alpha, tol=1e-6, maxIter=100):
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
    num_nodes = U_init.shape[0]
    DIM = 2
    U = U_init.reshape(-1)  # shape (2*num_nodes,)
    
    # 记录每次迭代位移增量 norm
    norm_of_W_list = []
    
    # 获取有效自由度索引（可以自定义，例如排除固定边界点）
    ind_fem_not_zero = np.arange(len(U))  # 默认全自由度有效
    # 如果有固定自由度，可用 mask / boolean array 来排除
    
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
    
    
        
        
        
        

        
        
        
        
        
    