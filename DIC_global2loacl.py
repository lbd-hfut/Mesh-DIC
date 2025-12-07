import os
import time
import numpy as np
from threading import RLock, Event
from collections import deque
from tqdm import tqdm
from scipy.io import savemat
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from DIC_create_mesh import read_elements, read_nodes
from DIC_shape_function import shape_functions_8node

class Global2Local_buffer:
    nodes_coord = None
    id2idx = None
    elements = None
    Inform = None
    threaddiagram = None
    plot_calcpoints = None
    plot_validpoints = None
    plot_J = None
    plot_global_coords = None
    plot_local_coords = None
    mask = None
    seeds_info = None
    # write data lock
    data_lock = RLock()
    parallel_flag = None
    max_workers = None
    plot_Jn = None

SUCCESS = 1
FAILED = 0

class Comp_global2local:
    def __init__(self, config, mask):
        self.mesh_dir = config.mesh_dir
        Global2Local_buffer.mask = mask
        Global2Local_buffer.parallel_flag = config.parallel
        Global2Local_buffer.max_workers = config.max_workers
        self.stop_event = Event()
        
    def load_mesh_buffer(self):
        nodes_file = os.path.join(self.mesh_dir, "nodes.txt")
        elements_file = os.path.join(self.mesh_dir, "elements.txt")
        Inform_file = os.path.join(self.mesh_dir, "Inform.npy")
        Global2Local_buffer.nodes_coord, Global2Local_buffer.id2idx = read_nodes(nodes_file)
        Global2Local_buffer.elements = read_elements(elements_file)
        Global2Local_buffer.Inform = np.load(Inform_file)
        Global2Local_buffer.threaddiagram = build_eie_idx_matrix(
            Inform=Global2Local_buffer.Inform,
            mask=Global2Local_buffer.mask
        )
        Global2Local_buffer.plot_calcpoints  = np.zeros_like(Global2Local_buffer.mask, dtype=bool)
        Global2Local_buffer.plot_validpoints = np.zeros_like(Global2Local_buffer.mask, dtype=bool)
        H, W = Global2Local_buffer.mask.shape
        Global2Local_buffer.plot_J = np.zeros((H, W, 2, 2))
        Global2Local_buffer.plot_global_coords = np.zeros((H, W, 2))
        Global2Local_buffer.plot_local_coords = np.zeros((H, W, 2))
        Global2Local_buffer.plot_Jn = np.zeros((H, W))
        read_seeds_info()
        
    def solve(self):
        self.stop_event.clear()  # 每次开始前清空停止标志
        queues = [deque() for _ in Global2Local_buffer.seeds_info]
        n_points = len(Global2Local_buffer.Inform)
        print(f"总点数 = {n_points}")
        pbar_lock = RLock()
        global_pbar = tqdm(total=n_points, desc="Solving Global points to local points", unit="pt")
        def worker(queue, seed_info):
            analysis_queue(
                queue=queue,
                seed_info=seed_info,
                pbar=global_pbar,
                pbar_lock=pbar_lock,
                stop_event=self.stop_event
            )
        # ========== 单线程模式 ==========
        if not Global2Local_buffer.parallel_flag:
            for i in range(len(Global2Local_buffer.seeds_info)):
                # try:
                worker(queues[i], Global2Local_buffer.seeds_info[i])
                # except Exception as exc:
                #     print(f"Seed {i} raised exception: {exc}")
            global_pbar.close()
            return    # 单线程直接返回即可
        # ========== 多线程模式 ==========
        try:
            max_workers = Global2Local_buffer.max_workers

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_seed = {
                    executor.submit(worker, queues[i], Global2Local_buffer.seeds_info[i]): i
                    for i in range(len(Global2Local_buffer.seeds_info))}
                for future in as_completed(future_to_seed):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"Seed {future_to_seed[future]} raised exception: {exc}")
            global_pbar.close()
        except KeyboardInterrupt:
            print("用户终止，停止所有线程...")
            global_pbar.close()
            self.stop_event.set()
            
    def save_results(self):
        output_dir = self.mesh_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"global2local_J.mat")
        # 构建要保存的字典，对应 MATLAB struct
        dic_struct = {
            
            'plot_calcpoints': Global2Local_buffer.plot_calcpoints,
            'plot_validpoints': Global2Local_buffer.plot_validpoints,
            'plot_global_coords': Global2Local_buffer.plot_global_coords,
            'plot_local_coords': Global2Local_buffer.plot_local_coords,
            'eie_idx_matrix': Global2Local_buffer.threaddiagram,
            'plot_J': Global2Local_buffer.plot_J,
            'seeds_info ': Global2Local_buffer.seeds_info ,
            'cond_Jn': Global2Local_buffer.plot_Jn
        }
        # 使用 savemat 保存，结构体形式
        savemat(save_path, {'global2local_J': dic_struct})
        print(f"Saved MATLAB .mat file: {save_path}")
        
    def plot_mesh_calcpoints(self, plot_var=None, var_name="plot_validpoints"):
        # 获取 plot_validpoints 的形状（用于后续坐标对齐）
        H, W = plot_var.shape
        # 创建画布
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_aspect("equal")
        # 绘制 plot_validpoints 作为背景（灰度图）
        ax.imshow(
            plot_var,  
            extent=[0, W, H, 0],  # 定义坐标范围（假设 plot_validpoints 是 [0, W] x [0, H]）
            origin="upper",      # 图像原点在左下
            cmap="binary",       # 黑白背景
            alpha=0.5            # 半透明
        )
        # 绘制 Quad9 的四边形边界（前8个节点）
        draw_order = [0, 4, 1, 5, 2, 6, 3, 7, 8]  # 二阶边顺序
        for eid, conn in enumerate(Global2Local_buffer.elements, start=1):
            quad9_nodes = np.array([Global2Local_buffer.nodes_coord[Global2Local_buffer.id2idx[nid]] for nid in conn])
            quad9_nodes_reordered  = quad9_nodes[draw_order]
            pts_polygon = quad9_nodes_reordered[:8]
            # 闭合 polygon
            pts_closed   = np.vstack([pts_polygon, pts_polygon[0]])
            # 画出单元边界
            ax.plot(pts_closed[:, 0], pts_closed[:, 1], '-k')
            center_id = conn[8]
            ax.plot(pts_closed[:, 0], pts_closed[:, 1], '-k')
            center_id = conn[8]
            for nid in conn:
                x, y = Global2Local_buffer.nodes_coord[Global2Local_buffer.id2idx[nid]]
                if nid == center_id:
                    plt.text(x, y, str(eid), color='blue', fontsize=4)
                    continue
                plt.text(x, y, str(nid), color='red', fontsize=4)
        save_path = os.path.join(self.mesh_dir, var_name+".png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

## --------- 函数部分 ---------
def analysis_queue(queue, seed_info, pbar, pbar_lock, stop_event):
    center_global_coord, center_local_coord, quad8_nodes, J, num_thread = seed_info
    center_global_coord = center_global_coord.astype(int)
    outstate, residual, local_coord, Jcobi = analyzepoint(
        queue, center_global_coord, center_local_coord, 
        quad8_nodes, J, num_thread,
        pbar, pbar_lock, stop_event
    )
    if outstate==FAILED:
        paramvector = (center_global_coord, local_coord, Jcobi, num_thread)
        queue.append(paramvector)
    # print(f"Thread {num_thread} initial point done: state={outstate}, residual={residual:.4f}")
    while queue:
        if stop_event.is_set():   # ★ 线程立即退出
            return
        prePoint_info = queue.popleft()
        global_coord, local_coord, Jcobi, num_thread = prePoint_info
        x, y = global_coord
        with Global2Local_buffer.data_lock:
            Global2Local_buffer.plot_J[y,x,:,:] = Jcobi[:,:]
            Global2Local_buffer.plot_global_coords[y,x,:] = global_coord[:]
            Global2Local_buffer.plot_local_coords[y,x,:] = local_coord[:]
        # 四邻域
        neighs = [
                    global_coord + np.array([ 0, 1]),
                    global_coord + np.array([ 0,-1]),
                    global_coord + np.array([ 1, 0]),
                    global_coord + np.array([-1, 0])
                ]
        for global_coord in neighs:
            analyzepoint(
                queue, global_coord, local_coord, 
                quad8_nodes, Jcobi, num_thread,
                pbar, pbar_lock, stop_event
                )
        if queue:
            continue
        else:
            with Global2Local_buffer.data_lock:
                # 查找该线程未计算点
                ys_u, xs_u = np.where(
                    (Global2Local_buffer.threaddiagram == num_thread) & ~Global2Local_buffer.plot_calcpoints
                    )
            Num_left_points =len(ys_u)
            if Num_left_points== 0:
                continue
            else:
                # print(f"Thread {num_thread}: 寻找未计算点，剩余 {Num_left_points} 点")
                # 计算距离
                last_global_coord = global_coord
                last_local_coord = local_coord
                last_Jcobi = Jcobi
                dx = xs_u - last_global_coord[0]
                dy = ys_u - last_global_coord[1]
                dist2 = dx * dx + dy * dy
                # 距离最短的点
                idx = np.argmin(dist2)
                x_uncal = xs_u[idx]
                y_uncal = ys_u[idx]
                outstate, residual, local_coord, Jcobi = analyzepoint(
                    queue, np.array([x_uncal, y_uncal]), 
                    last_local_coord, 
                    quad8_nodes, last_Jcobi, num_thread,
                    pbar, pbar_lock, stop_event
                )
                if outstate==FAILED:
                    paramvector = (np.array([x_uncal, y_uncal]), local_coord, Jcobi, num_thread)
                    queue.append(paramvector)
            

def analyzepoint(
    queue, 
    global_coord, 
    local_coord_init, 
    quad8_nodes,
    J_init,
    num_thread, 
    pbar, pbar_lock, 
    stop_event
):
    if stop_event.is_set():   # 立即退出该点计算
        return
    cutoff_residual = 0.5
    x,y = global_coord
    H, W = Global2Local_buffer.mask.shape
    if x < 0 or x >= W or y < 0 or y >= H:
        return
    if not Global2Local_buffer.mask[y, x]:
        return
    if Global2Local_buffer.plot_calcpoints[y,x]:
        return
    if Global2Local_buffer.threaddiagram[y, x] != num_thread:
        return
    # ---------------- 执行 calcpoint ----------------
    outstate, residual, local_coord, Jcobi = solve_point(
        global_coord=global_coord, quad8_nodes=quad8_nodes)
    # ---------------- 加入队列 ----------------
    if (outstate == SUCCESS and
        residual < cutoff_residual):
        paramvector = (global_coord, local_coord, Jcobi, num_thread)
        queue.append(paramvector)
        with Global2Local_buffer.data_lock:
            Global2Local_buffer.plot_validpoints[y,x] = True
    else:
        # 中心点不收敛
        outstate, residual_night, local_coord_night, Jcobi_night = cal_point_g2L(
            global_coord=global_coord,
            local_coord_init=local_coord_init,
            quad8_nodes=quad8_nodes,
            J_init=J_init)
        if outstate == SUCCESS:
            paramvector = (global_coord, local_coord_night, Jcobi_night, num_thread)
            queue.append(paramvector)
            with Global2Local_buffer.data_lock:
                Global2Local_buffer.plot_validpoints[y,x] = True
        else:
            if residual_night < residual:
                local_coord = local_coord_night
                Jcobi = Jcobi_night
            with Global2Local_buffer.data_lock:
                Global2Local_buffer.plot_J[y,x,:,:] = Jcobi[:,:]
                Global2Local_buffer.plot_global_coords[y,x,:] = global_coord[:]
                Global2Local_buffer.plot_local_coords[y,x,:] = local_coord[:]
    # ---------------- 标记已计算 ----------------
    with Global2Local_buffer.data_lock:
        Global2Local_buffer.plot_calcpoints[y,x] = True
    # ---------------- 线程更新进度 ----------------
    if pbar is not None and pbar_lock is not None:
        with pbar_lock:
            pbar.update(1)
    return outstate, residual, local_coord, Jcobi

def solve_point(
    global_coord, 
    quad8_nodes,
    tol_Global: float = 0.1,
    max_iter=1000,
    xi0=0, 
    eta0=0, 
    debug: bool = False
):
    xi, eta = xi0, eta0
    if debug:
        print(f"\n==== Solve global point {global_coord} ====")
    status = FAILED
    for it in range(max_iter):
        N, dN_dxi, dN_deta = shape_functions_8node(xi, eta)
        x_m = np.dot(N, quad8_nodes[:,0])
        y_m = np.dot(N, quad8_nodes[:,1])
        R = np.array([global_coord[0] - x_m, global_coord[1] - y_m])

        J = np.array([
            [np.dot(dN_dxi, quad8_nodes[:,0]), np.dot(dN_deta, quad8_nodes[:,0])],
            [np.dot(dN_dxi, quad8_nodes[:,1]), np.dot(dN_deta, quad8_nodes[:,1])]
        ])
        if debug:
            if (it + 1) % 200 == 0 or it == 0:
                print(f"it={it+1}: xi={xi:.6f} eta={eta:.6f}  R={R}  det(J)={np.linalg.det(J):.4e}")
        if np.linalg.norm(R) < tol_Global:
            status = SUCCESS
            break

        delta = np.linalg.solve(J, R)
        xi += delta[0]
        eta += delta[1]

    return status, np.linalg.norm(R), np.array([xi, eta], dtype=float), J

def cal_point_g2L(
    global_coord: np.ndarray,       # shape = (2,) 全局点 (x,y)
    local_coord_init: np.ndarray,   # 初始局部坐标 [xi0, eta0]
    quad8_nodes: np.ndarray,        # Q8 单元节点坐标，shape=(8,2)
    J_init: np.ndarray = None,      # 可选初始Jacobian
    tol_Global: float = 0.1,
    tol_Local: float = 1e-8,
    max_iter: int = 1000,
    debug: bool = False,
):
    # 1) 对节点和全局点做归一化
    xmin, xmax = quad8_nodes[:,0].min(), quad8_nodes[:,0].max()
    ymin, ymax = quad8_nodes[:,1].min(), quad8_nodes[:,1].max()
    # xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    sx = 0.5 * (xmax - xmin) * 2
    sy = 0.5 * (ymax - ymin) * 2
    # 避免除零
    sx = sx if sx != 0 else 1.0
    sy = sy if sy != 0 else 1.0
    # 节点归一化到 [-1,1]
    nodes_norm = np.empty_like(quad8_nodes)
    nodes_norm[:, 0] = (quad8_nodes[:, 0] - cx) / sx
    nodes_norm[:, 1] = (quad8_nodes[:, 1] - cy) / sy
    # 全局点归一化
    point_norm = np.array([(global_coord[0] - cx) / sx,
                           (global_coord[1] - cy) / sy])
    
    prev1 = None
    prev2 = None
    
    def res_orig(xi_, eta_):
        Nloc, _, _ = shape_functions_8node(xi_, eta_)
        rx = global_coord[0] - float(np.dot(Nloc, quad8_nodes[:,0]))
        ry = global_coord[1] - float(np.dot(Nloc, quad8_nodes[:,1]))
        return np.hypot(rx, ry)
    
    # 2) Newton 迭代（归一化空间）
    xi, eta = local_coord_init.copy()
    # xi, eta = 0.0, 0.0
    if debug:
        print("golbal_coord:", global_coord)
    # iteration
    max_step = 0.3 # 最大步长限制
    for it in range(max_iter):
        # 形函数与一阶导数
        N, dN_dxi, dN_deta = shape_functions_8node(xi, eta)
        # 当前点（归一化空间）的全局映射 x(ξ,η)
        x_m = np.dot(N, nodes_norm[:, 0])
        y_m = np.dot(N, nodes_norm[:, 1])
        # 残差：目标点 - 映射点
        Rn = np.array([point_norm[0] - x_m,
                       point_norm[1] - y_m])
        # 计算雅可比矩阵 J（归一化空间）
        Jn11 = np.dot(dN_dxi,  nodes_norm[:, 0])   # dx/dxi
        Jn12 = np.dot(dN_deta, nodes_norm[:, 0])   # dx/deta
        Jn21 = np.dot(dN_dxi,  nodes_norm[:, 1])   # dy/dxi
        Jn22 = np.dot(dN_deta, nodes_norm[:, 1])   # dy/deta
        Jn  = np.array([[Jn11, Jn12],
                        [Jn21, Jn22]])
        # 收敛判断
        resG = res_orig(xi, eta)
        if debug:
            print(f"it={it} xi={xi:.6g} eta={eta:.6g} resG={resG:.6g} det(Jn)={np.linalg.det(Jn):.6g}")
        if resG < tol_Global:
            status = SUCCESS
            break
        # Newton 增量 Δ = J^{-1} R
        try:
            delta = np.linalg.solve(Jn , Rn)
        except np.linalg.LinAlgError:
            # 若雅可比奇异，用最小二乘兜底
            delta, *_ = np.linalg.lstsq(Jn, Rn, rcond=None)
        if np.linalg.det(Jn) < 1e-2 and np.linalg.norm(delta, ord=np.inf) > max_step:
            delta = delta * (max_step / (np.linalg.norm(delta, ord=np.inf) + 1e-16))
        # try full step, but with damping/backtracking and 2-cycle detection
        alpha = 1.0
        xi_trial = xi + alpha*delta[0]
        eta_trial = eta + alpha*delta[1]
        # detect immediate 2-cycle tendency
        if prev2 is not None and np.allclose([xi_trial, eta_trial], prev2, atol=1e-12):
            alpha = 0.25
            xi_trial = xi + alpha*delta[0]; eta_trial = eta + alpha*delta[1]
            if debug: print("2-cycle detected, reducing alpha to", alpha)
        res_old = resG
        res_trial = res_orig(xi_trial, eta_trial)
        # backtracking: require residual decrease
        bt_iter = 0
        while res_trial > res_old and alpha > 1e-4 and bt_iter < 10:
            alpha *= 0.5
            xi_trial = xi + alpha*delta[0]
            eta_trial = eta + alpha*delta[1]
            res_trial = res_orig(xi_trial, eta_trial)
            bt_iter += 1
            if debug: print(f" backtrack alpha={alpha:.4g}, res_trial={res_trial:.6g}, detla={delta}")
        # commit
        prev2 = prev1.copy() if prev1 is not None else None
        prev1 = np.array([xi, eta], dtype=float)
        xi, eta = xi_trial, eta_trial
        # small step -> consider converged
        if np.linalg.norm(delta)*alpha < tol_Local:
            status = SUCCESS
            break
        else:
            status = FAILED  
        if res_trial > 5:
            status = FAILED
            break
    # 4) 恢复 Jacobian 到原坐标尺度
    Nf, dNf_dxi, dNf_deta = shape_functions_8node(xi, eta)
    Jn11 = float(np.dot(dNf_dxi, nodes_norm[:,0]))
    Jn12 = float(np.dot(dNf_deta, nodes_norm[:,0]))
    Jn21 = float(np.dot(dNf_dxi, nodes_norm[:,1]))
    Jn22 = float(np.dot(dNf_deta, nodes_norm[:,1]))
    Jn = np.array([[Jn11, Jn12],[Jn21, Jn22]], dtype=float)
    with Global2Local_buffer.data_lock:
        Global2Local_buffer.plot_Jn[int(global_coord[1]), int(global_coord[0])] = np.linalg.cond(Jn, 2)
    J = np.array([[sx * Jn[0,0], sx * Jn[0,1]],[sy * Jn[1,0], sy * Jn[1,1]]], dtype=float)
    resG_final = res_orig(xi, eta)
    if debug: time.sleep(1)
    return status, float(resG_final), np.array([xi, eta], dtype=float), J
    
        
def read_seeds_info():
    seeds_info = []
    for eid, conn in enumerate(Global2Local_buffer.elements, start=1):
        nodes = np.array([Global2Local_buffer.nodes_coord[Global2Local_buffer.id2idx[nid]] for nid in conn])
        center_global_coord = np.array(nodes[8])
        center_local_coord = np.array([0.0, 0.0])
        quad8_nodes = nodes[:8,:]  # 取前8个节点坐标
        J = compute_J_at(quad8_nodes, 0.0, 0.0)
        seeds_info.append((center_global_coord, 
                          center_local_coord,
                          quad8_nodes, J, eid))
    Global2Local_buffer.seeds_info = seeds_info


def compute_J_at(elem_coords, xi, eta):
    """
    elem_coords: (8,2) array of node coords [x_i,y_i]
    返回 2x2 雅可比矩阵 J evaluated at (xi,eta)
    """
    _, dN_dxi, dN_deta = shape_functions_8node(xi, eta)
    dX_dxi = np.sum(dN_dxi * elem_coords[:,0])
    dX_deta = np.sum(dN_deta * elem_coords[:,0])
    dY_dxi = np.sum(dN_dxi * elem_coords[:,1])
    dY_deta = np.sum(dN_deta * elem_coords[:,1])
    J = np.array([[dX_dxi, dX_deta],
                  [dY_dxi, dY_deta]])
    return J


def build_eie_idx_matrix(Inform, mask):
    """
    Inform: array of shape (N, 3)  每行 = [x, y, elem_id]
    mask_shape: (H, W)   与 ROI 或图像大小一致

    返回:
        eie_idx_matrix: (H, W) int32
    """
    H, W = mask.shape

    # 全为 -1
    eie_idx_matrix = -1 * np.ones((H, W), dtype=np.int32)

    # 填充单元编号
    for x, y, eid in Inform:
        # 注意 Inform 是 x,y，而数组的行列是 [y,x]
        if 0 <= y < H and 0 <= x < W and mask[y, x]:
            eie_idx_matrix[y, x] = int(eid)
    return eie_idx_matrix


if __name__ == "__main__":
    from DIC_load_config import load_mesh_dic_config
    from DIC_read_image import Img_Dataset
    from DIC_create_mesh import create_mesh_elemet
    
    cfg = load_mesh_dic_config("./config.json")
    imgGenDataset = Img_Dataset(cfg)
    
    mask = imgGenDataset._get_roiRegion()
    
    nodes_file = os.path.join(cfg.mesh_dir, "nodes.txt")
    if os.path.exists(nodes_file):
        pass
    else:
        create_mesh_elemet(
            mask=mask,
            mesh_size=cfg.mesh_size,
            simplify_eps=cfg.simplify_roi_boundary_poly,
            output_dir=cfg.mesh_dir
        )
    
    comp_g2l = Comp_global2local(cfg, mask)
    comp_g2l.load_mesh_buffer()
    comp_g2l.solve()
    comp_g2l.save_results()
    comp_g2l.plot_mesh_calcpoints(Global2Local_buffer.plot_validpoints, var_name="mesh_plot_validpoints")
    comp_g2l.plot_mesh_calcpoints(Global2Local_buffer.plot_calcpoints, var_name="mesh_plot_calcpoints")
    cond_Jn = np.log10(1+Global2Local_buffer.plot_Jn)
    comp_g2l.plot_mesh_calcpoints(cond_Jn, var_name="mesh_plot_cond_Jn")