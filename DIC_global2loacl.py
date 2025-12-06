import os
import numpy as np
from threading import RLock, Event
from collections import deque
from tqdm import tqdm
from scipy.io import savemat
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
        if Global2Local_buffer.parallel_flag:
            max_workers = Global2Local_buffer.max_workers
        else:
            max_workers = 1
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_seed = {
                    executor.submit(worker, queues[i], Global2Local_buffer.seeds_info[i]): i
                    for i in range(len(Global2Local_buffer.seeds_info))
                }
                # 等待所有线程完成
                for future in as_completed(future_to_seed):
                    try:
                        future.result()     # 捕获线程内部异常
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
        }
        # 使用 savemat 保存，结构体形式
        savemat(save_path, {'global2local_J': dic_struct})
        print(f"Saved MATLAB .mat file: {save_path}")


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
            ys, xs = np.where(Global2Local_buffer.threaddiagram == num_thread)
            uncal_flag = ~Global2Local_buffer.plot_calcpoints[ys, xs]
            print(f"Thread {num_thread}: 寻找未计算点，剩余 {np.sum(uncal_flag)} 点")
            if np.any(uncal_flag):
                ys_u  = ys[uncal_flag]
                xs_u  = xs[uncal_flag]
                # 计算距离
                dx = xs_u - center_global_coord[0]
                dy = ys_u - center_global_coord[1]
                dist2 = dx * dx + dy * dy
                # 距离最短的点
                idx = np.argmin(dist2)
                x_uncal = xs_u[idx]
                y_uncal = ys_u[idx]
                
                outstate, residual, local_coord, Jcobi = analyzepoint(
                    queue, np.array([x_uncal, y_uncal]), 
                    center_local_coord, 
                    quad8_nodes, J, num_thread,
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
    outstate, residual, local_coord, Jcobi = cal_point_g2L(
        global_coord=global_coord,
        local_coord_init=local_coord_init,
        quad8_nodes=quad8_nodes,
        J_init=J_init)
    # ---------------- 加入队列 ----------------
    if (outstate == SUCCESS and
        residual < cutoff_residual):
        paramvector = (global_coord, local_coord, Jcobi, num_thread)
        queue.append(paramvector)
        Global2Local_buffer.plot_validpoints[y,x] = True
    else:
        # paramvector = (global_coord, local_coord, Jcobi, num_thread)
        # queue.append(paramvector)
        with Global2Local_buffer.data_lock:
            Global2Local_buffer.plot_J[y,x,:,:] = Jcobi[:,:]
            Global2Local_buffer.plot_global_coords[y,x,:] = global_coord[:]
            Global2Local_buffer.plot_local_coords[y,x,:] = local_coord[:]
    # ---------------- 标记已计算 ----------------
    Global2Local_buffer.plot_calcpoints[y,x] = True
    # ---------------- 线程更新进度 ----------------
    if pbar is not None and pbar_lock is not None:
        with pbar_lock:
            pbar.update(1)
    return outstate, residual, local_coord, Jcobi
        
def cal_point_g2L(
    global_coord: np.ndarray,       # shape = (2,) 全局点 (x,y)
    local_coord_init: np.ndarray,   # 初始局部坐标 [xi0, eta0]
    quad8_nodes: np.ndarray,        # Q8 单元节点坐标，shape=(8,2)
    J_init: np.ndarray = None,      # 可选初始Jacobian
    tol_Global: float = 0.05,
    tol_Local: float = 1e-8,
    max_iter: int = 1000,
):
    # 1) 对节点和全局点做归一化
    xmin, xmax = quad8_nodes[:,0].min(), quad8_nodes[:,0].max()
    ymin, ymax = quad8_nodes[:,1].min(), quad8_nodes[:,1].max()
    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    sx = 0.5 * (xmax - xmin)
    sy = 0.5 * (ymax - ymin)
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
    
    # 2) Newton 迭代（归一化空间）
    xi, eta = local_coord_init.copy()
    for iter in range(max_iter):
        # 形函数与一阶导数
        N, dN_dxi, dN_deta = shape_functions_8node(xi, eta)
        # 当前点（归一化空间）的全局映射 x(ξ,η)
        x_m = np.dot(N, nodes_norm[:, 0])
        y_m = np.dot(N, nodes_norm[:, 1])
        # 残差：目标点 - 映射点
        Rn = np.array([point_norm[0] - x_m,
                       point_norm[1] - y_m])
        # 收敛判断
        res_Global = np.linalg.norm([
            global_coord[0] - (np.dot(N, quad8_nodes[:,0])),
            global_coord[1] - (np.dot(N, quad8_nodes[:,1]))
        ])
        if res_Global < tol_Global:
            break
        # 计算雅可比矩阵 J（归一化空间）
        if J_init is not None and iter == 0:
            J  = J_init.copy()
            Jn = np.array([
                [J[0,0]/sx, J[0,1]/sx],
                [J[1,0]/sy, J[1,1]/sy],
            ])
        elif J_init is None and iter == 0:
            pass
        else:
            Jn11 = np.dot(dN_dxi,  nodes_norm[:, 0])   # dx/dxi
            Jn12 = np.dot(dN_deta, nodes_norm[:, 0])   # dx/deta
            Jn21 = np.dot(dN_dxi,  nodes_norm[:, 1])   # dy/dxi
            Jn22 = np.dot(dN_deta, nodes_norm[:, 1])   # dy/deta
            Jn  = np.array([[Jn11, Jn12],
                          [Jn21, Jn22]])
        # Newton 增量 Δ = J^{-1} R
        try:
            delta = np.linalg.solve(Jn , Rn)
        except np.linalg.LinAlgError:
            # 若雅可比奇异，用最小二乘兜底
            delta, *_ = np.linalg.lstsq(Jn, Rn, rcond=None)
        # 更新局部坐标
        xi  += delta[0]
        eta += delta[1]
        # 再次收敛判断
        if np.linalg.norm(delta) < tol_Local:
            break
    # 4) 恢复 Jacobian 到原坐标尺度
    J = np.array([
        [sx * Jn[0,0], sx * Jn[0,1]],
        [sy * Jn[1,0], sy * Jn[1,1]],
    ])
    try:
        delta = np.linalg.solve(Jn , Rn)
        return SUCCESS, res_Global, np.array([xi, eta]), J
    except np.linalg.LinAlgError:
        return FAILED, res_Global, np.array([xi, eta]), J
    
        
def read_seeds_info():
    seeds_info = []
    for eid, conn in enumerate(Global2Local_buffer.elements, start=1):
        nodes = np.array([Global2Local_buffer.nodes_coord[Global2Local_buffer.id2idx[nid]] for nid in conn])
        center_global_coord = np.array(nodes[8])
        center_local_coord = np.array([0.0, 0.0])
        quad8_order = [0, 4, 1, 5, 2, 6, 3, 7]
        quad8_nodes = nodes[quad8_order]
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