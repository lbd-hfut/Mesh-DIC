import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scipy.io import savemat
import matplotlib.pyplot as plt
import random

from DIC_create_mesh import read_elements, read_nodes
from DIC_shape_function import shape_functions_8node_batch

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_everything(42)

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
    parallel_flag = None
    max_workers = None
    plot_Jn = None
    

class Comp_global2local:
    def __init__(self, config, mask):
        self.mesh_dir = config.mesh_dir
        Global2Local_buffer.mask = mask
        self.load_mesh_buffer()
        self.model = NNModel().to(device)
        result_file = os.path.join(self.mesh_dir, f"global2local_J.npz")
        if os.path.exists(result_file):
            self.load_Global2Local_buffer(result_file)
            print(f"读取全局局部对应点数据：{result_file}")
        else:
            print("求解全局局部对应点数据")
            self.solve()
            self.save_results()
            self.plot_mesh_points(
                Global2Local_buffer.plot_validpoints, 
                var_name="mesh_plot_validpoints"
                )
            self.plot_mesh_points(
                Global2Local_buffer.plot_calcpoints, 
                var_name="mesh_plot_calcpoints"
                )
        
        
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
        
    def solve(self):
        eid_list = [eid for eid in range(1, len(Global2Local_buffer.elements)+1)]
        def worker(elem_id):
            analysis_element(
                model=self.model,
                elem_id=elem_id,
                cut_err=0.1
            )
        for eid in eid_list:
            worker(eid)
    
    def save_results(self):
        output_dir = self.mesh_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path_mat = os.path.join(output_dir, f"global2local_J.mat")
        save_path_npy = os.path.join(output_dir, f"global2local_J.npz")
        # 构建要保存的字典，对应 MATLAB struct
        dic_struct = {
            'plot_calcpoints': Global2Local_buffer.plot_calcpoints,
            'plot_validpoints': Global2Local_buffer.plot_validpoints,
            'plot_global_coords': Global2Local_buffer.plot_global_coords,
            'plot_local_coords': Global2Local_buffer.plot_local_coords,
            'eie_idx_matrix': Global2Local_buffer.threaddiagram,
            'plot_J': Global2Local_buffer.plot_J,
            'cond_Jn': Global2Local_buffer.plot_Jn
        }
        # 使用 savemat 保存，结构体形式
        savemat(save_path_mat, {'global2local_J': dic_struct})
        np.savez(save_path_npy, **dic_struct)
        print(f"Saved MATLAB .mat file: {save_path_mat}")
        print(f"Saved Python .npy file: {save_path_npy}")
        
    def load_Global2Local_buffer(self, result_file):
        # 读取 npz 文件
        data = np.load(result_file, allow_pickle=True)
        # 赋值回 Global2Local_buffer
        Global2Local_buffer.plot_calcpoints = data['plot_calcpoints']
        Global2Local_buffer.plot_validpoints = data['plot_validpoints']
        Global2Local_buffer.plot_global_coords = data['plot_global_coords']
        Global2Local_buffer.plot_local_coords = data['plot_local_coords']
        Global2Local_buffer.threaddiagram = data['eie_idx_matrix']
        Global2Local_buffer.plot_J = data['plot_J']
        Global2Local_buffer.plot_Jn = data['cond_Jn']
        print(f"Loaded Global2Local_buffer from {result_file}")
    
    def plot_mesh_points(self, plot_var=None, var_name="plot_validpoints"):
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
    
    
def analysis_element(
    model=None,
    elem_id=None,
    cut_err = 0.1
):
    model._init_weights()
    ys, xs = np.where(Global2Local_buffer.threaddiagram == elem_id)
    global_points = np.vstack((xs, ys)).T  # (N, 2)  x,y
    conn = Global2Local_buffer.elements[elem_id-1][:8]
    quad8_nodes = np.array(
        [Global2Local_buffer.nodes_coord[Global2Local_buffer.id2idx[nid]] for nid in conn]
        )  # (8, 2)
    
    # ------------------ 准备训练 ------------------
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_bfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=100,
        history_size=50,
        line_search_fn="strong_wolfe"
        )
    num_adam_epochs = 400  # 测试用，可适当增加
    num_bfgs_epochs = 36
    total_epochs = num_adam_epochs + num_bfgs_epochs * 100
    epoch_log = [0]
    
    print(f"Training NN for element {elem_id} with {global_points.shape[0]} points...")
    # ------------------ 使用 Adam 优化器 ------------------
    for epoch in range(num_adam_epochs):
        optimizer_adam.zero_grad()
        mse_loss = model.forward(global_points, quad8_nodes)[0]
        if mse_loss.item() < 1e-4:
            break
        mse_loss.backward()
        optimizer_adam.step()
        if (epoch_log[0]+1) % 200 == 0:
            print(f"Adam: Epoch {epoch_log[0]+1}/{total_epochs}, MSE: {mse_loss.item():.12f}")
        epoch_log[0] = epoch_log[0] + 1
    # ------------------ 使用 LBFGS 优化器 -----------------
    def closure():
        optimizer_bfgs.zero_grad()
        mse_loss, _ = model.forward(global_points, quad8_nodes)
        mse_loss.backward()
        if (epoch_log[0]+1) % 20 == 0:
            print(f"Bfgs: Epoch {epoch_log[0]+1}/{total_epochs}, MSE: {mse_loss.item():.12f}")
        epoch_log[0] = epoch_log[0] + 1
        return mse_loss
    for epoch in range(num_bfgs_epochs):
        mse_loss = optimizer_bfgs.step(closure)
        if mse_loss.item() < 1e-4:
            break

    # ------------------ 预测局部坐标 ------------------
    x_local_pred = model.predict(global_points, quad8_nodes)
    Global2Local_buffer.plot_global_coords[ys, xs] = global_points
    Global2Local_buffer.plot_local_coords[ys, xs] = x_local_pred
    Global2Local_buffer.plot_calcpoints[ys, xs] = True
    J = compute_J_at_batch(quad8_nodes, x_local_pred[:,0], x_local_pred[:,1])
    cond_Jn = compute_cond_batch(J)
    Global2Local_buffer.plot_J[ys, xs] = J
    Global2Local_buffer.plot_Jn[ys, xs] = cond_Jn
    # Reconstruct global coords to compute J and check error
    xi, eta = x_local_pred[:,0], x_local_pred[:,1]
    N, _, _ = shape_functions_8node_batch(xi, eta)
    x_recon_global = np.matmul(N, quad8_nodes)  # (N, 2)
    err_norm_rows = np.linalg.norm(x_recon_global - global_points, axis=1)
    idx_valid = err_norm_rows <= cut_err
    ys_valid = ys[idx_valid]
    xs_valid = xs[idx_valid]
    Global2Local_buffer.plot_validpoints[ys_valid, xs_valid] = True
    print(f"Element {elem_id} done. Valid points: {len(xs_valid)}/{len(xs)}")
        
        
class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.bn1 = nn.BatchNorm1d(2)
        self.fc1 = nn.Linear(2, 64)  # 输入层（全局坐标x, y）
        self.fc2 = nn.Linear(64, 64) # 隐藏层
        self.fc3 = nn.Linear(64, 2)  # 输出层（局部坐标xi, eta）
        self.MSE = nn.MSELoss()  # 均方误差损失函数
        self.double()

    def forward(self, x_global, node_x_coords):
        if not isinstance(x_global, torch.Tensor):
            x_global = torch.tensor(x_global, dtype=torch.float64)
        if x_global.device != device:
            x_global = x_global.to(device)
        if not isinstance(node_x_coords, torch.Tensor):
            node_x_coords = torch.tensor(node_x_coords, dtype=torch.float64)
        if x_global.device != device:
            node_x_coords = node_x_coords.to(device)

        # 先通过归一化层
        x = self.bn1(x_global)
        # 然后通过全连接层和激活函数
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x_local = self.fc3(x)

        N = self.shapef(x_local[:, 0], x_local[:, 1])  # 计算形状函数
        xp_gloabl = torch.matmul(N, node_x_coords)  # 计算预测全局坐标

        # 计算误差
        mse = self.MSE(xp_gloabl, x_global)
        return mse, x_local
        
    def _init_weights(self):
        # Xavier 初始化（适合 tanh）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def shapef(self, xi, eta):
        N = torch.zeros(xi.shape[0], 8, dtype=torch.float64).to(device)
        xi2 = xi * xi
        eta2 = eta * eta
        # ---------- Shape functions N_i ----------
        N[:, 0] = -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta)
        N[:, 1] = -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta)
        N[:, 2] = -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta)
        N[:, 3] = -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta)
        N[:, 4] = 0.5 * (1 - xi2) * (1 - eta)
        N[:, 5] = 0.5 * (1 + xi) * (1 - eta2)
        N[:, 6] = 0.5 * (1 - xi2) * (1 + eta)
        N[:, 7] = 0.5 * (1 - xi) * (1 - eta2)
        return N
    
    def predict(self, x_global, node_x_coords):
        if not isinstance(x_global, torch.Tensor):
            x_global = torch.tensor(x_global, dtype=torch.float64)
        if x_global.device != device:
            x_global = x_global.to(device)
        if not isinstance(node_x_coords, torch.Tensor):
            node_x_coords = torch.tensor(node_x_coords, dtype=torch.float64)
        if x_global.device != device:
            node_x_coords = node_x_coords.to(device)
        with torch.no_grad():
            # 预测局部坐标
            x_local = self.forward(x_global, node_x_coords)[1]
            return x_local.cpu().numpy()


def compute_J_at_batch(elem_coords, xis, etas):
    Np = len(xis)
    # 得到批处理梯度
    _, dN_dxi, dN_deta = shape_functions_8node_batch(xis, etas)

    xs = elem_coords[:, 0]   # (8,)
    ys = elem_coords[:, 1]   # (8,)

    # dX/dxi = sum_i dN_dxi_i * x_i
    dX_dxi  = dN_dxi @ xs     # (N,)
    dX_deta = dN_deta @ xs    # (N,)
    dY_dxi  = dN_dxi @ ys     # (N,)
    dY_deta = dN_deta @ ys    # (N,)

    # 组装成 (N,2,2)
    J = np.stack([
        np.stack([dX_dxi, dX_deta], axis=1),
        np.stack([dY_dxi, dY_deta], axis=1)
    ], axis=1)

    return J

def compute_cond_batch(J):
    # SVD 批处理
    _, S, _ = np.linalg.svd(J)   # S shape = (N, 2)

    # 2-范数条件数 = 最大奇异值 / 最小奇异值
    return S[:, 0] / S[:, 1]


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
    # comp_g2l.solve()
    # comp_g2l.save_results()
    # comp_g2l.plot_mesh_points(Global2Local_buffer.plot_validpoints, var_name="mesh_plot_validpoints")
    # comp_g2l.plot_mesh_points(Global2Local_buffer.plot_calcpoints, var_name="mesh_plot_calcpoints")
