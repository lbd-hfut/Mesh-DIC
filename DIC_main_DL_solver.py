import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat

from DIC_load_config import load_mesh_dic_config
from DIC_read_image import Img_Dataset, BufferManager, collate_fn
from DIC_create_mesh import create_mesh_elemet
from DIC_nodeuv_init import node_uv_init, NodeUVInit_buffer
from DIC_g2l_DL import Global2Local_buffer, Comp_global2local, seed_everything
from DIC_result_plot import visualize_imshow, visualize_contourf
from DIC_calc_Hb import interp_uv_strain, assemble_global_stiffness_Q8, \
     global_ICGN
from DIC_post_processing import DIC_Strain_from_Displacement, DIC_smooth_Displacement

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_everything(42)
     
class Mesh_DIC_buffer:
    plot_u = None
    plot_v = None
    plot_ex = None
    plot_ey = None
    plot_rxy = None
    
class Q8Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.elements = torch.tensor(
            Global2Local_buffer.elements, dtype=torch.long, device=device)
        
        self.nodes_coord = torch.tensor(
            Global2Local_buffer.nodes_coord, dtype=torch.float64, device=device)
        
        self.id2idx = Global2Local_buffer.id2idx
        
        self.threaddiagram = torch.tensor(
            Global2Local_buffer.threaddiagram, dtype=torch.long, device=device)
        
        self.plot_validpoints = torch.tensor(
            Global2Local_buffer.plot_validpoints, dtype=torch.bool, device=device)
        
        self.plot_global_coords = torch.tensor(
            Global2Local_buffer.plot_global_coords, dtype=torch.float64, device=device)
        
        self.plot_local_coords = torch.tensor(
            Global2Local_buffer.plot_local_coords, dtype=torch.float64, device=device)
        
        self.refImg = torch.tensor(
            BufferManager.refImg, dtype=torch.float64, device=device)
        
        nodes_uv = torch.tensor(
            NodeUVInit_buffer.nodes_coord_uv, dtype=torch.float64, device=device
        )
        
        self.nodes_u = nn.Parameter(nodes_uv[:, 0])
        self.nodes_v = nn.Parameter(nodes_uv[:, 1])
        # self.nodes_u = nn.Parameter(torch.empty(nodes_uv.shape[0], dtype=torch.float64, device=device).uniform_(-1.0, 1.0))
        # self.nodes_v = nn.Parameter(torch.empty(nodes_uv.shape[0], dtype=torch.float64, device=device).uniform_(-1.0, 1.0))
        
        self.defImg = torch.tensor(
            BufferManager.defImg, dtype=torch.float64, device=device)
        
        H, W = BufferManager.refImg.shape
        self.QKBQKT_def = torch.zeros(
            (H, W, 6, 6), dtype=torch.float64, device=device)
        for (y, x), mat_np in BufferManager.QKBQKT_def.items():
            self.QKBQKT_def[y, x] = torch.tensor(
                mat_np, dtype=torch.float64, device=device)
        
    def forward(self):
        loss = 0.0
        for i, ele in enumerate(self.elements, start=1):
            conn = ele[:8]
            conn_list = conn.tolist()
            ele_node_u = self.nodes_u[torch.tensor([self.id2idx[nid] for nid in conn_list], device=self.nodes_u.device)]
            ele_node_v = self.nodes_v[torch.tensor([self.id2idx[nid] for nid in conn_list], device=self.nodes_u.device)]
            row_list, col_list = torch.where(self.threaddiagram == i)
            valid_flag = self.plot_validpoints[row_list, col_list]
            if torch.sum(valid_flag) == 0:
                continue
            row_list = row_list[valid_flag]
            col_list = col_list[valid_flag]
            # 获取像素坐标及对应的单元内局部坐标
            global_coords = self.plot_global_coords[row_list, col_list] 
            x_px_list, y_px_list = global_coords[:,0], global_coords[:,1]
            local_coords = self.plot_local_coords[row_list, col_list]
            xi_list, eta_list = local_coords[:,0], local_coords[:,1]
            N_list, _, _ = self.shape_functions_8node_batch(xi_list, eta_list)
            u_px_list = torch.einsum('pk,k->p', N_list, ele_node_u)
            v_px_list = torch.einsum('pk,k->p', N_list, ele_node_v)
            r_img  = (self.refImg[row_list, col_list] - self.interpqbs(
                        x_px_list + u_px_list, y_px_list + v_px_list))
            r_img = torch.mean(r_img**2)
            loss = loss + r_img
        return loss
    
    def predict(self):
        if self.nodes_u.device.type != "cpu":
            nodes_u = self.nodes_u.detach().cpu()
            nodes_v = self.nodes_v.detach().cpu()
        else:
            nodes_u = self.nodes_u.detach()
            nodes_v = self.nodes_v.detach()
        nodes_u = nodes_u.numpy()
        nodes_v = nodes_v.numpy()
        nodes_uv = np.concatenate([nodes_u[:, None], nodes_v[:, None]], axis=1)
        Mesh_DIC_buffer.plot_u, Mesh_DIC_buffer.plot_v, \
            Mesh_DIC_buffer.plot_ex, Mesh_DIC_buffer.plot_ey, \
                Mesh_DIC_buffer.plot_rxy = interp_uv_strain(nodes_uv)
        
    def interpqbs(self, xs, ys, device=None):
        if device is None:
            device = xs.device
        # 1. floor 坐标 (N,)
        xs_floor = torch.floor(xs).long()
        ys_floor = torch.floor(ys).long()
        num_pts = xs.shape[0]
        # (N,2) → 注意顺序仍然为 (y, x)
        coords = torch.stack([ys_floor, xs_floor], dim=1)
        # 2. 取出 QK_B_QKT_arr (N,6,6)
        Q_arr = self.QKBQKT_def[coords[:, 0], coords[:, 1]]
        # 3. 构造 (N,6) x_powers, y_powers
        xd = xs - xs_floor.float()
        yd = ys - ys_floor.float()
        x_powers = torch.stack([xd ** i for i in range(6)], dim=1)  # (N,6)
        y_powers = torch.stack([yd ** i for i in range(6)], dim=1)  # (N,6)
        # 4. tmp = y_vec @ M → (N,6)
        tmp = torch.einsum("ni,nij->nj", y_powers, Q_arr)
        # 5. 再与 x_vec 做内积 → (N,)
        values = torch.einsum("ni,ni->n", tmp, x_powers)
        return values
    
    def shape_functions_8node_batch(self, xi, eta):
        # 确保 torch 类型
        device = xi.device
        dtype = xi.dtype
        Npts = xi.shape[0]
        # 分配输出
        N = torch.zeros((Npts, 8), dtype=dtype, device=device)
        dN_dxi = torch.zeros((Npts, 8), dtype=dtype, device=device)
        dN_deta = torch.zeros((Npts, 8), dtype=dtype, device=device)
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
    

class Mesh_DIC_Solver:
    def __init__(self, config_path):
        self.config = load_mesh_dic_config(config_path)
        self.imgGenDataset = Img_Dataset(self.config)
        self.img_loader = torch.utils.data.DataLoader(
            self.imgGenDataset, batch_size=1, 
            shuffle=False, collate_fn=collate_fn)
        nodes_file = os.path.join(self.config.mesh_dir, "nodes.txt")
        elements_file= os.path.join(self.config.mesh_dir, "elements.txt")
        Inform_file = os.path.join(self.config.mesh_dir, "Inform.npy")
        if os.path.exists(nodes_file) and os.path.exists(elements_file) and os.path.exists(Inform_file):
            pass
        else:
            create_mesh_elemet(
                mask=BufferManager.mask,
                mesh_size=self.config.mesh_size,
                simplify_eps=self.config.simplify_roi_boundary_poly,
                output_dir=self.config.mesh_dir
            )
        self.node_init_solver = node_uv_init(self.config)
        self.comp_g2l = Comp_global2local(self.config, BufferManager.mask)
        
    def solve_each(self, idx):
        # 初始化的网格节点位移保存在 NodeUVInit_buffer.nodes_coord_uv 中
        self.node_init_solver.solve_all_seed_points()
        Q8nn = Q8Model()
        optimizer_adam = optim.Adam(Q8nn.parameters(), lr=1e-3)
        optimizer_bfgs = torch.optim.LBFGS(
            Q8nn.parameters(),
            lr=1.0,
            max_iter=100,
            history_size=50,
            line_search_fn="strong_wolfe"
            )
        num_adam_epochs = 100  # 测试用，可适当增加
        num_bfgs_epochs = 1
        total_epochs = num_adam_epochs + num_bfgs_epochs * 100
        epoch_log = [0]
        # ------------------ 使用 Adam 优化器 ------------------
        print(f"---------- solve No.{idx} defImg ----------")
        for epoch in range(num_adam_epochs):
            optimizer_adam.zero_grad()
            mse_loss = Q8nn.forward()
            mse_loss.backward()
            optimizer_adam.step()
            if (epoch_log[0]+1) % (total_epochs//10) == 0:
                print(f"Adam: Epoch {epoch_log[0]+1}/{total_epochs}, MSE: {mse_loss.item():.12f}")
            epoch_log[0] = epoch_log[0] + 1
        # ------------------ 使用 LBFGS 优化器 -----------------
        def closure():
            optimizer_bfgs.zero_grad()
            mse_loss = Q8nn.forward()
            mse_loss.backward()
            if (epoch_log[0]+1) % (total_epochs//10) == 0:
                print(f"Bfgs: Epoch {epoch_log[0]+1}/{total_epochs}, MSE: {mse_loss.item():.12f}")
            epoch_log[0] = epoch_log[0] + 1
            return mse_loss
        for epoch in range(num_bfgs_epochs):
            mse_loss = optimizer_bfgs.step(closure)
        # 输出预测结果
        Q8nn.predict()
        if self.config.smooth_flag:
            Mesh_DIC_buffer.plot_u, Mesh_DIC_buffer.plot_v = \
                DIC_smooth_Displacement(
                    Mesh_DIC_buffer.plot_u, Mesh_DIC_buffer.plot_v,
                    Global2Local_buffer.plot_validpoints, self.config.smooth_sigma)
        
        if self.config.strain_calculate_flag:
            Mesh_DIC_buffer.plot_ex, Mesh_DIC_buffer.plot_ey, Mesh_DIC_buffer.plot_rxy = \
                DIC_Strain_from_Displacement(
                    Mesh_DIC_buffer.plot_u, Mesh_DIC_buffer.plot_v,
                    Global2Local_buffer.plot_validpoints, step=1,
                    SmoothLen=self.config.strain_window_half_size
                )
        self.save_result(idx)
        visualize_imshow(
            idx, Mesh_DIC_buffer, 
            Global2Local_buffer, 
            self.config.output_dir
        )
        visualize_contourf(
            idx, Mesh_DIC_buffer, 
            Global2Local_buffer, 
            self.config.output_dir
        )
        
    def solve(self):
        for idx, _ in enumerate(self.img_loader):
            self.solve_each(idx)
            
    def save_result(self, idx):
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, f"Mesh_DIC_{idx+1:03d}.mat")
        # 构建要保存的字典，对应 MATLAB struct
        dic_struct = {
            'mesh_size': self.config.mesh_size,
            'plot_calcpoints': Global2Local_buffer.plot_calcpoints,
            'plot_validpoints': Global2Local_buffer.plot_validpoints,
            'plot_u': Mesh_DIC_buffer.plot_u,
            'plot_v': Mesh_DIC_buffer.plot_v,
            'plot_ex': Mesh_DIC_buffer.plot_ex,
            'plot_ey': Mesh_DIC_buffer.plot_ey,
            'plot_rxy': Mesh_DIC_buffer.plot_rxy,
        }
        savemat(save_path, {'DIC_result': dic_struct})
        print(f"Saved MATLAB .mat file: {save_path}")
            
if __name__ == "__main__":
    config_path = "./config.json"
    DIC_solver = Mesh_DIC_Solver(config_path=config_path)
    DIC_solver.solve()
    


