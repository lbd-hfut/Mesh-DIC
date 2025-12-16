import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat
import csv
import torch.nn.functional as F

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
    scale=None
    
class Q8nn(nn.Module):
    def __init__(self):
        super(Q8nn, self).__init__()
        shape = NodeUVInit_buffer.nodes_coord_uv.shape
        # 随机初始化在 [-1, 1] 之间，dtype 为 float64
        rand_tensor = (2 * torch.rand(shape, dtype=torch.float64) - 1)
        self.params = nn.Parameter(rand_tensor)
    def forward(self):
        return self.params
        
    
class Q8Model:
    def __init__(self):
        # super().__init__()
        self.dnn = Q8nn()
        self.dnn = self.dnn.double()
        self.dnn = self.dnn.to(device)
        H,L = BufferManager.refImg.shape
        y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L)
        IX, IY = np.meshgrid(x, y)
        IX = torch.tensor(IX, dtype=torch.float32)
        IY = torch.tensor(IY, dtype=torch.float32)
        self.XY = torch.stack((IX, IY), dim=2).unsqueeze(0).to(device)
        
        self.elements = torch.tensor(
            Global2Local_buffer.elements, dtype=torch.long, device=device)
        self.nodes_coord = torch.tensor(
            Global2Local_buffer.nodes_coord, dtype=torch.float64, device=device)
        
        min_val = self.nodes_coord.min(dim=0, keepdim=True).values
        max_val = self.nodes_coord.max(dim=0, keepdim=True).values
        self.nodes_coord_norm = 2 * (self.nodes_coord - min_val) / (max_val - min_val) - 1
        
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
            BufferManager.refImg, dtype=torch.float64, device=device) * 255
        self.scale = torch.tensor(Mesh_DIC_buffer.scale)
        self.scale = self.scale.to(device)
        self.defImg = torch.tensor(
            BufferManager.defImg, dtype=torch.float64, device=device) * 255
        self.epoch = 0
        self.freq = 10
    
    def set_optim(self):
        self.optimizer_adam = optim.Adam(self.dnn.parameters(), lr=1e-3)
        self.optimizer_bfgs = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-3,      # 当梯度范数小于此值时提前停止
            tolerance_change=1e-4,    # 当参数变化小于此值时提前停止tolerance_grad=1e-6,     
            )
        
    def Q8_uv(self):
        U = self.dnn()
        nodes_u = U[:,0]
        nodes_v = U[:,1]
        u_global = torch.zeros_like(self.refImg).to(device)  # 每个像素点的 x 位移
        v_global = torch.zeros_like(self.refImg).to(device)  # 每个像素点的 y 位移
        for i, ele in enumerate(self.elements, start=1):
            conn = ele[:8]
            conn_list = conn.tolist()
            ele_node_u = nodes_u[torch.tensor([self.id2idx[nid] for nid in conn_list], device=device)]
            ele_node_v = nodes_v[torch.tensor([self.id2idx[nid] for nid in conn_list], device=device)]
            row_list, col_list = torch.where(self.threaddiagram == i)
            valid_flag = self.plot_validpoints[row_list, col_list]
            if torch.sum(valid_flag) == 0:
                continue
            row_list = row_list[valid_flag]
            col_list = col_list[valid_flag]
            # 获取像素坐标及对应的单元内局部坐标
            local_coords = self.plot_local_coords[row_list, col_list]
            xi_list, eta_list = local_coords[:,0], local_coords[:,1]
            N_list, _, _ = self.shape_functions_8node_batch(xi_list, eta_list)
            u_px_list = torch.einsum('pk,k->p', N_list, ele_node_u)
            v_px_list = torch.einsum('pk,k->p', N_list, ele_node_v)
            u_global[row_list, col_list] = u_px_list
            v_global[row_list, col_list] = v_px_list
        u_global = u_global * self.scale[0] + self.scale[2]
        v_global = v_global * self.scale[1] + self.scale[3]
        return u_global, v_global
    
    def loss_fn(self):
        self.optimizer_bfgs.zero_grad()
        self.optimizer_adam.zero_grad()
        U, V = self.Q8_uv()
        # Interpolate a new deformed image
        target_height = self.defImg.shape[0]
        target_width  = self.defImg.shape[1]
        u = U/(target_width/2); v = V/(target_height/2)
        uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
        X_new = self.XY + uv_displacement
        # 插值新的散斑图
        new_Iref = F.grid_sample(
            self.defImg.view(1, 1, target_height, target_width), 
            X_new.view(1, target_height, target_width, 2), 
            mode='bilinear', align_corners=True
            )
        # 计算两张图的相关数
        abs_error = (new_Iref[0, 0] - self.refImg)**2 * self.plot_validpoints
        loss = torch.sum(abs_error) / torch.sum(self.plot_validpoints[:,:])
        loss.backward()
        mae = torch.abs(new_Iref[0, 0] - self.refImg) * self.plot_validpoints
        mae = torch.sum(mae) / torch.sum(self.plot_validpoints[:,:])
        self.epoch = self.epoch+1
        if self.epoch % self.freq == 1:   
            print(f'Epoch [{self.epoch}], Loss: {mae.item():.4f}')
        return loss
    
    def predict(self):
        with torch.no_grad():
            U, V = self.Q8_uv()
            # U = self.dnn(self.nodes_coord_norm)
            # node_uv = U.cpu().detach().numpy()
            # Mesh_DIC_buffer.plot_u, Mesh_DIC_buffer.plot_v, \
            # Mesh_DIC_buffer.plot_ex, Mesh_DIC_buffer.plot_ey, \
            #     Mesh_DIC_buffer.plot_rxy = interp_uv_strain(node_uv=node_uv)
        u = U.cpu().detach().numpy()
        v = V.cpu().detach().numpy()
        Mesh_DIC_buffer.plot_u, Mesh_DIC_buffer.plot_v = u, v
    
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
        Q8nn = Q8Model()
        Q8nn.set_optim()
        num_adam_epochs = 0  # 测试用，可适当增加
        num_bfgs_epochs = 1
        # ------------------ 使用 Adam 优化器 ------------------
        print(f"---------- solve No.{idx} defImg ----------")
        for iter in range(num_adam_epochs):
            loss = Q8nn.loss_fn()
            Q8nn.optimizer_adam.step()
        # ------------------ 使用 LBFGS 优化器 -----------------
        for iter in range(num_bfgs_epochs):
            Q8nn.optimizer_bfgs.step(Q8nn.loss_fn)
        # 输出预测结果
        Q8nn.predict()
        # if self.config.smooth_flag:
        #     Mesh_DIC_buffer.plot_u, Mesh_DIC_buffer.plot_v = \
        #         DIC_smooth_Displacement(
        #             Mesh_DIC_buffer.plot_u, Mesh_DIC_buffer.plot_v,
        #             Global2Local_buffer.plot_validpoints, self.config.smooth_sigma)
        
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
        csv_file = self.config.input_dir+f'Q8DIC/scale_information/SCALE.csv'
        SCALE_list = []
        with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                converted_row = []
                for element in row:
                    converted_element = float(element)
                    converted_row.append(converted_element)
                SCALE_list.append(converted_row)
                
        for idx, _ in enumerate(self.img_loader):
            Mesh_DIC_buffer.scale=SCALE_list[idx]
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
    


