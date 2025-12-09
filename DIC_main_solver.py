import os
import torch
import numpy as np
from scipy.io import savemat

from DIC_load_config import load_mesh_dic_config
from DIC_read_image import Img_Dataset, BufferManager, collate_fn
from DIC_create_mesh import create_mesh_elemet
from DIC_nodeuv_init import node_uv_init, NodeUVInit_buffer
from DIC_g2l_DL import Global2Local_buffer, Comp_global2local
from DIC_result_plot import visualize_imshow, visualize_contourf
from DIC_calc_Hb import interp_uv_strain, assemble_global_stiffness_Q8, \
     global_ICGN
     
class Mesh_DIC_buffer:
    plot_u = None
    plot_v = None
    plot_ex = None
    plot_ey = None
    plot_rxy = None

class Mesh_DIC_Solver:
    def __init__(self, config_path):
        self.alpha = 0
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
        # 计算组装刚度矩阵在StiffnessMatrixBuffer.A_global
        assemble_global_stiffness_Q8(alpha=self.alpha, output_dir=self.config.mesh_dir)
        
    def solve_each(self, idx):
        # 初始化的网格节点位移保存在 NodeUVInit_buffer.nodes_coord_uv 中
        self.node_init_solver.solve_all_seed_points()
        nodes_uv, norm_of_W_list = global_ICGN(
            alpha=self.alpha, tol=self.config.cutoff_diffnorm, maxIter=self.config.max_iterations
            )
        # 插值全场位移场和导数计算位移以及保存
        Mesh_DIC_buffer.plot_u, Mesh_DIC_buffer.plot_v, \
            Mesh_DIC_buffer.plot_ex, Mesh_DIC_buffer.plot_ey, \
                Mesh_DIC_buffer.plot_rxy = interp_uv_strain(nodes_uv)
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
    


