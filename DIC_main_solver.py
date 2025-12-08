import os
import torch
import numpy as np
from scipy.sparse import csr_matrix

from DIC_load_config import load_mesh_dic_config
from DIC_read_image import Img_Dataset, BufferManager, collate_fn
from DIC_create_mesh import create_mesh_elemet
from DIC_nodeuv_init import node_uv_init, NodeUVInit_buffer
from DIC_g2l_DL import Global2Local_buffer, Comp_global2local
from DIC_calc_Hb import StiffnessMatrixBuffer, assemble_global_stiffness_Q8, \
    assemble_global_residual_Q8, global_ICGN

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
        FEM_file = os.path.join(self.config.mesh_dir, f"FEM_system.npz")
        if os.path.exists(FEM_file):
            npzfile = np.load(FEM_file)
            StiffnessMatrixBuffer.A_global = \
                csr_matrix(
                    (npzfile['A_data'], npzfile['A_indices'], npzfile['A_indptr']), 
                    shape=npzfile['A_shape']
                    )
            StiffnessMatrixBuffer.DN_list_elem = npzfile['DN']
            StiffnessMatrixBuffer.Nmat_list_elem = npzfile['Nmat']
        else:
            assemble_global_stiffness_Q8(alpha=self.alpha, output_dir=self.config.mesh_dir)
        
    def solve_each(self, idx):
        # 初始化的网格节点位移保存在 NodeUVInit_buffer.nodes_coord_uv 中
        self.node_init_solver.solve_all_seed_points()
        nodes_uv, norm_of_W_list = global_ICGN(
            alpha=self.alpha, tol=self.config.cutoff_diffnorm, maxIter=self.config.max_iterations
            )
        插值全场位移场和导数计算位移以及保存
        
    def solve(self):
        for idx, _ in enumerate(self.img_loader):
            self.solve_each(idx)


