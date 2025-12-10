import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from DIC_read_image import BufferManager
from DIC_create_mesh import read_nodes
from DIC_local_icgn import iterativesearch_local
from scipy.io import savemat

class NodeUVInit_buffer:
    nodes_coord = None
    id2idx = None
    subset_r = None
    search_radius = None
    nodes_coord_uv = None
    
class node_uv_init:
    def __init__(self, config):
        self.mesh_dir = config.mesh_dir
        self.parallel = config.parallel
        self.max_workers = config.max_workers
        nodes_file = os.path.join(self.mesh_dir, "nodes.txt")
        NodeUVInit_buffer.nodes_coord, NodeUVInit_buffer.id2idx = read_nodes(nodes_file)
        NodeUVInit_buffer.nodes_coord_uv = np.zeros_like(NodeUVInit_buffer.nodes_coord, dtype=np.float32)
        NodeUVInit_buffer.subset_r = config.subset_r
        NodeUVInit_buffer.search_radius = config.search_radius
        self.seed_points_list = [(i, node.astype(int)) for i, node in enumerate(NodeUVInit_buffer.nodes_coord)]
        x_offsets = np.arange(-config.subset_r, config.subset_r + 1, dtype=np.int32)
        y_offsets = np.arange(-config.subset_r, config.subset_r + 1, dtype=np.int32)
        xv, yv = np.meshgrid(x_offsets, y_offsets)  # shape (S,S)
        NodeUVInit_buffer.X_flat = xv.reshape(-1)   # (subset_area,)
        NodeUVInit_buffer.Y_flat = yv.reshape(-1)
        
        
    # ⭐⭐ 多线程求解所有单元节点 ⭐⭐
    def solve_all_seed_points(self):
        # 多线程求解全部 seed 点（使用 self.seed_points_list）
        def worker(seed):
            idx, seed_xy = seed
            cx, cy = int(seed_xy[0]), int(seed_xy[1])
            flag, defvector, corrcoef = cal_seed_point(idx, cy, cx)
            return (cx, cy, flag, defvector, corrcoef)
        
        seed_points = self.seed_points_list
        n_points = len(seed_points)
        results = []
        if self.parallel:
            max_workers = self.max_workers
        else:
            max_workers = 1
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_seed = {
                executor.submit(worker, seed): seed
                for seed in seed_points
            }
            with tqdm(total=n_points, desc="Solving seed points", unit="pt") as pbar:
                for future in as_completed(future_to_seed):
                    results.append(future.result())
                    pbar.update(1)
        return results
    
    def plot_init_uv(self):
        os.makedirs(self.mesh_dir, exist_ok=True)
        uv_coords = NodeUVInit_buffer.nodes_coord_uv
        plt.figure(figsize=(20, 20))
        plt.quiver(
            NodeUVInit_buffer.nodes_coord[:, 0],
            NodeUVInit_buffer.nodes_coord[:, 1],
            uv_coords[:, 0],
            uv_coords[:, 1],
            angles='xy', scale_units='xy', scale=0.1, color='r'
        )
        plt.gca().invert_yaxis()
        plt.title('Initial Displacement Vectors')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.axis('equal')
        plt.grid()
        plt.savefig(
            os.path.join(
                self.mesh_dir, 
                'initial_displacement_vectors.png')
            )
        plt.close()
        

def cal_seed_point(
    idx:int,
    cy: int, cx: int, 
    max_iter: int = 200,
    cutoff_diffnorm: float = 1e-5,
    lambda_reg: float = 1e-3
):
    mask_pad = BufferManager.mask_pad
    
    v0, u0 = coarse_search_int(idx, cy, cx)
    defvector_init = np.zeros(12)
    defvector_init[0], defvector_init[1] = u0, v0
    
    py = cy + NodeUVInit_buffer.subset_r
    px = cx + NodeUVInit_buffer.subset_r
    y0, y1 = py - NodeUVInit_buffer.subset_r, py + NodeUVInit_buffer.subset_r + 1
    x0, x1 = px - NodeUVInit_buffer.subset_r, px + NodeUVInit_buffer.subset_r + 1
    mask_seg = mask_pad[y0:y1, x0:x1].reshape(-1)
    valid_idx = np.nonzero(mask_seg)[0]
    dx, dy = NodeUVInit_buffer.X_flat[valid_idx], NodeUVInit_buffer.Y_flat[valid_idx]
    
    flag, defvector, corrcoef = iterativesearch_local(
        defvector_init=defvector_init, 
        xc=cx, yc=cy, dx=dx, dy=dy,
        max_iter=max_iter,
        cutoff_diffnorm=cutoff_diffnorm,
        lambda_reg=lambda_reg
        )
    NodeUVInit_buffer.nodes_coord_uv[idx, 0] = defvector[0]
    NodeUVInit_buffer.nodes_coord_uv[idx, 1] = defvector[1]
    return flag, defvector, corrcoef

def coarse_search_int(idx, cy, cx):
    mask, subset_r, search_radius = \
        BufferManager.mask, \
        50, NodeUVInit_buffer.search_radius
        
    H, W = BufferManager.defImg.shape
    y0 = cy - subset_r; y1 = cy + subset_r + 1
    x0 = cx - subset_r; x1 = cx + subset_r + 1   
    # clip to image
    y0c = max(0, y0); y1c = min(H, y1)
    x0c = max(0, x0); x1c = min(W, x1)
    h = y1c - y0c; w = x1c - x0c
    ref_patch = BufferManager.refImg[y0c:y1c, x0c:x1c].astype(np.float32)
    mask_patch = mask[y0c:y1c, x0c:x1c].astype(np.float32)
    
    y_min = int(max(0, y0c - search_radius))
    y_max = int(min(H, y1c + search_radius + 1))
    x_min = int(max(0, x0c - search_radius))
    x_max = int(min(W, x1c + search_radius + 1))
    
    search_def = BufferManager.defImg[y_min:y_max, x_min:x_max].astype(np.float32)
    res = cv2.matchTemplate(
        search_def, ref_patch,
        cv2.TM_CCOEFF_NORMED,
        mask=mask_patch
    )
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    best_y = y_min + max_loc[1]
    best_x = x_min + max_loc[0]
    
    dy_int = best_y - y0c
    dx_int = best_x - x0c
    return dy_int, dx_int




if __name__ == "__main__":
    from DIC_load_config import load_mesh_dic_config
    from DIC_read_image import Img_Dataset, collate_fn
    from DIC_create_mesh import create_mesh_elemet
    import torch
    
    cfg = load_mesh_dic_config("./config.json")
    imgGenDataset = Img_Dataset(cfg)
    
    imgGenDataset._get_QK_QKdx_QKdxx()
    mask = imgGenDataset._get_roiRegion()
    imgGenDataset._get_refImg()
    imgGenDataset._get_image_gradient()
    
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
    
    node_init_solver = node_uv_init(cfg)
    
    img_loader = torch.utils.data.DataLoader(
        imgGenDataset, batch_size=1, 
        shuffle=False, collate_fn=collate_fn)
    
    for idx, DimageL in enumerate(img_loader):
        results = node_init_solver.solve_all_seed_points()
        node_init_solver.plot_init_uv()
        savemat(
            os.path.join(
                cfg.mesh_dir, 
                f'node_uv_init_{idx+1:04d}.mat'),
            {'node_uv_init': {
                'nodes_coord_uv': NodeUVInit_buffer.nodes_coord_uv,
                'nodes_coord': NodeUVInit_buffer.nodes_coord
            }}
        )
        
    
    