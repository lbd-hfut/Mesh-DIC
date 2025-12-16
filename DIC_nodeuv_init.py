import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import LinearNDInterpolator
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from scipy.io import loadmat

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
    nodes_uv_flage = None
    
class node_uv_init:
    def __init__(self, config):
        self.mesh_dir = config.mesh_dir
        self.parallel = config.parallel
        self.config = config
        self.max_workers = config.max_workers
        nodes_file = os.path.join(self.mesh_dir, "nodes.txt")
        NodeUVInit_buffer.nodes_coord, NodeUVInit_buffer.id2idx = read_nodes(nodes_file)
        NodeUVInit_buffer.nodes_coord_uv = np.zeros_like(NodeUVInit_buffer.nodes_coord, dtype=np.float32)
        NodeUVInit_buffer.nodes_uv_flage = np.ones(NodeUVInit_buffer.nodes_coord.shape[0], dtype=np.bool)
        NodeUVInit_buffer.subset_r = config.subset_r
        NodeUVInit_buffer.search_radius = config.search_radius
        self.seed_points_list = [(i, node.astype(int)) for i, node in enumerate(NodeUVInit_buffer.nodes_coord)]
        x_offsets = np.arange(-config.subset_r, config.subset_r + 1, dtype=np.int32)
        y_offsets = np.arange(-config.subset_r, config.subset_r + 1, dtype=np.int32)
        xv, yv = np.meshgrid(x_offsets, y_offsets)  # shape (S,S)
        NodeUVInit_buffer.X_flat = xv.reshape(-1)   # (subset_area,)
        NodeUVInit_buffer.Y_flat = yv.reshape(-1)
        
    def load_uv_seed(self):
        mat_file = next(Path(self.config.input_dir).glob("*.mat"), None)
        if mat_file is None:
            return
        else:
            data = loadmat(mat_file)
            u, v = data['u'], data['v']
            for seed in self.seed_points_list:
                idx, seed_xy = seed
                cx, cy = int(seed_xy[0]), int(seed_xy[1])
                if BufferManager.mask[cy,cx]:
                    pass
                else:
                    r = 20
                    py = cy + r
                    px = cx + r
                    y0, y1 = py - r, py + r + 1
                    x0, x1 = px - r, px + r + 1
                    mask_seg = BufferManager.mask_pad[y0:y1, x0:x1]
                    mask_flat = mask_seg.reshape(-1)
                    valid_idx = np.nonzero(mask_flat)[0]
                    if len(valid_idx) == 0:
                        print("subset 区域内没有任何有效 mask 点！")
                        NodeUVInit_buffer.nodes_uv_flage[idx] = 0
                    else:
                        H, W = mask_seg.shape   # 都是 (2r+1)
                        yy, xx = np.indices((H, W))     # (H,W) 网格
                        xx_f = xx.reshape(-1)
                        yy_f = yy.reshape(-1)
                        dist2 = (xx_f[valid_idx] - r)**2 + (yy_f[valid_idx] - r)**2
                        min_idx = valid_idx[np.argmin(dist2)]
                        uy = yy_f[min_idx]
                        ux = xx_f[min_idx]
                        cy = y0 + uy
                        cx = x0 + ux
                NodeUVInit_buffer.nodes_coord_uv[idx, 0] = u[cy, cx]
                NodeUVInit_buffer.nodes_coord_uv[idx, 1] = v[cy, cx]
                NodeUVInit_buffer.nodes_uv_flage[idx] = 1
        
    # ⭐⭐ 多线程求解所有单元节点 ⭐⭐
    def solve_all_seed_points(self):
        # 多线程求解全部 seed 点（使用 self.seed_points_list）
        def worker(seed):
            idx, seed_xy = seed
            cx, cy = int(seed_xy[0]), int(seed_xy[1])
            if BufferManager.mask[cy,cx]:
                pass
            else:
                r = 20
                py = cy + r
                px = cx + r
                y0, y1 = py - r, py + r + 1
                x0, x1 = px - r, px + r + 1
                mask_seg = BufferManager.mask_pad[y0:y1, x0:x1]
                mask_flat = mask_seg.reshape(-1)
                valid_idx = np.nonzero(mask_flat)[0]
                if len(valid_idx) == 0:
                    print("subset 区域内没有任何有效 mask 点！")
                else:
                    H, W = mask_seg.shape   # 都是 (2r+1)
                    yy, xx = np.indices((H, W))     # (H,W) 网格
                    xx_f = xx.reshape(-1)
                    yy_f = yy.reshape(-1)
                    dist2 = (xx_f[valid_idx] - r)**2 + (yy_f[valid_idx] - r)**2
                    min_idx = valid_idx[np.argmin(dist2)]
                    uy = yy_f[min_idx]
                    ux = xx_f[min_idx]
                    cy = y0 + uy
                    cx = x0 + ux
                
            flag, defvector, corrcoef = cal_seed_point(idx, cy, cx)
            return (cx, cy, flag, defvector, corrcoef)
        
        seed_points = self.seed_points_list
        n_points = len(seed_points)
        with tqdm(total=n_points, desc="Solving seed points", unit="pt") as pbar:
            for seed in seed_points:
                worker(seed)
                pbar.update(1)    
        if not (NodeUVInit_buffer.nodes_uv_flage == 1).all():
            print("interp failure nodes uv")
            self.fail_seed_uv()
        uv = NodeUVInit_buffer.nodes_coord_uv
        max_u = np.max(np.abs(uv[:, 0]))
        max_v = np.max(np.abs(uv[:, 1]))
        print(f"Max |u|: {max_u:.6f}, Max |v|: {max_v:.6f},  Overall Max |uv|: {max(max_u, max_v):.6f}")

    
    def fail_seed_uv(self):
        nodes_coord_uv = NodeUVInit_buffer.nodes_coord_uv
        nodes_uv_flag = NodeUVInit_buffer.nodes_uv_flage
        node_xy = NodeUVInit_buffer.nodes_coord
        # 1) 第一次插值（LinearNDInterpolator）
        known_mask = (nodes_uv_flag == 1)
        unknown_mask = (nodes_uv_flag == 0)
        interp_u = LinearNDInterpolator(node_xy[known_mask], nodes_coord_uv[known_mask,0])
        interp_v = LinearNDInterpolator(node_xy[known_mask], nodes_coord_uv[known_mask,1])
        nodes_coord_uv[unknown_mask,0] = interp_u(node_xy[unknown_mask])
        nodes_coord_uv[unknown_mask,1] = interp_v(node_xy[unknown_mask])
        # 2) 检查是否存在 NaN
        nan_mask = np.isnan(nodes_coord_uv[:, 0]) | np.isnan(nodes_coord_uv[:, 1])
        if nan_mask.any():
            print(f"检测到 {nan_mask.sum()} 个 NaN 点，使用最近邻补全...")
            # 3) 最近邻插值修补 NaN
            valid_mask = ~nan_mask
            nbrs = NearestNeighbors(n_neighbors=1).fit(node_xy[valid_mask])
            _, idx_near = nbrs.kneighbors(node_xy[nan_mask])
            nearest_idx = idx_near[:, 0]
            nodes_coord_uv[nan_mask] = nodes_coord_uv[valid_mask][nearest_idx]
        NodeUVInit_buffer.nodes_coord_uv = nodes_coord_uv
        
            
    
    def plot_init_uv(self,idx):
        uv_coords = NodeUVInit_buffer.nodes_coord_uv
        node_xy   = NodeUVInit_buffer.nodes_coord
        flags     = NodeUVInit_buffer.nodes_uv_flage
        plt.figure(figsize=(20, 20))
        mask_red = (flags == 1)
        plt.quiver(
            node_xy[mask_red, 0], node_xy[mask_red, 1],
            uv_coords[mask_red, 0], uv_coords[mask_red, 1],
            angles='xy', scale_units='xy', scale=0.1, color='r',
            label="flag = 1"
        )
        mask_blue = flags == 0
        plt.quiver(
            node_xy[mask_blue, 0], node_xy[mask_blue, 1],
            uv_coords[mask_blue, 0], uv_coords[mask_blue, 1],
            angles='xy', scale_units='xy', scale=0.1, color='b',
            label="flag = 0"
        )
        plt.legend()
        plt.gca().invert_yaxis()
        plt.title('Initial Displacement Vectors')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.axis('equal')
        plt.grid()
        plt.savefig(
            os.path.join(
                self.config.output_dir, 
                f'initial_displacement_vectors{idx}.png')
            )
        plt.close()

def cal_seed_point(
    idx:int,
    cy: int, cx: int, 
    max_iter: int = 50,
    cutoff_diffnorm: float = 1e-5,
    lambda_reg: float = 1e-3
):
    mask_pad = BufferManager.mask_pad
    
    flag, v0, u0 = coarse_search_int(idx, cy, cx)
    defvector_init = np.zeros(6)
    defvector_init[0], defvector_init[1] = u0, v0
    if flag==0:
        NodeUVInit_buffer.nodes_uv_flage[idx] = flag
        return flag, defvector_init, 10
    
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
    if flag:
        if (defvector[0]-u0) > 1.5:
            flag = 0
        if (defvector[1]-v0) > 1.5:
            flag = 0
    if np.isnan(defvector).any():
        flag = 0
    # print(f"flag: {flag},  corrcoef:{corrcoef}")
    NodeUVInit_buffer.nodes_coord_uv[idx, 0] = defvector[0]
    NodeUVInit_buffer.nodes_coord_uv[idx, 1] = defvector[1]
    NodeUVInit_buffer.nodes_uv_flage[idx] = flag
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
    mask_zero = (search_def == 0)
    if mask_zero.any():
        search_def += mask_zero * (np.random.rand(*search_def.shape).astype(np.float32) * 1e-6)
    mask_zero_ref = (ref_patch == 0)
    if mask_zero_ref.any():
        ref_patch += mask_zero_ref * (np.random.rand(*ref_patch.shape).astype(np.float32) * 1e-6)
        
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
    # if dx_int >1 or dy_int > 1:
    #     print(max_val)
    if max_val > 0.9:
        return 1, dy_int, dx_int
    else:
        return 0, dy_int, dx_int
    




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
        # results = node_init_solver.solve_all_seed_points()
        node_init_solver.load_uv_seed()
        node_init_solver.plot_init_uv(idx)
        savemat(
            os.path.join(
                cfg.mesh_dir, 
                f'node_uv_init_{idx+1:04d}.mat'),
            {'node_uv_init': {
                'nodes_coord_uv': NodeUVInit_buffer.nodes_coord_uv,
                'nodes_coord': NodeUVInit_buffer.nodes_coord
            }}
        )
        
    
    