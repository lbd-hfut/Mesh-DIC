import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import label
from math import factorial
import gc
from tqdm import tqdm

class BufferManager:
    QKBQKT_ref = None
    QKBQKT_def = None
    fx_ref = None
    fy_ref = None
    fx_def = None
    fy_def = None
    refImg = None
    defImg = None
    mask = None
    mask_pad = None

class Img_Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        train_root = config.input_dir
        image_files = np.array([x.path for x in os.scandir(train_root)
                             if (x.name.endswith(".bmp") or
                             x.name.endswith(".png") or 
                             x.name.endswith(".JPG") or 
                             x.name.endswith(".tiff"))
                             ])
        image_files.sort()
        
        self.rfimage_files = [image_files[0]]
        self.mask_files = [image_files[-1]]
        self.dfimage_files = image_files[1:-1]
        self._get_QK_QKdx_QKdxx()
        self._get_roiRegion()
        refImg, refImg_bcoef = self._get_refImg()
        self._get_image_gradient(Img=refImg, plot_bcoef=refImg_bcoef, flag="ref")
        
    def __len__(self):
        return len(self.dfimage_files)
    
    def __getitem__(self, idx):
        # Open images
        df_image = self.open_image(self.dfimage_files[idx])
        BufferManager.defImg = df_image
        defImg_bcoef = self._form_bcoef(df_image, self.config)
        print(f"create QKBQKT_def{idx+1}:")
        BufferManager.QKBQKT_def = self._get_buffer_QK_B_QKT(defImg_bcoef)
        print(f"create QKBQKT_def{idx+1} over!")
        # print("create fx_def, fy_def:")
        # BufferManager.fx_def, BufferManager.fy_def = self._get_image_gradient(df_image, defImg_bcoef, flag='def')
        # print("create fx_def, fy_def over!")
        return df_image, defImg_bcoef
    
    def open_image(self,name):
        img = Image.open(name).convert('L')
        img = np.array(img)
        return img / 255
    
    def _get_refImg(self):
        refImg = self.open_image(self.rfimage_files[0])
        if BufferManager.refImg is None:
            BufferManager.refImg = refImg
        refImg_bcoef = self._form_bcoef(refImg, self.config)
        if BufferManager.QKBQKT_ref is None:
            print("create QKBQKT_ref:")
            BufferManager.QKBQKT_ref = self._get_buffer_QK_B_QKT(refImg_bcoef)
            print("create QKBQKT_ref over!")
        if BufferManager.fx_ref is None or BufferManager.fy_ref is None:
            print("create fx_ref, fy_ref:")
            BufferManager.fx_ref, BufferManager.fy_ref = self._get_image_gradient(refImg, refImg_bcoef, flag='ref')
            print("create fx_ref, fy_ref over!")
        return refImg, refImg_bcoef
    
    def _get_roiRegion(self):
        mask_bin = self.open_image(self.mask_files[0]) > 0
        _, num_labels = label(mask_bin)
        if num_labels == 0:
            raise RuntimeError("Mask 中没有前景像素！")
        if num_labels > 1:
            raise RuntimeError("Mask 只支持单连通域！")
        
        if BufferManager.mask is None:
            BufferManager.mask = mask_bin
        if BufferManager.mask_pad is None:
            BufferManager.mask_pad = np.pad(
                mask_bin, pad_width=self.config.subset_r, 
                mode='constant', constant_values=False)
        return mask_bin
    
    def beta5_nth(self, x, n=0):
        """
        Quintic B-spline β5(x) n-th derivative
        x: np.float64 or np.array
        n: 0~5
        """
        x = np.asarray(x, dtype=np.float32)

        def plus_power(y, p):
            return y**p * (y>0)

        coeffs = [1, -6, 15, -20, 15, -6, 1]
        shifts = [-3, -2, -1, 0, 1, 2, 3]

        result = np.zeros_like(x)
        for c, s in zip(coeffs, shifts):
            factor = factorial(5) // factorial(5-n)
            result += c * factor * plus_power(x + s, 5-n)
        return result / 120
    
    def _get_QK_QKdx_QKdxx(self):
        x_samples = np.array([-2, -1, 0, 1, 2, 3], dtype=np.float32)
        
        # 生成 QK（n = 0~5）
        QK = np.zeros((6, len(x_samples)))  # QK[n, :]
        for n in range(6):
            QK[n, :] = ((-1) ** n) * self.beta5_nth(x_samples, n=n) / factorial(n)
        self.QK = QK

    
    def _form_bcoef(self, img, config):
        """Replicate padding by 'border' pixels on each side."""
        if config.bcoef_border >= 3:
            border = config.bcoef_border
        else:
            border = 3
        # padding img
        plot_gs = np.pad(img, pad_width=border, mode='edge')
        h, w = plot_gs.shape
        if h < 5 or w < 5:
            raise ValueError("Array must be >= 5×5 or empty")
        plot_bcoef = np.zeros_like(plot_gs, dtype=np.complex128)
        x_sample = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        kernel_b = self.beta5_nth(x_sample, n=0)
        # Row kernel
        kernel_b_x = np.zeros(w, dtype=np.float32)
        kernel_b_x[:3] = kernel_b[2:]
        kernel_b_x[-2:] = kernel_b[:2]
        kernel_b_x = np.fft.fft(kernel_b_x)
        # across rows
        for i in range(h):
            plot_bcoef[i, :] = np.fft.ifft(np.fft.fft(plot_gs[i, :]) / kernel_b_x)
        # Column kernel
        kernel_b_y = np.zeros(h, dtype=np.float32)
        kernel_b_y[:3] = kernel_b[2:]
        kernel_b_y[-2:] = kernel_b[:2]
        kernel_b_y = np.fft.fft(kernel_b_y)
        # across columns
        for j in range(w):
            plot_bcoef[:, j] = np.fft.ifft(np.fft.fft(plot_bcoef[:, j]) / kernel_b_y)
        return plot_bcoef.real
    
    def _get_image_gradient(self, Img, plot_bcoef, flag='ref'):
        if not hasattr(self, "QK"):
            self._get_QK_QKdx_QKdxx()
        if flag == 'ref':
            # 加载参考图像及其 B 样条系数
            roi = self._get_roiRegion()
        elif flag == 'def':
            roi = np.ones_like(Img, dtype=bool)
        else:
            raise ValueError("flag must be 'ref' or 'def'")
        
        H, W = Img.shape
        fx = np.zeros_like(Img, dtype=np.float32)
        fy = np.zeros_like(Img, dtype=np.float32)
        
        if self.config.bcoef_border >= 3:
            border = self.config.bcoef_border
        else:
            border = 3
        offset = 2
        roi_pixels = []
        ys, xs = np.where(roi)
        roi_pixels.extend(zip(ys, xs))

        for y, x in tqdm(roi_pixels, desc="Computing image gradients", total=len(roi_pixels)):
            x_vec = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)  # (6,)
            w_vec = np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)
            top = y + border - offset
            left = x + border - offset
            bottom = top + 6
            right = left + 6
            block = plot_bcoef[top:bottom, left:right]
            fx[y, x] = x_vec @ (self.QK @ (block @ (self.QK.T @ w_vec)))
            fy[y, x] = w_vec @ (self.QK @ (block @ (self.QK.T @ x_vec)))
        return fx, fy
        
    def _get_buffer_QK_B_QKT(self, plot_bcoef):
        # Torch 全局设定
        device = torch.device("cpu")  # 也可以 cuda 加速
        dtype = torch.float32
        # ref_bcoef pad to avoid border checks for 6x6 block
        border = self.config.bcoef_border if self.config.bcoef_border >= 3 else 3
        offset = 2
        # QK、梯度、参考图全部转 torch
        if not hasattr(self, "QK"):
            self._get_QK_QKdx_QKdxx()
        QK = _to_torch(self.QK, device, dtype)

        # reference and bcoef
        ref_bcoef = _to_torch(plot_bcoef, device, dtype)

        # ROI list and padded ROI's
        roi = self._get_roiRegion()
        H, W = roi.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        ys = ys.reshape(-1)
        xs = xs.reshape(-1)
        num_pts = ys.numel()

        # -------------------------
        # Pre-allocate buffers ONCE
        # -------------------------
        QK_B_QKT_6 = torch.zeros((num_pts, 6, 6), device=device, dtype=dtype)
        tmp6_buffer = torch.empty((6, 6), device=device, dtype=dtype)

        # Loop over points (only this single loop remains)
        for p_idx in tqdm(range(num_pts), desc="Computing QK B QK^T"):
            yc = ys[p_idx].item()
            xc = xs[p_idx].item()
            # ---- compute QK * block * QK^T and store (6x6 small) ----
            top = yc + border - offset
            left = xc + border - offset
            block = ref_bcoef[top:top + 6, left:left + 6]  # view or small array
            # small tmp created here (6x6) - unavoidable
            torch.matmul(QK, block, out=tmp6_buffer)
            torch.matmul(tmp6_buffer, QK.T, out=QK_B_QKT_6[p_idx])
        # convert to numpy
        QK_B_QKT_6_np = _to_numpy(QK_B_QKT_6)
        QK_B_QKT_hash_map = {
            (int(ys[i]), int(xs[i])): QK_B_QKT_6_np[i]
            for i in range(num_pts)
        }
        
        # return buffers
        return QK_B_QKT_hash_map
    
def collate_fn(batch):
    return batch  

def _to_torch(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def _to_numpy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        device = getattr(x, "device", None)
        if device == "cuda":
            x_cpu = x.detach().to("cpu")
            np_x = x_cpu.numpy()
        else:
            np_x = x.numpy()
        if dtype is not None:
            return np_x.astype(dtype)
        return np_x

if __name__ == "__main__":
    from DIC_load_config import load_dic_config
    from scipy.io import savemat
    import time

    cfg = load_dic_config("./config.json")
    imgGenDataset = Img_Dataset(cfg)
    
    imgGenDataset._get_QK_QKdx_QKdxx()
    print("_get_QK_QKdx_QKdxx over")
    
    start_time = time.time()
    imgGenDataset._get_image_gradient()
    total_time = time.time()-start_time
    print("_get_image_gradient over")
    print(f"cost {total_time}s")
    
    start_time = time.time()
    refImg, ref_bcoef = imgGenDataset._get_refImg()
    total_time = time.time()-start_time
    print("_get_refImg over")
    print(f"cost {total_time}s")
    
    imgGenDataset._get_roiRegion()
    print("_get_roiRegion over")
    

    
    
    
    