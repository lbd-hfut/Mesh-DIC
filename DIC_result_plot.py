import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_imshow(idx, Mesh_DIC_buffer, Global2Local_buffer, output_dir):
    """
    Visualize Mesh_DIC_buffer results and save images.
    First row: u, v, valid points
    Second row: ex, ey, ry (create 0-matrix if None)
    Colorbar for u/v based on valid points, ex/ey/ry fixed to [-1, 1] if zeros
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"result_imshow_{idx+1:03d}.png")

    # 获取数据
    u = Mesh_DIC_buffer.plot_u
    v = Mesh_DIC_buffer.plot_v
    ex = Mesh_DIC_buffer.plot_ex
    ey = Mesh_DIC_buffer.plot_ey
    rxy = Mesh_DIC_buffer.plot_rxy
    valid = Global2Local_buffer.plot_validpoints

    # 检查 ex/ey/rxy 是否为 None，如果是则创建 0 矩阵
    shape = u.shape if u is not None else (1,1)
    ex = np.zeros(shape) if ex is None else ex
    ey = np.zeros(shape) if ey is None else ey
    rxy = np.zeros(shape) if rxy is None else rxy

    # 准备绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 第一行
    for ax, data, title in zip(axes[0], [u, v, valid], ["u", "v", "valid points"]):
        if title != "valid points":
            data_to_show = np.where(valid, data, np.nan)
            vmin = np.nanmin(data_to_show)
            vmax = np.nanmax(data_to_show)
            im = ax.imshow(data_to_show, cmap='jet', vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(data, cmap='gray')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")

    # 第二行
    for ax, data, title in zip(axes[1], [ex, ey, rxy], ["ex", "ey", "rxy"]):
        data_to_show = np.where(valid, data, np.nan)
        # 如果全为零，则固定 colorbar 为 [-1, 1]
        if np.allclose(data, 0):
            vmin, vmax = -1, 1
        else:
            vmin = np.nanmin(data_to_show)
            vmax = np.nanmax(data_to_show)
        im = ax.imshow(data_to_show, cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_contourf(idx, Mesh_DIC_buffer, Global2Local_buffer, output_dir):
    """
    Visualize Mesh_DIC_buffer results with contourf and save images.
    First row: u, v, valid points
    Second row: ex, ey, rxy (create 0-matrix if None)
    Colorbar for u/v based on valid points, ex/ey/rxy fixed to [-1, 1] if zeros
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"result_contourf_{idx+1}.png")

    # 获取数据
    u = Mesh_DIC_buffer.plot_u
    v = Mesh_DIC_buffer.plot_v
    ex = Mesh_DIC_buffer.plot_ex
    ey = Mesh_DIC_buffer.plot_ey
    rxy = Mesh_DIC_buffer.plot_rxy
    valid = Global2Local_buffer.plot_validpoints

    # 检查 ex/ey/rxy 是否为 None，如果是则创建 0 矩阵
    shape = u.shape if u is not None else (1,1)
    ex = np.zeros(shape) if ex is None else ex
    ey = np.zeros(shape) if ey is None else ey
    rxy = np.zeros(shape) if rxy is None else rxy

    # 准备绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 第一行
    for ax, data, title in zip(axes[0], [u, v, valid], ["u", "v", "valid points"]):
        if title != "valid points":
            data_to_show = np.where(valid, data, np.nan)
            vmin = np.nanmin(data_to_show)
            vmax = np.nanmax(data_to_show)
            im = ax.contourf(data_to_show, levels=100, cmap='jet', vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(data, cmap='gray')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")

    # 第二行
    for ax, data, title in zip(axes[1], [ex, ey, rxy], ["ex", "ey", "rxy"]):
        data_to_show = np.where(valid, data, np.nan)
        if np.allclose(data, 0):
            vmin, vmax = -1, 1
        else:
            vmin = np.nanmin(data_to_show)
            vmax = np.nanmax(data_to_show)
        im = ax.contourf(data_to_show, levels=100, cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()