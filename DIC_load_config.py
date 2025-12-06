import os
import json
from types import SimpleNamespace

def load_mesh_dic_config(json_path):
    """
    读取 Mesh-DIC 配置文件（JSON），返回 SimpleNamespace 类型 cfg 对象：
        cfg.input_dir
        cfg.output_dir
        cfg.mesh_dir
        cfg.mesh_size
        cfg.max_iterations
        cfg.cutoff_diffnorm
        cfg.lambda_reg
        cfg.displacement_init
        ...
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"配置文件不存在：{json_path}")

    # === 读取 JSON 文件 ===
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # === 默认参数 ===
    default_cfg = {
        "input_dir": "./case/case1/",
        "output_dir": "./case/case1/result/",
        "mesh_dir": "./case/case1_mesh/",
        "mesh_size": 30.0,
        "simplify_roi_boundary_poly": 2.0,
        "bcoef_border": 3,
        "parallel": False,
        "max_workers": 8,
        "max_iterations": 30,
        "cutoff_diffnorm": 1e-4,
        "lambda_reg": 1e-6,
        "displacement_init": "int_pixels", 
        "subset_r": 20,
        "search_radius": 20,           
        "smooth_flag": True,
        "smooth_method": "gaussian",
        "smooth_sigma": 1.0,
        "strain_calculate_flag": True,
        "strain_method": "gaussian_window",
        "strain_window_half_size": 25,
        "show_plot": True,
        "save_mesh_plot": True
    }

    # === 补齐缺失的默认值 ===
    for key, val in default_cfg.items():
        if key not in cfg:
            cfg[key] = val

    # === 类型转换，保证安全 ===
    int_keys = ["bcoef_border", "max_workers", "strain_window_half_size", "max_iterations", "subset_r", "search_radius"]
    float_keys = ["mesh_size", "simplify_roi_boundary_poly", "cutoff_diffnorm", "lambda_reg", "smooth_sigma"]
    bool_keys = ["parallel", "smooth_flag", "strain_calculate_flag", "show_plot", "save_mesh_plot"]
    str_keys  = ["input_dir", "output_dir", "mesh_dir", "displacement_init", "smooth_method", "strain_method"]

    for k in int_keys:
        cfg[k] = int(cfg[k])
    for k in float_keys:
        cfg[k] = float(cfg[k])
    for k in bool_keys:
        cfg[k] = bool(cfg[k])
    for k in str_keys:
        cfg[k] = str(cfg[k])

    # === 确保路径最后有 "/" ===
    for key in ["input_dir", "output_dir", "mesh_dir"]:
        if not cfg[key].endswith("/"):
            cfg[key] += "/"

    # === 转换为 SimpleNamespace，支持 cfg.xxx 访问 ===
    return SimpleNamespace(**cfg)
