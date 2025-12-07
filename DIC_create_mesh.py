import numpy as np
from skimage import measure
import pygmsh
import gmsh
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.path import Path
import os

def create_mesh_elemet(mask, mesh_size, simplify_eps, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mask = np.array(mask)   # 转为 NumPy 数组
    mesh = generate_Q8_mesh_from_mask(
        mask=mask, 
        mesh_size=mesh_size, 
        simplify_eps=simplify_eps) 
    print("单元类型:", mesh.cells_dict.keys())  
    
    # 写入单元序号与单元节点序号
    elements_file = os.path.join(output_dir, "elements.txt")
    quad9_cells = mesh.cells_dict.get("quad9", None)
    if quad9_cells is None:
        raise ValueError("mesh.cells_dict 中没有 'quad9' 单元！")
    # 判断节点编号是否从 0 开始，如果是则转成 1-based
    if quad9_cells.min() == 0:
        quad9_cells = quad9_cells + 1   # 全部 +1
    quad9_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 转成 DIC 要求的节点顺序 4角点-4边点-中心点
     # 写入 elements.txt
    with open(elements_file, "w") as fid_quad9:
        for i, conn in enumerate(quad9_cells, start=1):
            conn_reordered = [conn[idx] for idx in quad9_order]
            fid_quad9.write(f"{i}, " + ", ".join(map(str, conn_reordered)) + "\n")
    print("quad9 单元写入完成:", elements_file)
    
    # 写入单元节点序号与单元节点坐标
    nodes_file = os.path.join(output_dir, "nodes.txt")
    quad9_cells = mesh.cells_dict.get("quad9", None)
    if quad9_cells is None:
        raise ValueError("mesh.cells_dict 中没有 'quad9' 单元！")
    # 如果是 0-based 转成 1-based（保持和单元文件一致）
    if quad9_cells.min() == 0:
        quad9_cells = quad9_cells + 1
    nodes_all = mesh.points  # Nx2 节点坐标
    # 1) 找到 quad9 所有节点编号（唯一）
    quad9_node_ids = sorted(set(quad9_cells.flatten().tolist()))
    # 例如 [1,2,3,4,5,6,7,8,9,...]
    # 2) 写 nodes.txt（只写这些节点）
    with open(nodes_file, "w") as fid_nodes:
        for nid in quad9_node_ids:
            coord = nodes_all[nid - 1]  # 注意 1-based → python 0-based
            x, y = coord[:2]
            fid_nodes.write(f"{nid}, {x:.6f}, {y:.6f}\n")
    print("quad9 节点写入完成:", nodes_file)
    
    # 画出mesh网格结构
    coords, id2idx = read_nodes(nodes_file)
    elements = read_elements(elements_file)
    plot_mesh(coords, id2idx, elements, output_dir)
    
    # 获取每个单元内所有整数像素点集合
    Inform_file = os.path.join(output_dir, "Inform.npy")
    build_inform(
        nodes_file=nodes_file,
        elements_file=elements_file,
        Inform_file=Inform_file
    )
    plot_elements_by_number(Inform_file=Inform_file, output_dir=output_dir)
    

def extract_polygon_from_mask(mask, simplify_eps=1.0):
    """
    提取外边界和孔洞边界，返回有序多边形
    """
    contours = measure.find_contours(mask.astype(float), 0.5)
    polygons = []
    for c in contours:
        poly = np.flip(c, axis=1)  # row,col -> x,y
        # 多边形简化
        poly = measure.approximate_polygon(poly, tolerance=simplify_eps)

        # 至少需要 4 点（3角形也不行）
        if len(poly) < 4:
            continue
        polygons.append(poly)
    # 按面积排序（大的是外边界）
    areas = [polygon_area(p) for p in polygons]
    idx = np.argsort(areas)[::-1]
    return [polygons[i] for i in idx]

def polygon_area(poly):
    print(f"边界点数量{len(poly)}")
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def add_polygon_to_geom(geom, poly, mesh_size):
    """
    将 polygon 手动构造成 CurveLoop（而不是使用 add_polygon）
    这样最稳定，不会再触发 AssertionError
    """
    # 添加控制点
    pts = [geom.add_point([x, y, 0], mesh_size=mesh_size) for x, y in poly]
    pts.append(pts[0])
    # B-spline 曲线
    curve = geom.add_bspline(pts)
    # 闭合
    loop = geom.add_curve_loop([curve])
    return loop

def generate_Q8_mesh_from_mask(mask, mesh_size=8.0, simplify_eps=2.0):
    """
    mask → polygon → Q8 mesh
    支持：
        - 任意形状 ROI
        - 多连通域（holes）
        - pygmsh 7.x（无 set_order）
        - 强制用 gmsh API 生成二阶网格，然后抽取 Q8
    """
    # ---- 1. 提取 polygon ----
    polys = extract_polygon_from_mask(mask, simplify_eps=simplify_eps)
    outer = polys[0]
    holes = polys[1:] if len(polys) > 1 else []

    with pygmsh.geo.Geometry() as geom:
        # 外边界
        outer_loop = add_polygon_to_geom(geom, outer, mesh_size)
        # 内孔洞
        hole_loops = []
        for h in holes:
            hole_loops.append(add_polygon_to_geom(geom, h, mesh_size))
        # 构造面
        surface = geom.add_plane_surface(outer_loop, holes=hole_loops)
        # 强制生成四边形网格
        geom.set_recombined_surfaces([surface])
        # 设置全局单元阶数为2（二次单元）
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        # 设置全局单元阶数为2（二次单元）
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        # 生成网格
        mesh = geom.generate_mesh(dim=2)
    return mesh


def read_nodes(node_file):
    node_ids = []
    coords = []
    with open(node_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            nid = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            node_ids.append(nid)
            coords.append([x, y])
    node_ids = np.array(node_ids)
    coords = np.array(coords)
    # 建立 node_id → 索引 的映射
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    return coords, id2idx


def read_elements(elem_file):
    elements = []
    with open(elem_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            # 跳过第一个 elem_id
            conn = [int(v) for v in parts[1:]]
            elements.append(conn)
    return elements


def plot_mesh(coords, id2idx, elements, output_dir):
    plt.figure(figsize=(20, 20))
    quad9_order0based = [0, 4, 1, 5, 2, 6, 3, 7] # 单元逆时针顺序
    for eid, conn in enumerate(elements, start=1):
        # 获取当前单元所有节点坐标
        pts = np.array([coords[id2idx[nid]] for nid in conn])
        pts_reordered  = pts[quad9_order0based]
        pts_polygon = pts_reordered[:8]
        # 闭合 polygon
        pts_closed   = np.vstack([pts_polygon, pts_polygon[0]])
        # 画出单元边界
        plt.plot(pts_closed[:, 0], pts_closed[:, 1], '-k')
        # 可选：画出节点标签
        center_id = conn[8]
        for nid in conn:
            x, y = coords[id2idx[nid]]
            if nid == center_id:
                plt.text(x, y, str(eid), color='blue', fontsize=4)
                continue
            plt.text(x, y, str(nid), color='red', fontsize=4)
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()
    plt.title("Mesh Visualization")
    plt.grid(True)
    save_path = os.path.join(output_dir, "mesh_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    
def get_integer_points_of_element(nodes, center=None):
    """
    nodes: np.array, shape=(8,2) 或 (9,2)
           8 个外圈节点坐标（忽略中心点）
    center: 可选中心点坐标 [Cx, Cy]，不使用时可为 None
    返回: list of [x, y] 所有整数像素点
    """
    quad9_order0based = [0, 4, 1, 5, 2, 6, 3, 7] # 单元逆时针顺序
    if nodes.shape[0] > 8:
        mesh_nodes = nodes[:8, :]
    else:
        mesh_nodes = nodes
        
    polygon = mesh_nodes[quad9_order0based]
    polygon_loop = np.vstack((polygon, polygon[0]))
    # print(polygon_loop.shape)
    path = Path(polygon_loop)

    xmin = int(np.floor(np.min(polygon[:, 0])))
    xmax = int(np.ceil(np.max(polygon[:, 0])))
    ymin = int(np.floor(np.min(polygon[:, 1])))
    ymax = int(np.ceil(np.max(polygon[:, 1])))

    xv, yv = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    points = np.vstack([xv.ravel(), yv.ravel()]).T

    mask = path.contains_points(points)
    integer_points = points[mask]
    return integer_points.tolist()


def build_inform(nodes_file, elements_file, Inform_file):
    
    coords, id2idx = read_nodes(nodes_file)
    elements = read_elements(elements_file)

    Inform = []
    for eid, conn in enumerate(elements, start=1):
        # 获取该单元的节点坐标
        nodes_coord = np.array([coords[id2idx[nid]] for nid in conn])
        # 获取单元内整数像素点
        integer_points = get_integer_points_of_element(nodes_coord, center=None)
        # 保存 x, y, 单元编号
        for pt in integer_points:
            Inform.append([pt[0], pt[1], eid])

    Inform = np.array(Inform)
    np.save(Inform_file, Inform)
    print(f"Inform 构建完成，共 {Inform.shape[0]} 个像素点，已保存到 {Inform_file}")
    return Inform


def plot_elements_by_number(Inform_file, output_dir):
    """
    读取 output_dir 下的 'Inform.npy' 文件，按单元编号绘制散点图，并保存到 output_dir。
    
    参数:
        output_dir (str): 包含 'Inform.npy' 的目录路径，图像也保存到此目录。
    """
    # 1. 加载数据
    Inform = np.load(Inform_file)  # 读取 .npy 文件
    
    # 2. 初始化存储不同编号的元素
    numElements = int(np.max(Inform[:, 2]))  # 找到最大的单元编号
    elements = [[] for _ in range(numElements)]  # 初始化空列表
    
    # 3. 按单元编号分类存储
    for row in Inform:
        num = int(row[2]) - 1  # 转换为 0-based 索引
        elements[num].append(row[:2])  # 存储 (x, y) 坐标
    
    # 4. 定义固定颜色列表
    fixed_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#9c3a3a", "#ff5a36", "#3d9e87", "#cf7e42", "#8a44d0", "#22a5db", "#d0a62b", "#7bce51", "#6e7e45", "#f44336",
        "#3e9e6d", "#d85c2d", "#fdfc23", "#9c09be", "#a3f636", "#491b1b", "#0ac1fe", "#b3fc93", "#9b39e3", "#3f725b",
        "#1f5b3c", "#bc6f53", "#6e44b5", "#25f1c2", "#a66f53", "#2986a5", "#8269c3", "#45f779", "#d45839", "#49e9fe",
        "#f46216", "#3ab4e1", "#69c347", "#c0b73d", "#44bba7", "#bb7a29", "#28d3ef", "#b76d57", "#c3b4db", "#c8bcb4"
    ]
    
    # 5. 绘制散点图
    plt.figure(figsize=(20, 20))
    for num in range(numElements):
        if elements[num]:  # 检查该编号是否有数据
            elements_num = np.array(elements[num])  # 转换为 NumPy 数组
            color_idx = num % len(fixed_colors)  # 循环使用颜色
            plt.scatter(
                elements_num[:, 0], 
                elements_num[:, 1], 
                s=1, 
                c=[fixed_colors[color_idx]], 
                label=f'Element {num + 1}'  # 显示 1-based 编号
            )
    
    # 6. 设置图形属性
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()  # 关键代码：y 轴向下
    plt.title('Scatter Plot of Elements by Number')
    plt.axis('equal')  # 确保坐标轴比例一致
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2, fontsize=8)  # 可选：显示图例
    
    # 7. 保存图像
    save_path = os.path.join(output_dir, 'elements_scatter_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，避免内存泄漏
    
    
if __name__ == "__main__":
    from DIC_load_config import load_mesh_dic_config
    from DIC_read_image import Img_Dataset
    
    cfg = load_mesh_dic_config("./config.json")
    imgGenDataset = Img_Dataset(cfg)
    
    mask = imgGenDataset._get_roiRegion()
    
    create_mesh_elemet(
        mask=mask,
        mesh_size=cfg.mesh_size,
        simplify_eps=cfg.simplify_roi_boundary_poly,
        output_dir=cfg.mesh_dir
    )