import numpy as np
from PIL import Image
from render.texture_manager import TextureManager
import os


def normalize(v):
    # 归一化操作
    unit = np.linalg.norm(v)
    if unit == 0:
        return np.zeros(3, dtype=np.float64)
    return v / unit


def get_mtl_filename(obj_filename):
    mtl_filename = None
    with open(obj_filename, encoding='utf-8') as f:
        for line in f:
            line = line.replace('  ', ' ')
            if line.startswith("mtllib "):
                mtl_filename = line.strip("mtllib").strip()
    return mtl_filename


class GourandModel:
    def __init__(self, filename, texture_filename, div_num=1):
        """
        https://en.wikipedia.org/wiki/Wavefront_.obj_file#Vertex_normal_indices
        """
        self.shader = 'Gouraud'
        self.vertices = []
        self.uv_vertices = []
        self.norm_vertices = []
        self.uv_indices = []
        self.indices = []
        self.norm_indices = []
        self.div_num = div_num

        self.texture_manager = TextureManager(*Image.open(texture_filename).size, 1)
        self.texture_manager.add_texture(texture_filename)

        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.replace('  ', ' ')
                if line.startswith("v "):
                    x, y, z = [float(d) for d in line.strip("v").strip().split(" ")][:3]
                    self.vertices.append(np.array([x/self.div_num, y/self.div_num, z/self.div_num, 1]))
                elif line.startswith("vn "):
                    norm = [float(d) for d in line.strip("vn").strip().split(" ")][:3]
                    self.norm_vertices.append(normalize(np.array(norm, dtype=np.float64)))
                elif line.startswith("vt "):
                    u, v = [float(d) for d in line.strip("vt").strip().split(" ")][:2]
                    self.uv_vertices.append([u, 1-v])
                elif line.startswith("f "):
                    facet = [d.split("/") for d in line.strip("f").strip().split(" ")]
                    self.indices.append([int(d[0])-1 for d in facet])
                    self.uv_indices.append([int(d[1])-1 for d in facet])
                    self.norm_indices.append([int(d[2])-1 for d in facet])
        self.vertices = np.array(self.vertices, dtype=np.float64)
        self.norm_vertices = np.array(self.norm_vertices, dtype=np.float64)
        self.uv_vertices = np.array(self.uv_vertices, dtype=np.float64)
        self.indices = np.array(self.indices, dtype=np.uint32)
        self.uv_indices = np.array(self.uv_indices, dtype=np.uint32)
        self.norm_indices = np.array(self.norm_indices, dtype=np.uint32)
        self.uv_index = np.zeros(self.indices.shape[0], dtype=np.uint32)
        print(f'3D模型 {filename} 加载成功')
        print('总面数:', len(self.indices))
        print('总顶点数:', len(self.vertices))
        print('--------------------------------')


class GourandModelNew:
    def __init__(self, filename, div_num=1, offset_y=0):
        """
        https://en.wikipedia.org/wiki/Wavefront_.obj_file#Vertex_normal_indices
        """
        self.shader = 'Gouraud'
        self.vertices = []
        self.uv_vertices = []
        self.norm_vertices = []
        self.uv_indices = []
        self.indices = []
        self.norm_indices = []
        self.uv_index = []
        self.div_num = div_num

        texture = {}
        vis = {}
        max_w, max_h = 0, 0
        path = os.path.dirname(filename)
        with open(path + '/' + get_mtl_filename(filename), encoding='utf-8') as f:
            mtl_name = None
            for line in f:
                line = line.replace('  ', ' ')
                if line.startswith("newmtl "):
                    mtl_name = line.strip("newmtl").strip()
                elif line.startswith("map_Kd "):
                    name = line.strip("map_Kd").strip()
                    w, h = Image.open(path + '/' + name).size
                    max_w, max_h = max(max_w, w), max(max_h, h)
                    texture[mtl_name] = path + '/' + name
                    vis[path + '/' + name] = 1

        self.texture_manager = TextureManager(max_w, max_h, len(vis))

        with open(filename, encoding='utf-8') as f:
            _uv_index = 0
            for line in f:
                line = line.replace('  ', ' ')
                if line.startswith("v "):
                    x, y, z = [float(d) for d in line.strip("v").strip().split(" ")][:3]
                    self.vertices.append(np.array([x/self.div_num, y/self.div_num+offset_y, z/self.div_num, 1]))
                elif line.startswith("vn "):
                    norm = [float(d) for d in line.strip("vn").strip().split(" ")][:3]
                    self.norm_vertices.append(normalize(np.array(norm, dtype=np.float64)))
                elif line.startswith("vt "):
                    u, v = [float(d) for d in line.strip("vt").strip().split(" ")][:2]
                    if u < 0:
                        u = 1-abs(u)
                    if v < 0:
                        v = 1-abs(v)
                    self.uv_vertices.append([u, 1-v])
                elif line.startswith("usemtl "):
                    mtl_name = line.strip("usemtl").strip()
                    _uv_index = self.texture_manager.add_texture(texture[mtl_name])
                elif line.startswith("f "):
                    facet = [d.split("/") for d in line.strip("f").strip().split(" ")]
                    self.indices.append([int(d[0])-1 for d in facet])
                    self.uv_indices.append([int(d[1])-1 for d in facet])
                    self.norm_indices.append([int(d[2])-1 for d in facet])
                    self.uv_index.append(_uv_index)

        self.vertices = np.array(self.vertices, dtype=np.float64)
        self.norm_vertices = np.array(self.norm_vertices, dtype=np.float64)
        self.uv_vertices = np.array(self.uv_vertices, dtype=np.float64)
        self.indices = np.array(self.indices, dtype=np.uint32)
        self.uv_indices = np.array(self.uv_indices, dtype=np.uint32)
        self.norm_indices = np.array(self.norm_indices, dtype=np.uint32)
        self.uv_index = np.array(self.uv_index, dtype=np.uint32)
        print(f'3D模型 {filename} 加载成功')
        print('总面数:', len(self.indices))
        print('总顶点数:', len(self.vertices))
        print('--------------------------------')
