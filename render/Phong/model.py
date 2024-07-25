import numpy as np
from PIL import Image
from render.matrix import normalize


class PhongModel:
    def __init__(self, filename, diffuse_filename, norm_filename, spec_filename=None, div_num=1):
        self.shader = 'Phong'
        self.vertices = []
        self.uv_vertices = []
        self.norm_vertices = []
        self.uv_indices = []
        self.indices = []
        self.norm_indices = []
        self.div_num = div_num

        self.texture = np.array(Image.open(diffuse_filename).convert('RGB'))  # 漫反射
        self.norm = np.array(Image.open(norm_filename).convert('RGB').resize(self.texture.shape[:2][::-1])) * 2.0 / 255.0 - 1.0  # 切线空间法线
        self.norm[:, :, :2] = 1 / self.norm[:, :, :2]
        if spec_filename is None:
            self.spec = np.zeros(self.texture.shape[:2], dtype=np.uint8)
        else:
            self.spec = np.array(Image.open(spec_filename).convert('L').resize(self.texture.shape[:2][::-1]))  # 镜面反射

        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.replace('  ', ' ')
                if line.startswith("v "):
                    x, y, z = [float(d) for d in line.strip("v").strip().split(" ")][:3]
                    self.vertices.append(np.array([x/self.div_num, y/self.div_num, z/self.div_num, 1]))
                elif line.startswith("vn "):
                    norm = [float(d) for d in line.strip("vn").strip().split(" ")][:3]
                    self.norm_vertices.append(np.append(normalize(np.array(norm, dtype=np.float64)), 0))
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
        print(f'3D模型 {filename} 加载成功')
        print('总面数:', len(self.indices))
        print('总顶点数:', len(self.vertices))
        print('--------------------------------')
