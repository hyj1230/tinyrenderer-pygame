from .speedup import *
import pygame
import numpy as np


def render(model, screen, zbuffer, final_matrix, light_dirs, O2):
    # 绘制 3d 模型
    pts = np.matmul(model.vertices, final_matrix)  # 存储屏幕坐标
    norm_vertices = np.zeros(model.norm_vertices.shape[0], dtype=np.float64)

    for light_dir in light_dirs:
        norm_vertices += np.maximum(np.dot(model.norm_vertices, light_dir), 0.0)

    generate_faces_new(
        pygame.surfarray.pixels3d(screen),
        model.indices, model.uv_indices, model.norm_indices, pts,
        model.uv_vertices, norm_vertices, model.texture_manager.texture_array,
        model.texture_manager.textures_sizes, model.uv_index, zbuffer, O2
    )  # 逐个绘制三角形
