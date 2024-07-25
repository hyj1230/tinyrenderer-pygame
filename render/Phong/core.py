from .speedup import *
import pygame
import numpy as np


def render(model, screen, zbuffer, final_matrix, model_matrix, uniform_l, O2):
    # 绘制 3d 模型
    pts = np.matmul(model.vertices, final_matrix)  # 存储屏幕坐标
    view_tri = np.matmul(model.vertices, model_matrix.T)[:, :3]
    norm_vertices = np.matmul(model.norm_vertices, np.linalg.inv(model_matrix))[:, :3]

    generate_faces_new(
        pygame.surfarray.pixels3d(screen),
        model.indices, model.uv_indices, model.norm_indices, pts,
        model.uv_vertices, norm_vertices, view_tri, uniform_l,
        model.texture, model.norm, model.spec, zbuffer, O2
    )  # 逐个绘制三角形
