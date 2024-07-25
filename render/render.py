import numpy as np
import pygame
from .matrix import calc_matrix_gounard, normalize
from .Gouraud import core as gouraud_core
from .Gouraud import model as gouraud_model
from .Phong import core as phong_core


light_dirs = [normalize(np.array([1, 1, 1], dtype=np.float64)),
              normalize(np.array([-1, 1, -1], dtype=np.float64))][:1]

light_dirs1 = [normalize(np.array([1, 1, 1], dtype=np.float64)),
               normalize(np.array([0, 0, -1], dtype=np.float64))]


def render_light(screen, final_matrix, light_dirs, z_buffer):
    light_dirs_np = np.concatenate((np.array(light_dirs), np.ones((len(light_dirs), 1))), axis=1)
    light_dirs_np = np.matmul(light_dirs_np, final_matrix)
    for light_index in np.argsort(light_dirs_np[:, 2])[-1::-1]:
        light_pos = light_dirs_np[light_index]
        if light_pos[3] > -0.1:
            continue
        x, y, z = light_pos[0] / light_pos[3], light_pos[1] / light_pos[3], light_pos[2]
        if x < 0 or x >= z_buffer.shape[0] or y < 0 or y >= z_buffer.shape[1] or z > z_buffer[int(x), int(y)]:
            continue
        pygame.draw.rect(screen, (0,)*3, (x-2, y-2, 4, 4))


def render_model(_model, screen, z_buffer, angle_x, angle_y, scale, tx, ty, tz, O2=True):
    if _model.shader == 'Gouraud':
        if isinstance(_model, gouraud_model.GourandModelNew):
            angle_y += np.pi
            _light_dirs = light_dirs1
        else:
            _light_dirs = light_dirs
        final_matrix, _ = calc_matrix_gounard(*screen.get_size(), angle_x, angle_y, scale, tx, ty, tz)
        gouraud_core.render(_model, screen, z_buffer, final_matrix, _light_dirs, O2)
        # render_light(screen, final_matrix, light_dirs, z_buffer)
    if _model.shader == 'Phong':
        final_matrix, model_matrix = calc_matrix_gounard(*screen.get_size(), angle_x, angle_y, scale, tx, ty, tz)
        uniform_l = normalize(np.dot(np.array([1, 1, 1, 0], dtype=np.float64), model_matrix.T)[:3])
        phong_core.render(_model, screen, z_buffer, final_matrix, model_matrix, uniform_l, O2)
