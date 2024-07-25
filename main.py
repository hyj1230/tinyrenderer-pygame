from render import render_model, gouraud_model, phong_model
import pygame
import sys
import numpy as np

pygame.init()
screen = pygame.display.set_mode((600, 600), pygame.RESIZABLE)
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 11)

model = gouraud_model.GourandModel(filename="data/african_head/african_head.obj",
                                   texture_filename="data/african_head/african_head_diffuse.tga")
model1 = gouraud_model.GourandModel(filename="data/african_head/african_head_eye_inner.obj",
                                    texture_filename="data/african_head/african_head_eye_inner_diffuse.tga")
model2 = phong_model.PhongModel(filename="data/african_head/african_head.obj",
                                diffuse_filename="data/african_head/african_head_diffuse.tga",
                                norm_filename="data/african_head/african_head_nm_tangent.tga",
                                spec_filename="data/african_head/african_head_spec.tga")
model3 = phong_model.PhongModel(filename="data/african_head/african_head_eye_inner.obj",
                                diffuse_filename="data/african_head/african_head_eye_inner_diffuse.tga",
                                norm_filename="data/african_head/african_head_eye_inner_nm_tangent.tga",
                                spec_filename="data/african_head/african_head_eye_inner_spec.tga")
'''model3 = gouraud_model.GourandModel(filename="data/other/axe.obj",
                                    texture_filename="data/other/axe.tga")
model4 = gouraud_model.GourandModel(filename="data/other/jinx.obj",
                                    texture_filename="data/other/jinx.tga")
model5 = gouraud_model.GourandModel(filename="data/other/monkey.obj",
                                    texture_filename="data/other/monkey.png")
model6 = gouraud_model.GourandModel(filename="data/pbr/backpack.obj",
                                    texture_filename="data/pbr/diffuse.jpg", div_num=1/0.6)
model7 = gouraud_model.GourandModelNew("data/ht/hutao.obj", 10, offset_y=-1.1)'''

models = [[model, model1], [model2, model3]]  # , [model4], [model5], [model6], [model7]]
index = 0

angle_x = 0.0
angle_y = 0.0
scale = 1.0
mouse_dragging = False
O2 = False  # 是否开启背面剔除
tx, ty, tz = 0, 0, 0


FPS_list = []

while 1:
    FPS = str(round(clock.get_fps()))
    pygame.display.set_caption('fps:' + FPS)
    pygame_events = pygame.event.get()
    for event in pygame_events:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_dragging = True
            if event.button == 4:
                scale += 0.1 * scale
                scale = min(scale, 10.0)
            if event.button == 5:
                scale -= 0.1 * scale
                scale = max(0.1, scale)
        if event.type == pygame.MOUSEMOTION:
            if mouse_dragging:
                angle_x += event.rel[1]/300
                angle_y += (-event.rel[0])/100
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_dragging = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_o:
                O2 = not O2
            if event.key == pygame.K_LEFT:
                index -= 1
                index %= len(models)
            if event.key == pygame.K_RIGHT:
                index += 1
                index %= len(models)
    screen.fill((255, 255, 255))
    z_buffer = np.full(screen.get_size(), np.inf, dtype=np.float64)
    for _model in models[index]:
        render_model(_model, screen, z_buffer, angle_x, angle_y, scale, tx, ty, tz, O2)
    pygame.display.flip()
    clock.tick(114514)
