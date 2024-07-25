import numpy as np
from PIL import Image


class TextureManager:
    def __init__(self, max_texture_width, max_texture_height, max_textures):
        self.max_texture_width = max_texture_width
        self.max_texture_height = max_texture_height
        self.max_textures = max_textures

        self.textures = []

        self.texture_array = np.zeros((self.max_textures, self.max_texture_height, self.max_texture_width, 4), dtype=np.uint32)
        self.textures_sizes = np.zeros((self.max_textures, 2), dtype=np.uint32)
        self.have_alpha = np.zeros(self.max_textures, dtype=np.uint8)

    @staticmethod
    def load_texture_img(name):
        return np.array(Image.open(name).convert('RGBA'))

    def add_texture(self, texture):
        if texture not in self.textures:
            self.textures.append(texture)
            texture_index = len(self.textures) - 1
            texture_arr = self.load_texture_img(texture)
            texture_width, texture_height = texture_arr.shape[1], texture_arr.shape[0]
            self.texture_array[texture_index, :texture_height, :texture_width] = texture_arr[:, :]
            self.textures_sizes[texture_index, :] = texture_width, texture_height
            if np.any(texture_arr[:, :, 3] == 0):
                self.have_alpha[texture_index] = 1
        return self.textures.index(texture)
