import numpy as np


eye = np.array([0, 0, 3], dtype=np.float64)  # 相机位置
center = np.array([0, 0, 0], dtype=np.float64)  # 观察点的位置
up = np.array((0, 1, 0), dtype=np.float64)  # 相机朝向


def normalize(v):
    # 归一化操作
    unit = np.linalg.norm(v)
    if unit == 0:
        return np.zeros(v.shape[0], dtype=np.float64)
    return v / unit


def rotate_x(angle):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1],
        ], dtype=np.float64
    )


def rotate_y(angle):
    return np.array(
        [
            [np.cos(angle), 0, -np.sin(angle), 0],
            [0, 1, 0, 0],
            [np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ], dtype=np.float64
    )


def rotate_z(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64
    )


def zoom(scale):
    return np.array(
        [
            [scale, 0, 0, 0],
            [0, scale, 0, 0],
            [0, 0, scale, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64
    )


def look_at(_eye: np.ndarray, _center: np.ndarray, _up: np.ndarray):
    """
    Args:
        _eye: 摄像机的世界坐标位置
        _center: 观察点的位置
        _up: 就是你想让摄像机立在哪个方向
    """
    z = normalize(_center - _eye)
    x = normalize(np.cross(_up, z))
    y = normalize(np.cross(z, x))

    Minv = np.array(
        [[*x, 0], [*y, 0], [*z, 0], [0, 0, 0, 1.0]],
        dtype=np.float64
    )
    Tr = np.array(
        [[1, 0, 0, -_eye[0]], [0, 1, 0, -_eye[1]], [0, 0, 1, -_eye[2]], [0, 0, 0, 1.0]],
        dtype=np.float64
    )

    return np.dot(Minv, Tr)


def perspective_project(f, w, h):
    return np.array(
        [
            [1,  0,    0, 0],
            [0,  w/h,    0, 0],
            [0,  0,    1, 0],
            [0,  0, -1/f, 0],
        ], dtype=np.float64
    )


def viewport(x, y, w, h):
    return np.array(
        [
            [w/2, 0, 0, x+w/2],
            [0, h/2, 0, y+h/2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64
    )


def translation(tx, ty, tz):
    return np.array(
        [
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1],
        ], dtype=np.float64
    )


def calc_matrix_gounard(width, height, angle_x, angle_y, scale, tx, ty, tz):
    transform_matrix = np.dot(
        translation(tx, ty, tz),
        np.dot(
            np.dot(
                rotate_x(angle_x), rotate_y(angle_y)
            ),
            zoom(scale)
        )
    )
    ModelView = look_at(eye, center, up)  # 模型矩阵
    Projection = perspective_project(np.linalg.norm(eye - center), width, height)  # 透视矩阵
    Viewport = viewport(width / 8 / (4-3), height / 8 / (4-3), width * 3 / 4, height * 3 / 4)  # 视口矩阵

    # 矩阵预乘
    final_matrix = np.dot(Viewport, np.dot(Projection, np.dot(ModelView, transform_matrix))).T
    return final_matrix, np.dot(ModelView, transform_matrix)
