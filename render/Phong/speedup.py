import numpy as np
from numba import jit, prange
import math


@jit('UniTuple(int32, 2)(float64,float64,float64)', nopython=True, cache=True)
def get_min_max(a, b, c):
    return int(min(min(a, b), c)), int(max(max(a, b), c))


@jit('int32(int32,int32,int32)', nopython=True, cache=True)
def clip(a, b, c):
    return max(b, min(a, c))


@jit('float64(float64[:],float64[:])', nopython=True, cache=True, fastmath=True)
def get_intersect_ratio(prev, curv):
    return (prev[3] + 0.1) / (prev[3] - curv[3])


@jit('float64[:](float64[:],float64[:],float64)', nopython=True, cache=True, fastmath=True)
def lerp_vec(start, end, alpha):
    return start + (end - start) * alpha


@jit(nopython=True, cache=True, fastmath=True)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@jit(nopython=True, cache=True, fastmath=True)
def normalized(a, b, c, norm):
    unit = math.sqrt(a * a + b * b + c * c) * norm
    return a / unit, b / unit, c / unit


@jit(nopython=True, cache=True, fastmath=True)
def normalized1(a, b, c):
    unit = math.sqrt(a * a + b * b + c * c)
    return a / unit, b / unit, c / unit


@jit(nopython=True, fastmath=True, looplift=True)
def render_opaque_face(screen, zbuffer, ptsa, ptsb, ptsc, uva, uvb, uvc,
                       norm_a, norm_b, norm_c, view_tri_a, view_tri_b, view_tri_c, uniform_l,
                       texture, norm_texture, spec_texture, O2):
    width, height = texture.shape[1], texture.shape[0]
    clip_a, clip_b, clip_c = ptsa[2], ptsb[2], ptsc[2]
    pts2a = ptsa[0] / ptsa[3], ptsa[1] / ptsa[3]
    pts2b = ptsb[0] / ptsb[3], ptsb[1] / ptsb[3]
    pts2c = ptsc[0] / ptsc[3], ptsc[1] / ptsc[3]
    bc_c = (pts2c[0] - pts2a[0]) * (pts2b[1] - pts2a[1]) - (pts2b[0] - pts2a[0]) * (pts2c[1] - pts2a[1])
    if (bc_c if O2 else abs(bc_c)) < 1e-3:
        return
    _a, _b, _c = view_tri_b[0] - view_tri_a[0], view_tri_b[1] - view_tri_a[1], view_tri_b[2] - view_tri_a[2]
    _d, _e, _f = view_tri_c[0] - view_tri_a[0], view_tri_c[1] - view_tri_a[1], view_tri_c[2] - view_tri_a[2]

    vec_u = uvb[0] - uva[0], uvc[0] - uva[0]
    vec_v = uvb[1] - uva[1], uvc[1] - uva[1]

    ptsa, ptsb, ptsc = 1 / ptsa[3], 1 / ptsb[3], 1 / ptsc[3]
    minx, maxx = get_min_max(pts2a[0], pts2b[0], pts2c[0])
    miny, maxy = get_min_max(pts2a[1], pts2b[1], pts2c[1])
    minx, maxx = clip(minx, 0, screen.shape[0] - 1), clip(maxx + 1, 1, screen.shape[0])
    miny, maxy = clip(miny, 0, screen.shape[1] - 1), clip(maxy + 1, 1, screen.shape[1])
    a, b, c, d = (pts2c[0] - pts2a[0]) / bc_c, (pts2b[0] - pts2a[0]) / bc_c, (pts2c[1] - pts2a[1]) / bc_c, (pts2b[1] - pts2a[1]) / bc_c
    uva = uva[0] * width * ptsa, uva[1] * height * ptsa
    uvb = uvb[0] * width * ptsb, uvb[1] * height * ptsb
    uvc = uvc[0] * width * ptsc, uvc[1] * height * ptsc
    norm_a = norm_a[0] * ptsa, norm_a[1] * ptsa, norm_a[2] * ptsa
    norm_b = norm_b[0] * ptsb, norm_b[1] * ptsb, norm_b[2] * ptsb
    norm_c = norm_c[0] * ptsc, norm_c[1] * ptsc, norm_c[2] * ptsc
    clip_a, clip_b, clip_c = clip_a * ptsa, clip_b * ptsb, clip_c * ptsc
    tmp1 = pts2a[1] - miny + 1

    addm0, addm1, addm2 = -(a - b), a, -b
    bc_clip_add = addm0 * ptsa + addm1 * ptsb + addm2 * ptsc
    clip_add = addm0 * clip_a + addm1 * clip_b + addm2 * clip_c
    u_add, v_add = uva[1] * addm0 + uvb[1] * addm1 + uvc[1] * addm2, uva[0] * addm0 + uvb[0] * addm1 + uvc[0] * addm2
    norm_r_add = norm_a[0] * addm0 + norm_b[0] * addm1 + norm_c[0] * addm2
    norm_g_add = norm_a[1] * addm0 + norm_b[1] * addm1 + norm_c[1] * addm2
    norm_b_add = norm_a[2] * addm0 + norm_b[2] * addm1 + norm_c[2] * addm2

    addmm0, addmm1, addmm2 = -(d - c), -c, d
    add_bc_clip_sum = addmm0 * ptsa + addmm1 * ptsb + addmm2 * ptsc
    add_clip_sum = addmm0 * clip_a + addmm1 * clip_b + addmm2 * clip_c
    add_u = uva[1] * addmm0 + uvb[1] * addmm1 + uvc[1] * addmm2
    add_v = uva[0] * addmm0 + uvb[0] * addmm1 + uvc[0] * addmm2
    add_norm_r = norm_a[0] * addmm0 + norm_b[0] * addmm1 + norm_c[0] * addmm2
    add_norm_g = norm_a[1] * addmm0 + norm_b[1] * addmm1 + norm_c[1] * addmm2
    add_norm_b1 = norm_a[2] * addmm0 + norm_b[2] * addmm1 + norm_c[2] * addmm2

    temp = pts2a[0] - float(minx)
    mm0, mm1, mm2 = b * tmp1 - temp * d, temp * c - a * tmp1, 1 + temp * (d - c) + tmp1 * (a - b)
    _bc_clip_sum = mm2 * ptsa + mm1 * ptsb + mm0 * ptsc
    _clip_sum = mm2 * clip_a + mm1 * clip_b + mm0 * clip_c
    _u, _v = uva[1] * mm2 + uvb[1] * mm1 + uvc[1] * mm0, uva[0] * mm2 + uvb[0] * mm1 + uvc[0] * mm0
    _norm_r = norm_a[0] * mm2 + norm_b[0] * mm1 + norm_c[0] * mm0
    _norm_g = norm_a[1] * mm2 + norm_b[1] * mm1 + norm_c[1] * mm0
    _norm_b1 = norm_a[2] * mm2 + norm_b[2] * mm1 + norm_c[2] * mm0
    for j in prange(minx, maxx):
        flag = False
        m0, m1, m2 = mm0, mm1, mm2
        bc_clip_sum = _bc_clip_sum
        clip_sum = _clip_sum
        u, v = _u, _v
        norm_r, norm_g, norm_b1 = _norm_r, _norm_g, _norm_b1

        mm0 += addmm2
        mm1 += addmm1
        mm2 += addmm0
        _bc_clip_sum += add_bc_clip_sum
        _clip_sum += add_clip_sum
        _u += add_u
        _v += add_v
        _norm_r += add_norm_r
        _norm_g += add_norm_g
        _norm_b1 += add_norm_b1
        for k in prange(miny, maxy):
            # 必须显式转换成 double 参与底下的运算，不然结果是错的
            m0 += addm2
            m1 += addm1
            m2 += addm0
            bc_clip_sum += bc_clip_add
            clip_sum += clip_add
            u += u_add
            v += v_add
            norm_r += norm_r_add
            norm_g += norm_g_add
            norm_b1 += norm_b_add

            if m0 < 0 or m1 < 0 or m2 < 0:
                if flag:  # 优化：当可以确定超过的是右边界，可以直接换行
                    break
                continue
            flag = True

            frag_depth = clip_sum / bc_clip_sum

            if frag_depth > zbuffer[j, k]:
                continue
            __u, __v = int(u / bc_clip_sum), int(v / bc_clip_sum)
            if __u < 0 or __u >= height or __v < 0 or __v >= width:
                continue
            bn = norm_r / bc_clip_sum, norm_g / bc_clip_sum, norm_b1 / bc_clip_sum
            _g, _h, _i = bn
            _x, _y, _z = _e * _i - _f * _h, _f * _g - _d * _i, _d * _h - _g * _e
            det = _a * _x + _b * _y + _c * _z
            inv0 = _c * _h - _b * _i
            inv1 = _a * _i - _c * _g
            inv2 = _b * _g - _a * _h
            norm__r, norm__g, norm__b = norm_texture[__u, __v]
            vec_u0, vec_u1 = vec_u[0] / det, vec_u[1] / det
            vec_v0, vec_v1 = vec_v[0] / det, vec_v[1] / det
            vec_i = normalized(_x * vec_u0 + inv0 * vec_u1,
                               _y * vec_u0 + inv1 * vec_u1,
                               _z * vec_u0 + inv2 * vec_u1, norm__r)
            vec_j = normalized(_x * vec_v0 + inv0 * vec_v1,
                               _y * vec_v0 + inv1 * vec_v1,
                               _z * vec_v0 + inv2 * vec_v1, norm__g)
            n = normalized1(vec_i[0] + vec_j[0] + bn[0] * norm__b,
                            vec_i[1] + vec_j[1] + bn[1] * norm__b,
                            vec_i[2] + vec_j[2] + bn[2] * norm__b)
            diff = n[0] * uniform_l[0] + n[1] * uniform_l[1] + n[2] * uniform_l[2]
            r = normalized1(n[0] * diff * 2 - uniform_l[0], n[1] * diff * 2 - uniform_l[1],
                            n[2] * diff * 2 - uniform_l[2])
            intensity = max(0, diff) + max(-r[2], 0) ** (5 + spec_texture[__u, __v])
            zbuffer[j, k] = frag_depth
            color = texture[__u, __v]
            screen[j, k] = min(10 + int(color[0] * intensity), 255), min(10 + int(color[1] * intensity), 255), min(10 + int(color[2] * intensity), 255)


@jit(nopython=True, cache=True, fastmath=True)
def generate_faces_new(screen, indices, uv_indices, norm_indices, pts, uv_triangle,
                       norms, view_tri, uniform_l, texture, norm_texture, spec_texture, zbuffer, O2):
    # 使用 z-buffer 算法绘制三角形，以及 phong 着色

    length: int = indices.shape[0]  # 三角形总个数

    for i in prange(length):
        ptsa, ptsb, ptsc = pts[indices[i, 0]], pts[indices[i, 1]], pts[indices[i, 2]]
        norma, normb, normc = norms[norm_indices[i, 0]], norms[norm_indices[i, 1]], norms[norm_indices[i, 2]]
        uva = uv_triangle[uv_indices[i, 0], 0], uv_triangle[uv_indices[i, 0], 1]
        uvb = uv_triangle[uv_indices[i, 1], 0], uv_triangle[uv_indices[i, 1], 1]
        uvc = uv_triangle[uv_indices[i, 2], 0], uv_triangle[uv_indices[i, 2], 1]
        view_tri_a, view_tri_b, view_tri_c = view_tri[indices[i, 0]], view_tri[indices[i, 1]], view_tri[indices[i, 2]]

        nums = (ptsa[3] > -0.1) + (ptsb[3] > -0.1) + (ptsc[3] > -0.1)  # 指有几个点在屏幕外
        if nums:  # 透视裁剪
            if nums == 3:
                continue
            out_vert_num = 0
            out_pts = np.empty((4, 4), dtype=np.float64)
            out_uv = np.empty((4, 2), dtype=np.float64)
            out_norm = np.empty((4, 3), dtype=np.float64)
            out_view_tri = np.empty((4, 3), dtype=np.float64)
            for j in range(3):
                curv_index = j
                prev_index = (j - 1 + 3) % 3
                curv = pts[indices[i, curv_index]]
                prev = pts[indices[i, prev_index]]
                is_cur_inside = curv[3] <= -0.1
                is_pre_inside = prev[3] <= -0.1
                if is_cur_inside != is_pre_inside:
                    ratio = get_intersect_ratio(prev, curv)
                    out_pts[out_vert_num] = lerp_vec(prev, curv, ratio)
                    out_uv[out_vert_num] = lerp_vec(uv_triangle[uv_indices[i, prev_index]],
                                                    uv_triangle[uv_indices[i, curv_index]], ratio)
                    out_norm[out_vert_num] = lerp_vec(norms[norm_indices[i, prev_index]],
                                                      norms[norm_indices[i, curv_index]], ratio)
                    out_view_tri[out_vert_num] = lerp_vec(view_tri[indices[i, prev_index]],
                                                          view_tri[indices[i, curv_index]], ratio)
                    out_vert_num += 1
                if is_cur_inside:
                    out_pts[out_vert_num] = curv
                    out_uv[out_vert_num] = uv_triangle[uv_indices[i, curv_index]]
                    out_norm[out_vert_num] = norms[norm_indices[i, curv_index]]
                    out_view_tri[out_vert_num] = view_tri[indices[i, curv_index]]
                    out_vert_num += 1
            if out_vert_num == 3:
                uva = out_uv[0, 0], out_uv[0, 1]
                uvb = out_uv[1, 0], out_uv[1, 1]
                uvc = out_uv[2, 0], out_uv[2, 1]
                ptsa, ptsb, ptsc = out_pts[0], out_pts[1], out_pts[2]
                norma, normb, normc = out_norm[0], out_norm[1], out_norm[2]
                view_tri_a, view_tri_b, view_tri_c = out_view_tri[0], out_view_tri[1], out_view_tri[2]
            elif out_vert_num == 4:
                uva = out_uv[0, 0], out_uv[0, 1]
                uvb = out_uv[1, 0], out_uv[1, 1]
                uvc = out_uv[2, 0], out_uv[2, 1]
                ptsa, ptsb, ptsc = out_pts[0], out_pts[1], out_pts[2]
                norma, normb, normc = out_norm[0], out_norm[1], out_norm[2]
                view_tri_a, view_tri_b, view_tri_c = out_view_tri[0], out_view_tri[1], out_view_tri[2]
                render_opaque_face(screen, zbuffer, ptsa, ptsb, ptsc, uva, uvb, uvc,
                                   norma, normb, normc, view_tri_a, view_tri_b, view_tri_c, uniform_l,
                                   texture, norm_texture, spec_texture, O2)
                uva = out_uv[0, 0], out_uv[0, 1]
                uvb = out_uv[2, 0], out_uv[2, 1]
                uvc = out_uv[3, 0], out_uv[3, 1]
                ptsa, ptsb, ptsc = out_pts[0], out_pts[2], out_pts[3]
                norma, normb, normc = out_norm[0], out_norm[2], out_norm[3]
                view_tri_a, view_tri_b, view_tri_c = out_view_tri[0], out_view_tri[2], out_view_tri[3]
            else:
                continue

        render_opaque_face(screen, zbuffer, ptsa, ptsb, ptsc, uva, uvb, uvc,
                           norma, normb, normc, view_tri_a, view_tri_b, view_tri_c, uniform_l,
                           texture, norm_texture, spec_texture, O2)
