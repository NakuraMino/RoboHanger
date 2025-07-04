import taichi as ti

import os
import math
import numpy as np
import trimesh
import trimesh.transformations as tra
from typing import Callable, List, Dict, Literal, Tuple
import omegaconf
import scipy.interpolate as interp
import argparse

from robohang.sim.sim_utils import detect_self_intersection


COLOR_BODY = np.array([1., 0., 0.])
COLOR_CHEST = np.array([0., 0., 1.])
COLOR_SLEEVE = np.array([1., 1., 0.])
COLOR_COLLAR_BACK = np.array([0., 1., 0.])
COLOR_COLLAR_FRONT = np.array([1., 0., 1.])


def generate_thetas_for_ellipse(a: float, b: float, theta_lower: float, theta_upper: float, n: int,
                                endpoint=True, super_sample=1000):
    t = np.linspace(theta_lower, theta_upper, num=super_sample)
    x = a * np.cos(t)
    y = b * np.sin(t)
    d = np.zeros(super_sample)
    d[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
    l = np.cumsum(d)
    l /= l[-1]

    theta = interp.interp1d(l, t)(np.linspace(0., 1., n, endpoint=endpoint))
    return theta


class Vertex:
    def __init__(self, pos: np.ndarray, color: np.ndarray) -> None:
        self.pos = pos.copy()
        self.color = color.copy()

    def __hash__(self) -> int:
        return tuple(self.pos.tolist()).__hash__()
    
    def __eq__(self, __value: 'Vertex') -> bool:
        return (self.pos == __value.pos).all()
    
    def __str__(self) -> str:
        return f"<Vert>{self.pos}"
    
    def __repr__(self) -> str:
        return self.__str__()


def generate_mesh(cfg: omegaconf.DictConfig):
    # continuous check
    eps = 1e-5
    for check_pair in [
        (("body", "radius_x", -1), ("chest", "radius_x", 0)),
        (("body", "level_z", -1), ("chest", "level_z", 0)),
        (("chest", "range_x", -1), ("collar", "radius_x", 0)),
        (("chest", "level_z", -1), ("collar", "level_z", 0)),
        (("chest", "split_x", -1), ("collar", "split_xf", 0)),
    ]:
        (s1, s2, i1), (s3, s4, i2) = check_pair
        assert abs(
            getattr(getattr(cfg, s1), s2).points[i1] - getattr(getattr(cfg, s3), s4).points[i2]
        ) < eps, check_pair
    def match_vert_line(v1: List[Vertex], v2: List[Vertex]) -> List[List[Vertex]]:
        i = 0
        j = 0
        n = len(v1)
        m = len(v2)
        ret = []
        while i < n -1 or j < m - 1:
            if j == m - 1:
                ret.append([v1[i], v1[i + 1], v2[j]])
                i += 1
            elif i == n - 1:
                ret.append([v1[i], v2[j + 1], v2[j]])
                j += 1
            else:
                li = np.linalg.norm(v1[i + 1].pos - v2[j].pos)
                lj = np.linalg.norm(v1[i].pos - v2[j + 1].pos)
                flag = np.random.rand() > 0.5
                if li < lj or (li == lj and flag):
                    ret.append([v1[i], v1[i + 1], v2[j]])
                    i += 1
                else:
                    ret.append([v1[i], v2[j + 1], v2[j]])
                    j += 1

        return ret

    def match_vert_circle(v1: List[Vertex], v2: List[Vertex]) -> List[List[Vertex]]:
        return match_vert_line(v1 + [v1[0]], v2 + [v2[0]])

    def make_vert_idx(*args):
        vert_idx: Dict[Vertex, int] = {}
        idx_vert: Dict[int, Vertex] = {}
        for vert_list in args:
            for vert in vert_list:
                idx = len(vert_idx)
                vert_idx[vert] = idx
                idx_vert[idx] = vert
        return vert_idx, idx_vert

    def generate_face_type1(vert: List[List[Vertex]], vert_idx: Dict[Vertex, int], match_func: Callable) -> List[List[int]]:
        n = len(vert)
        faces = []
        for i, j in zip(range(n-1), range(1, n)):
            faces_list = match_func(vert[i], vert[j])
            for f in faces_list:
                faces.append([vert_idx[f[0]], vert_idx[f[1]], vert_idx[f[2]]])
        return faces
    
    def generate_face_type2(vert_a: List[Vertex], vert_b: List[Vertex], vert_idx: Dict[Vertex, int], match_func: Callable) -> List[List[int]]:
        faces = []
        faces_list = match_func(vert_a, vert_b)
        for f in faces_list:
            faces.append([vert_idx[f[0]], vert_idx[f[1]], vert_idx[f[2]]])
        return faces

    def generate_vert_ax(cfg_body):
        n_horizon = (int(cfg_body.n_horizon) + 1) // 2 * 2
        level_z_list = []
        segment_acc = [0]
        for idx, segment in enumerate(cfg_body.segment):
            z = np.linspace(cfg_body.level_z.points[idx], 
                            cfg_body.level_z.points[idx + 1],
                            segment, endpoint=False)
            level_z_list += z.tolist()
            segment_acc.append(segment_acc[-1] + segment)
        level_z_list += [cfg_body.level_z.points[-1]]
        radius_x_list = interp.interp1d(segment_acc,
                                        cfg_body.radius_x.points,
                                        kind=cfg_body.radius_x.interp)(np.arange(segment_acc[-1] + 1))
        radius_y_list = interp.interp1d(segment_acc,
                                        cfg_body.radius_y.points,
                                        kind=cfg_body.radius_y.interp)(np.arange(segment_acc[-1] + 1))
        
        vert_ax: List[List[Vertex]] = []
        for level_z, radius_x, radius_y in zip(level_z_list, radius_x_list, radius_y_list):
            theta = generate_thetas_for_ellipse(radius_x, radius_y, 0., np.pi * 2., n_horizon, endpoint=False)
            v = np.zeros((n_horizon, 3))
            v[:, 0] = radius_x * np.cos(theta)
            v[:, 1] = radius_y * np.sin(theta)
            v[:, 2] = level_z
            vert_ax.append([Vertex(pos, COLOR_BODY) for pos in v])
        vert_a = vert_ax[:-1]
        vert_x = vert_ax[-1]

        left_vert_idx = np.argmin(np.array([v.pos[0] for v in vert_x]))
        right_vert_idx = np.argmax(np.array([v.pos[0] for v in vert_x]))

        vert_f = []
        vert_b = [vert_x[right_vert_idx]]
        vert_idx = right_vert_idx + 1
        curr_state = "back"
        while vert_idx != right_vert_idx:
            if curr_state == "back":
                vert_b.append(vert_x[vert_idx])
            else:
                vert_f.append(vert_x[vert_idx])
            vert_idx = (vert_idx + 1) % len(vert_x)
            if vert_idx == left_vert_idx:
                vert_b.append(vert_x[vert_idx])
                curr_state = "front"
        vert_f.append(vert_x[right_vert_idx])

        keyv_a_l = vert_a[0][n_horizon // 2]
        keyv_a_r = vert_a[0][0]
        keyv_x_l = vert_x[n_horizon // 2]
        keyv_x_r = vert_x[0]
        return vert_a, vert_x, vert_x[left_vert_idx], vert_x[right_vert_idx], vert_f, vert_b, keyv_a_l, keyv_a_r, keyv_x_l, keyv_x_r

    def generate_vert_cf(cfg_chest, max_range=0.999, mid_eps=0.001):
        n_horizon = cfg_chest.n_horizon
        segment_acc = [0]
        for idx, segment in enumerate(cfg_chest.segment):
            segment_acc.append(segment_acc[-1] + segment)

        level_z_list = interp.interp1d(segment_acc,
                                       cfg_chest.level_z.points,
                                       kind=cfg_chest.level_z.interp)(np.arange(segment_acc[-1] + 1))
        range_x_list = interp.interp1d(segment_acc,
                                       cfg_chest.range_x.points,
                                       kind=cfg_chest.range_x.interp)(np.arange(segment_acc[-1] + 1))
        split_x_list = interp.interp1d(segment_acc,
                                       cfg_chest.split_x.points,
                                       kind=cfg_chest.split_x.interp)(np.arange(segment_acc[-1] + 1))
        radius_x_list = interp.interp1d(segment_acc,
                                        cfg_chest.radius_x.points,
                                        kind=cfg_chest.radius_x.interp)(np.arange(segment_acc[-1] + 1))
        radius_y_list = interp.interp1d(segment_acc,
                                        cfg_chest.radius_y.points,
                                        kind=cfg_chest.radius_y.interp)(np.arange(segment_acc[-1] + 1))

        theta_lower_list = np.pi + np.arccos(np.clip(range_x_list, a_max=radius_x_list * max_range, a_min=-radius_x_list * max_range) / radius_x_list)
        theta_upper_list = np.pi * 2 - np.arccos(np.clip(range_x_list, a_max=radius_x_list * max_range, a_min=-radius_x_list * max_range) / radius_x_list)
        theta_mid_l_list = np.pi + np.arccos(split_x_list / radius_x_list)
        theta_mid_r_list = np.pi * 2 - np.arccos(split_x_list / radius_x_list)

        vert_cf: List[List[Vertex]] = []
        split_pairs = []
        for idx, (level_z, theta_lower, theta_mid_l, theta_mid_r, split_x, theta_upper, radius_x, radius_y) in \
            enumerate(zip(level_z_list, theta_lower_list, theta_mid_l_list, theta_mid_r_list, split_x_list, theta_upper_list, radius_x_list, radius_y_list)):
            if idx == 0 and cfg_chest.remove_first:
                continue
            if np.abs(split_x * 2) < mid_eps:
                theta = generate_thetas_for_ellipse(radius_x, radius_y, theta_lower, theta_upper, n_horizon)
                is_split = False
            else:
                n_horizon_half = (int(n_horizon) + 1) // 2
                n_horizon = n_horizon_half * 2
                theta = np.concatenate([
                    generate_thetas_for_ellipse(radius_x, radius_y, theta_lower, theta_mid_l, n_horizon_half),
                    generate_thetas_for_ellipse(radius_x, radius_y, theta_mid_r, theta_upper, n_horizon_half),
                ])
                is_split = True

            v = np.zeros((n_horizon, 3))
            v[:, 0] = radius_x * np.cos(theta)
            v[:, 1] = radius_y * np.sin(theta)
            v[:, 2] = level_z
            vert_cf.append([Vertex(pos, COLOR_CHEST) for pos in v])
            if is_split:
                split_pairs.append((v[n_horizon_half - 1, :], v[n_horizon_half, :]))

        vert_cf_left = [vert[0] for vert in vert_cf]
        vert_cf_right = [vert[-1] for vert in vert_cf]
        return vert_cf, vert_cf_left, vert_cf_right, split_pairs

    def generate_vert_cb(cfg_chest, max_range=0.999):
        n_horizon = cfg_chest.n_horizon
        segment_acc = [0]
        for idx, segment in enumerate(cfg_chest.segment):
            segment_acc.append(segment_acc[-1] + segment)

        level_z_list = interp.interp1d(segment_acc,
                                       cfg_chest.level_z.points,
                                       kind=cfg_chest.level_z.interp)(np.arange(segment_acc[-1] + 1))
        range_x_list = interp.interp1d(segment_acc,
                                       cfg_chest.range_x.points,
                                       kind=cfg_chest.range_x.interp)(np.arange(segment_acc[-1] + 1))
        radius_x_list = interp.interp1d(segment_acc,
                                        cfg_chest.radius_x.points,
                                        kind=cfg_chest.radius_x.interp)(np.arange(segment_acc[-1] + 1))
        radius_y_list = interp.interp1d(segment_acc,
                                        cfg_chest.radius_y.points,
                                        kind=cfg_chest.radius_y.interp)(np.arange(segment_acc[-1] + 1))

        theta_lower_list = np.arccos(np.clip(range_x_list, a_max=radius_x_list * max_range, a_min=-radius_x_list * max_range) / radius_x_list)
        theta_upper_list = np.pi - np.arccos(np.clip(range_x_list, a_max=radius_x_list * max_range, a_min=-radius_x_list * max_range) / radius_x_list)

        vert_cb: List[List[Vertex]] = []
        for idx, (level_z, theta_lower, theta_upper, radius_x, radius_y) in \
            enumerate(zip(level_z_list, theta_lower_list, theta_upper_list, radius_x_list, radius_y_list)):
            if idx == 0 and cfg_chest.remove_first:
                continue
            theta = generate_thetas_for_ellipse(radius_x, radius_y, theta_lower, theta_upper, n_horizon)
            v = np.zeros((n_horizon, 3))
            v[:, 0] = radius_x * np.cos(theta)
            v[:, 1] = radius_y * np.sin(theta)
            v[:, 2] = level_z
            vert_cb.append([Vertex(pos, COLOR_CHEST) for pos in v])

        vert_cb_left = [vert[-1] for vert in vert_cb]
        vert_cb_right = [vert[0] for vert in vert_cb]
        return vert_cb, vert_cb_left, vert_cb_right

    def generate_vert_b(cfg_sleeve, side: Literal["left", "right"]):
        sign = -1.0 if side == "left" else +1.0
        n_horizon = (int(cfg_sleeve.n_horizon) + 1) // 2 * 2
        segment_acc = [0]
        for idx, segment in enumerate(cfg_sleeve.segment):
            segment_acc.append(segment_acc[-1] + segment)
        radius_x_list = interp.interp1d(segment_acc,
                                        cfg_sleeve.radius_x.points,
                                        kind=cfg_sleeve.radius_x.interp)(np.arange(segment_acc[-1] + 1))
        radius_y_list = interp.interp1d(segment_acc,
                                        cfg_sleeve.radius_y.points,
                                        kind=cfg_sleeve.radius_y.interp)(np.arange(segment_acc[-1] + 1))
        angle_y_list = interp.interp1d(segment_acc,
                                       cfg_sleeve.angle_y.points,
                                       kind=cfg_sleeve.angle_y.interp)(np.arange(segment_acc[-1] + 1)) * sign
        level_x_list = interp.interp1d(segment_acc,
                                       cfg_sleeve.level_x.points,
                                       kind=cfg_sleeve.level_x.interp)(np.arange(segment_acc[-1] + 1)) * sign
        level_z_list = interp.interp1d(segment_acc,
                                       cfg_sleeve.level_z.points,
                                       kind=cfg_sleeve.level_z.interp)(np.arange(segment_acc[-1] + 1))
        level_y_list = [0.] * (segment_acc[-1] + 1)

        vert_bl: List[List[Vertex]] = []
        for idx, (angle_y, radius_x, radius_y, level_x, level_y, level_z) in \
            enumerate(zip(angle_y_list, radius_x_list, radius_y_list, level_x_list, level_y_list, level_z_list)):
            if idx == 0 and cfg_sleeve.remove_first:
                continue
            theta = generate_thetas_for_ellipse(radius_x, radius_y, 0., np.pi * 2., n_horizon, endpoint=False)

            v = np.zeros((n_horizon, 3))
            v[:, 0] = radius_x * np.cos(theta)
            v[:, 1] = radius_y * np.sin(theta)
            v = tra.rotation_matrix(np.deg2rad(angle_y), [0., +1., 0.])[None, :3, :3] @ v[:, :, None]
            v[:, :] += np.array([level_x, level_y, level_z])[None, :, None]

            vert_bl.append([Vertex(pos[:3, 0], COLOR_SLEEVE) for pos in v])

        vert_end = vert_bl[-1][0 if side == "left" else (n_horizon // 2)] 
        return vert_bl, vert_end

    def generate_vert_yd(cfg_collar, mid_eps=0.001):
        subdivide_edges: List[Tuple[Vertex, Vertex]] = []

        segment_acc = [0]
        for idx, segment in enumerate(cfg_collar.segment):
            segment_acc.append(segment_acc[-1] + segment)
        radius_x_list = interp.interp1d(segment_acc,
                                        cfg_collar.radius_x.points,
                                        kind=cfg_collar.radius_x.interp)(np.arange(segment_acc[-1] + 1))
        angle_xb_list = interp.interp1d(segment_acc,
                                        cfg_collar.angle_xb.points,
                                        kind=cfg_collar.angle_xb.interp)(np.arange(segment_acc[-1] + 1))
        angle_xf_list = interp.interp1d(segment_acc,
                                        cfg_collar.angle_xf.points,
                                        kind=cfg_collar.angle_xf.interp)(np.arange(segment_acc[-1] + 1))
        split_xf_list = interp.interp1d(segment_acc,
                                        cfg_collar.split_xf.points,
                                        kind=cfg_collar.split_xf.interp)(np.arange(segment_acc[-1] + 1))
        level_z_list = interp.interp1d(segment_acc,
                                       cfg_collar.level_z.points,
                                       kind=cfg_collar.level_z.interp)(np.arange(segment_acc[-1] + 1))
        radius_yb_list = interp.interp1d(segment_acc,
                                         cfg_collar.radius_yb.points,
                                         kind=cfg_collar.radius_yb.interp)(np.arange(segment_acc[-1] + 1))
        radius_yf_list = interp.interp1d(segment_acc,
                                         cfg_collar.radius_yf.points,
                                         kind=cfg_collar.radius_yf.interp)(np.arange(segment_acc[-1] + 1))
        n_horizon_b_list = interp.interp1d(segment_acc,
                                           cfg_collar.n_horizon_b.points,
                                           kind=cfg_collar.n_horizon_b.interp)(np.arange(segment_acc[-1] + 1)).astype(int)
        n_horizon_f_list = interp.interp1d(segment_acc,
                                           cfg_collar.n_horizon_f.points,
                                           kind=cfg_collar.n_horizon_f.interp)(np.arange(segment_acc[-1] + 1)).astype(int)

        level_x_list = [0.] * (segment_acc[-1] + 1)
        level_y_list = [0.] * (segment_acc[-1] + 1)
    
        vert_yd: List[List[Vertex]] = []
        vert_d_f: List[List[Vertex]] = []
        vert_d_b: List[List[Vertex]] = []
        split_pairs = []
        for idx, (angle_xb, angle_xf, split_xf, radius_x, radius_yb, radius_yf, level_x, level_y, level_z, n_horizon_b, n_horizon_f) in \
            enumerate(zip(angle_xb_list, angle_xf_list, split_xf_list, radius_x_list, radius_yb_list, radius_yf_list, level_x_list, level_y_list, level_z_list, n_horizon_b_list, n_horizon_f_list)):
            if idx == 0 and cfg_collar.remove_first:
                continue

            theta_back = generate_thetas_for_ellipse(radius_x, radius_yb, 0., np.pi, n_horizon_b, endpoint=False)
            if np.abs(split_xf * 2) < mid_eps:
                n_horizon_f_half = (int(n_horizon_f) + 1) // 2
                n_horizon_f = n_horizon_f_half * 2
                theta_front = generate_thetas_for_ellipse(radius_x, radius_yf, np.pi, np.pi * 2, n_horizon_f, endpoint=False)
                is_split = False
            else:
                n_horizon_f_half = (int(n_horizon_f) + 1) // 2
                n_horizon_f = n_horizon_f_half * 2 - 1
                theta_front_1 = generate_thetas_for_ellipse(radius_x, radius_yf, np.pi, np.arccos(split_xf / radius_x) + np.pi, n_horizon_f_half, endpoint=True)
                theta_front_2 = generate_thetas_for_ellipse(radius_x, radius_yf, np.pi * 2 - np.arccos(split_xf / radius_x), np.pi * 2, n_horizon_f_half - 1, endpoint=False)
                theta_front = np.concatenate([theta_front_1, theta_front_2])
                is_split = True

            theta = np.concatenate([theta_back, theta_front])
            # theta = np.arange(n_horizon) / n_horizon * np.pi * 2
            v = np.zeros((n_horizon_b + n_horizon_f, 3))
            v[:, 0] = radius_x * np.cos(theta)
            v[:, 1] = radius_yb * np.sin(theta) * (np.sin(theta) > 0.) + radius_yf * np.sin(theta) * ~(np.sin(theta) > 0.)
            v = (
                tra.rotation_matrix(np.deg2rad(angle_xb), [1., 0., 0.])[None, :3, :3] @ v[:, :, None] * (np.sin(theta) > 0.)[:, None, None] + 
                tra.rotation_matrix(np.deg2rad(angle_xf), [1., 0., 0.])[None, :3, :3] @ v[:, :, None] * ~(np.sin(theta) > 0.)[:, None, None]
            )
            v[:, :] += np.array([level_x, level_y, level_z])[None, :, None]

            vert_yd.append([Vertex(pos[:3, 0], COLOR_COLLAR_BACK if vert_idx <= n_horizon_b else COLOR_COLLAR_FRONT) for vert_idx, pos in enumerate(v)])
            vert_d_b.append([])
            vert_d_f.append([])
            for vert_idx, vert in enumerate(vert_yd[-1]):
                if vert_idx <= n_horizon_b:
                    vert_d_b[-1].append(vert)
                else:
                    vert_d_f[-1].append(vert)

            if is_split:
                split_pairs.append((v[n_horizon_b + n_horizon_f_half - 1, :, 0], v[n_horizon_b + n_horizon_f_half, :, 0]))

            vert_d_r = vert_yd[-1][0]
            vert_d_l = vert_yd[-1][n_horizon_b]

        vert_d = vert_yd[1:]
        vert_y = vert_yd[0]

        left_vert_idx = np.argmin(np.array([v.pos[0] for v in vert_y]))
        right_vert_idx = np.argmax(np.array([v.pos[0] for v in vert_y]))

        vert_f = []
        vert_b = [vert_y[right_vert_idx]]
        vert_idx = right_vert_idx + 1
        curr_state = "back"
        while vert_idx != right_vert_idx:
            if curr_state == "back":
                vert_b.append(vert_y[vert_idx])
            else:
                vert_f.append(vert_y[vert_idx])
            vert_idx = (vert_idx + 1) % len(vert_y)
            if vert_idx == left_vert_idx:
                vert_b.append(vert_y[vert_idx])
                curr_state = "front"
        vert_f.append(vert_y[right_vert_idx])

        return vert_d, vert_y, vert_y[left_vert_idx], vert_y[right_vert_idx], vert_f, vert_b, split_pairs, vert_d_l, vert_d_r, vert_d_b, vert_d_f
    
    def delete_faces(faces: np.ndarray, vertices: np.ndarray, split_pairs: List[Tuple[np.ndarray, np.ndarray]]):
        faces_is_delete = []
        for face in faces:
            is_delete = False
            p0, p1, p2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            for v0, v1 in split_pairs:
                if (((v0 == p0).all() or (v0 == p1).all() or (v0 == p2).all()) and
                    ((v1 == p0).all() or (v1 == p1).all() or (v1 == p2).all())):
                    is_delete = True
            faces_is_delete.append(is_delete) 
        new_faces = faces[np.where(np.logical_not(np.array(faces_is_delete)))]
        return new_faces, vertices
    
    def reconnect(v1id: int, v2id: int, faces: np.ndarray):
        v1_face_loop = []
        v2_face_loop = []
        for fid, (v0, v1, v2) in enumerate(faces):
            if v1id in (v0, v1, v2) and v2id in (v0, v1, v2):
                return faces # don't do anything
            if v1id in (v0, v1, v2) and v2id not in (v0, v1, v2):
                v1_face_loop.append(fid)
            if v1id not in (v0, v1, v2) and v2id in (v0, v1, v2):
                v2_face_loop.append(fid)
        def find_face_pair():
            for f1id in v1_face_loop:
                for f2id in v2_face_loop:
                    common_v = []
                    for v in faces[f1id]:
                        if v in faces[f2id]:
                            common_v.append(v)
                    if len(common_v) == 2:
                        return f1id, f2id, common_v
        f1id, f2id, common_v = find_face_pair()
        f1_verts = faces[f1id]
        f1_verts[np.where(f1_verts == common_v[0])] = v2id
        faces[f1id] = f1_verts
        f2_verts = faces[f2id]
        f2_verts[np.where(f2_verts == common_v[1])] = v1id
        faces[f2id] = f2_verts
        return faces
    
    def split_one_edge(v1id: int, v2id: int, vertices: np.ndarray, faces: np.ndarray, vertices_color: np.ndarray):
        assert v1id != v2id
        faces = reconnect(v1id, v2id, faces)
        v3id = None
        v4id = None
        for fid, (v0, v1, v2) in enumerate(faces):
            if (
                v0 == v1id and v1 == v2id or
                v1 == v1id and v2 == v2id or
                v2 == v1id and v0 == v2id
            ):
                f1id = fid
                if v0 == v1id and v1 == v2id:
                    v3id = v2
                elif v1 == v1id and v2 == v2id:
                    v3id = v0
                else:
                    v3id = v1
            elif (
                v0 == v2id and v1 == v1id or
                v1 == v2id and v2 == v1id or
                v2 == v2id and v0 == v1id
            ):
                f2id = fid
                if v0 == v2id and v1 == v1id:
                    v4id = v2
                elif v1 == v2id and v2 == v1id:
                    v4id = v0
                else:
                    v4id = v1
        vertices = vertices.copy()
        faces = faces.copy()
        vertices_color = vertices_color.copy()
        v5id = len(vertices)
        vertices = np.concatenate([vertices, (vertices[v1id] / 2 + vertices[v2id] / 2)[None, :]], axis=0)
        vertices_color = np.concatenate([vertices_color, (vertices_color[v1id] / 2 + vertices_color[v2id] / 2)[None, :]], axis=0)
        faces[f1id] = np.array([v1id, v5id, v3id])
        faces[f2id] = np.array([v1id, v4id, v5id])
        faces = np.concatenate([faces, np.array([[v5id, v2id, v3id], [v5id, v4id, v2id]])], axis=0)
        return vertices, faces, vertices_color

    def assemble():
        vert_a, vert_x, vert_x_l, vert_x_r, vert_xf, vert_xb, keyv_a_l, keyv_a_r, keyv_x_l, keyv_x_r = generate_vert_ax(cfg.body)
        vert_d, vert_y, vert_y_l, vert_y_r, vert_yf, vert_yb, split_pairs_collar, vert_d_l, vert_d_r, vert_d_b, vert_d_f = generate_vert_yd(cfg.collar)
        vert_cf, vert_cf_l, vert_cf_r, split_pairs_chest = generate_vert_cf(cfg.chest)
        vert_cb, vert_cb_l, vert_cb_r = generate_vert_cb(cfg.chest)
        vert_bl, vert_b_le = generate_vert_b(cfg.sleeve, "left")
        vert_br, vert_b_re = generate_vert_b(cfg.sleeve, "right")
        
        # vert_z_l = [vert_y_l] + [vert_cb_l[i] for i in range(len(vert_cb_l) - 1, -1, -1)] + [vert_x_l] + vert_cf_l
        # vert_z_r = [vert_x_r] + vert_cb_r + [vert_y_r] + [vert_cf_r[i] for i in range(len(vert_cf_r) - 1, -1, -1)]
        vert_z_l = [vert_cb_l[i] for i in range(len(vert_cb_l) - 1, -1, -1)] + vert_cf_l
        vert_z_r = vert_cb_r + [vert_cf_r[i] for i in range(len(vert_cf_r) - 1, -1, -1)]

        vert_idx, idx_vert = make_vert_idx(*(vert_a + [vert_x] + [vert_y] + vert_cf + vert_cb + vert_bl + vert_br + vert_d))
        vertices = np.array([idx_vert[i].pos for i in range(len(vert_idx))])
        faces = np.array(
            generate_face_type1(vert_a, vert_idx, match_vert_circle) + 
            generate_face_type2(vert_a[-1], vert_x, vert_idx, match_vert_circle) +
            generate_face_type1(vert_cf, vert_idx, match_vert_line) +
            generate_face_type1(vert_cb, vert_idx, match_vert_line) + 
            generate_face_type2(vert_xf, vert_cf[0], vert_idx, match_vert_line) + 
            generate_face_type2(vert_xb, vert_cb[0], vert_idx, match_vert_line) +
            generate_face_type2(vert_cf[-1], vert_yf, vert_idx, match_vert_line) + 
            generate_face_type2(vert_cb[-1], vert_yb, vert_idx, match_vert_line) +
            generate_face_type1(vert_bl, vert_idx, match_vert_circle) + 
            generate_face_type1(vert_br, vert_idx, match_vert_circle) +
            generate_face_type2(vert_z_l, vert_bl[0], vert_idx, match_vert_circle) + 
            generate_face_type2(vert_z_r, vert_br[0], vert_idx, match_vert_circle) + 
            generate_face_type1(vert_d, vert_idx, match_vert_circle) + 
            generate_face_type2(vert_y, vert_d[0], vert_idx, match_vert_circle) + [
                [vert_idx[v0], vert_idx[v1], vert_idx[v2]] for v0, v1, v2 in 
                [
                    [vert_x_l, vert_cf_l[0], vert_cb_l[0]],
                    [vert_x_r, vert_cb_r[0], vert_cf_r[0]],
                    [vert_y_l, vert_cb_l[-1], vert_cf_l[-1]],
                    [vert_y_r, vert_cf_r[-1], vert_cb_r[-1]],
                ]
            ]
        )

        key_points = dict(
            upper_left=vert_idx[vert_d_l],
            upper_right=vert_idx[vert_d_r],
            lower_left=vert_idx[keyv_a_l],
            lower_right=vert_idx[keyv_a_r],
            armpit_left=vert_idx[keyv_x_l],
            armpit_right=vert_idx[keyv_x_r],
            sleeve_left=vert_idx[vert_b_le],
            sleeve_right=vert_idx[vert_b_re],
        )
        # faces = np.concatenate([faces, faces[:, [0, 2, 1]]], axis=0)
        faces, vertices = delete_faces(faces, vertices, split_pairs_chest + split_pairs_collar)
        vertices_color = np.array([idx_vert[i].color for i in range(len(vert_idx))])
        for vid in key_points.values():
            vertices_color[vid] = np.array([1., 1., 1.])

        # subdivide mesh
        if True:
            for vb_idx in range(len(vert_d_b) - 1):
                vf_idx = vb_idx
                
                vb1, vb2 = vert_d_b[vb_idx], vert_d_b[vb_idx + 1]
                vf1, vf2 = vert_d_f[vf_idx], vert_d_f[vf_idx + 1]
                vertices, faces, vertices_color = split_one_edge(vert_idx[vb1[0]], vert_idx[vb2[0]], vertices, faces, vertices_color)
                vertices, faces, vertices_color = split_one_edge(vert_idx[vb1[1]], vert_idx[vb2[1]], vertices, faces, vertices_color)
                vertices, faces, vertices_color = split_one_edge(vert_idx[vf1[-1]], vert_idx[vf2[-1]], vertices, faces, vertices_color)
                faces = reconnect(len(vertices) - 3, len(vertices) - 2, faces)
                faces = reconnect(len(vertices) - 3, len(vertices) - 1, faces)

                vertices, faces, vertices_color = split_one_edge(vert_idx[vb1[-1]], vert_idx[vb2[-1]], vertices, faces, vertices_color)
                vertices, faces, vertices_color = split_one_edge(vert_idx[vb1[-2]], vert_idx[vb2[-2]], vertices, faces, vertices_color)
                vertices, faces, vertices_color = split_one_edge(vert_idx[vf1[0]], vert_idx[vf2[0]], vertices, faces, vertices_color)
                faces = reconnect(len(vertices) - 3, len(vertices) - 2, faces)
                faces = reconnect(len(vertices) - 3, len(vertices) - 1, faces)
                
        mesh = trimesh.Trimesh(
            vertices=vertices, 
            faces=faces, 
            vertex_colors=vertices_color, 
        )
        mesh.apply_transform(eval(cfg.transform))

        return mesh, key_points
    
    return assemble()


def export_meta_data(cfg: omegaconf.DictConfig, key_points: dict, mesh_file_path: str):
    meta_data = omegaconf.DictConfig(dict())
    meta_data.cfg = cfg
    meta_data.key_points = key_points
    omegaconf.OmegaConf.save(meta_data, mesh_file_path + ".meta.yaml")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/asset/generate_cloth_mesh.yaml")
    parser.add_argument("-o", "--output", type=str, default="./assets/clothes/test/test.obj")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = omegaconf.OmegaConf.load(args.config)
    mesh_file_path = args.output

    ti.init(ti.cpu, default_fp=ti.f64, debug=False, fast_math=False)
    mesh, key_points = generate_mesh(cfg)
    print(mesh.vertices.shape, mesh.faces.shape)
    si, intersection_mask = detect_self_intersection(mesh)
    fids, eids = np.where(intersection_mask)
    annotates = []
    for fid in fids:
        for i in range(3):
            xyz = mesh.vertices[mesh.faces[fid, i]]
            annotates.append(trimesh.primitives.Sphere(.01, xyz))
    for eid in eids:
        for i in range(2):
            xyz = mesh.vertices[mesh.edges[eid, i]]
            annotates.append(trimesh.primitives.Box([.008] * 3, tra.translation_matrix(xyz)))
    for k, v in key_points.items():
        xyz = mesh.vertices[v]
        print(k, xyz)
        # annotates.append(trimesh.primitives.Sphere(.01, xyz))
    print(f"self intersection [fid, eid]:\n{fids, eids}")
    
    os.makedirs(os.path.dirname(mesh_file_path), exist_ok=True)
    trimesh.util.concatenate(annotates + [mesh]).export(mesh_file_path)
    export_meta_data(cfg, key_points, mesh_file_path)


if __name__ == "__main__":
    main()