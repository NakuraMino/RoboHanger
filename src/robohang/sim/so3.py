import taichi as ti

from typing import Tuple
import torch
from . import sim_utils


@ti.func
def mat3_to_quat_func(mat3: ti.math.mat3) -> ti.math.vec4:
    m = mat3
    p = ti.Vector.zero(float, 4)
    t = m.trace()
    if t > 0.:
        p[0] = t + 1.
        p[3] = m[1, 0] - m[0, 1]
        p[2] = m[0, 2] - m[2, 0]
        p[1] = m[2, 1] - m[1, 2]
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            i = ti.static(0)
            j = ti.static(1)
            k = ti.static(2)
            t = m[i, i] - (m[j, j] + m[k, k]) + 1.
            p[i] = t
            p[j] = m[i, j] + m[j, i]
            p[k] = m[k, i] + m[i, k]
            p[3] = m[k, j] - m[j, k]
        elif m[1, 1] > m[0, 0] and m[1, 1] > m[2, 2]:
            i = ti.static(1)
            j = ti.static(2)
            k = ti.static(0)
            t = m[i, i] - (m[j, j] + m[k, k]) + 1.
            p[i] = t
            p[j] = m[i, j] + m[j, i]
            p[k] = m[k, i] + m[i, k]
            p[3] = m[k, j] - m[j, k]
        else:
            i = ti.static(2)
            j = ti.static(0)
            k = ti.static(1)
            t = m[i, i] - (m[j, j] + m[k, k]) + 1.
            p[i] = t
            p[j] = m[i, j] + m[j, i]
            p[k] = m[k, i] + m[i, k]
            p[3] = m[k, j] - m[j, k]
        
        p[0], p[1], p[2], p[3] = p[3], p[0], p[1], p[2]
    p /= p.norm()
    if p[0] < 0.:
        p *= -1.
    return p


@ti.func
def quat_to_mat3_func(q: ti.math.vec4) -> ti.math.mat3:
    m = ti.Matrix.identity(float, 3)
    r, i, j, k = q
    q_norm_sqr = q.norm_sqr()
    two_s = 0.
    if q_norm_sqr > 0.:
        two_s = 2. / q_norm_sqr

    m[0, 0] = 1. - two_s * (j * j + k * k)
    m[0, 1] = two_s * (i * j - k * r)
    m[0, 2] = two_s * (i * k + j * r)
    m[1, 0] = two_s * (i * j + k * r)
    m[1, 1] = 1. - two_s * (i * i + k * k)
    m[1, 2] = two_s * (j * k - i * r)
    m[2, 0] = two_s * (i * k - j * r)
    m[2, 1] = two_s * (j * k + i * r)
    m[2, 2] = 1. - two_s * (i * i + j * j)

    return m


@ti.func
def matrix_to_pos7d_func(mat4: ti.math.mat4):
    p = ti.Vector.zero(float, 7)
    p[:3] = mat4[:3, 3]
    p[3:7] = mat3_to_quat_func(mat4[:3, :3])
    return p


@ti.kernel
def matrix_to_pos7d_kernel(m: ti.types.ndarray(dtype=ti.math.mat4), p: ti.types.ndarray(dtype=sim_utils.vec7)):
    for b in range(m.shape[0]):
        p[b][:] = matrix_to_pos7d_func(m[b])


def matrix_to_pos7d(m: torch.Tensor):
    """
    Input:
        - t: [B, 4, 4]
        
    Output:
        - m: [B, 7]
    """
    B, X, Y = m.shape
    assert X == 4 and Y == 4

    r = torch.zeros((B, 7), device=m.device, dtype=m.dtype)
    matrix_to_pos7d_kernel(m, r)
    return r


@ti.func
def pos7d_to_matrix_func(p: sim_utils.vec7) -> ti.math.mat4:
    m = ti.Matrix.identity(float, 4)
    m[:3, 3] = p[:3]
    m[:3, :3] = quat_to_mat3_func(p[3:7])
    return m


@ti.kernel
def pos7d_to_matrix_kernel(m: ti.types.ndarray(dtype=ti.math.mat4), p: ti.types.ndarray(dtype=sim_utils.vec7)):
    for b in range(m.shape[0]):
        m[b][:, :] = pos7d_to_matrix_func(p[b])


def pos7d_to_matrix(pos7d: torch.Tensor) -> torch.Tensor:
    B, D = pos7d.shape
    assert D == 7
    dtype, device = pos7d.dtype, pos7d.device
    M = sim_utils.create_zero_4x4_eye(B, dtype, device)
    pos7d_to_matrix_kernel(M, pos7d)
    return M


@ti.func
def pos7d_to_matinv_func(p: sim_utils.vec7) -> ti.math.mat4:
    r, i, j, k = p[3:7]
    q_norm_sqr = p[3:7].norm_sqr()
    two_s = 0.
    if q_norm_sqr > 0.:
        two_s = 2. / q_norm_sqr

    minv = ti.Matrix.identity(float, 4)
    minv[0, 0] = 1. - two_s * (j * j + k * k)
    minv[1, 0] = two_s * (i * j - k * r)
    minv[2, 0] = two_s * (i * k + j * r)
    minv[0, 1] = two_s * (i * j + k * r)
    minv[1, 1] = 1. - two_s * (i * i + k * k)
    minv[2, 1] = two_s * (j * k - i * r)
    minv[0, 2] = two_s * (i * k - j * r)
    minv[1, 2] = two_s * (j * k + i * r)
    minv[2, 2] = 1. - two_s * (i * i + j * j)

    minv[:3, 3] = -minv[:3, :3] @ p[:3]
    return minv


@ti.kernel
def pos7d_to_matinv_kernel(m: ti.types.ndarray(dtype=ti.math.mat4), p: ti.types.ndarray(dtype=sim_utils.vec7)):
    for b in range(m.shape[0]):
        m[b][:, :] = pos7d_to_matinv_func(p[b])


def pos7d_to_matinv(pos7d: torch.Tensor) -> torch.Tensor:
    B, D = pos7d.shape
    assert D == 7
    dtype, device = pos7d.dtype, pos7d.device
    M = sim_utils.create_zero_4x4_eye(B, dtype, device)
    pos7d_to_matinv_kernel(M, pos7d)
    return M


@ti.func
def axis_angle_to_matrix_func(axis_angle: ti.math.vec3):
    """return mat3"""
    mat = ti.Matrix.identity(float, 3)
    angle = axis_angle.norm()
    if angle > 0.:
        direction = axis_angle / angle
        cosa = ti.cos(angle)
        sina = ti.sin(angle)
        mat[0, 0] = mat[1, 1] = mat[2, 2] = cosa
        mat += direction.outer_product(direction) * (1. - cosa)
        d = direction * sina
        mat += ti.Matrix([
            [0, -d[2], +d[1]],
            [+d[2], 0, -d[0]],
            [-d[1], +d[0], 0],
        ], float)
    return mat


@ti.kernel
def axis_angle_to_matrix_kernel(aa: ti.types.ndarray(dtype=ti.math.vec3), mat: ti.types.ndarray(dtype=ti.math.mat3)):
    for b in range(mat.shape[0]):
        mat[b][:, :] = axis_angle_to_matrix_func(aa[b])


@ti.func
def axis_angle_to_dmat_func(axis_angle: ti.math.vec3):
    """Return mat3, mat3, mat3"""
    dmatx = ti.Matrix.zero(float, 3, 3)
    dmaty = ti.Matrix.zero(float, 3, 3)
    dmatz = ti.Matrix.zero(float, 3, 3)
    d_ = ti.Matrix.identity(float, 3)
    angle = axis_angle.norm()
    if angle > 0.:
        direction = axis_angle / angle
        op_mat = direction.outer_product(direction)
        angle_ = direction[:] # [3]
        direction_ = (ti.Matrix.identity(float, 3) - op_mat) / angle # [3, 3]
        cosa = ti.cos(angle)
        sina = ti.sin(angle)
        cosa_ = -angle_ * sina # [3]
        sina_ = angle_ * cosa # [3]
        # mat[0, 0] = mat[1, 1] = mat[2, 2] = cosa
        dmatx[0, 0] = dmatx[1, 1] = dmatx[2, 2] = cosa_[0]
        dmaty[0, 0] = dmaty[1, 1] = dmaty[2, 2] = cosa_[1]
        dmatz[0, 0] = dmatz[1, 1] = dmatz[2, 2] = cosa_[2]

        # mat += direction.outer_product(direction) * (1. - cosa)
        dmatx += -op_mat * cosa_[0] + (direction_[:, 0].outer_product(direction) + direction.outer_product(direction_[:, 0])) * (1. - cosa)
        dmaty += -op_mat * cosa_[1] + (direction_[:, 1].outer_product(direction) + direction.outer_product(direction_[:, 1])) * (1. - cosa)
        dmatz += -op_mat * cosa_[2] + (direction_[:, 2].outer_product(direction) + direction.outer_product(direction_[:, 2])) * (1. - cosa)

        '''d = direction * sina
        mat += ti.Matrix([
            [0, -d[2], +d[1]],
            [+d[2], 0, -d[0]],
            [-d[1], +d[0], 0],
        ], float)'''
        d_[:, :] = direction_ * sina + direction.outer_product(sina_)

    dx_ = d_[:, 0]
    dmatx += ti.Matrix([
        [0, -dx_[2], +dx_[1]],
        [+dx_[2], 0, -dx_[0]],
        [-dx_[1], +dx_[0], 0],
    ], float)
    dy_ = d_[:, 1]
    dmaty += ti.Matrix([
        [0, -dy_[2], +dy_[1]],
        [+dy_[2], 0, -dy_[0]],
        [-dy_[1], +dy_[0], 0],
    ], float)
    dz_ = d_[:, 2]
    dmatz += ti.Matrix([
        [0, -dz_[2], +dz_[1]],
        [+dz_[2], 0, -dz_[0]],
        [-dz_[1], +dz_[0], 0],
    ], float)

    return dmatx, dmaty, dmatz


@ti.kernel
def axis_angle_to_dmat_kernel(a: ti.types.ndarray(dtype=ti.math.vec3), 
                              dmatx: ti.types.ndarray(dtype=ti.math.mat3),
                              dmaty: ti.types.ndarray(dtype=ti.math.mat3),
                              dmatz: ti.types.ndarray(dtype=ti.math.mat3)):
    for b in range(a.shape[0]):
        x, y, z = axis_angle_to_dmat_func(a[b])
        dmatx[b][:, :] = x
        dmaty[b][:, :] = y
        dmatz[b][:, :] = z


@ti.func
def inverse_transform_func(mat: ti.math.mat4) -> ti.math.mat4:
    inv = mat[:, :]
    inv[:3, :3] = mat[:3, :3].transpose()
    inv[:3, 3] = -inv[:3, :3] @ mat[:3, 3]
    return inv


@ti.kernel
def inverse_transform_kernel(fwd: ti.types.ndarray(dtype=ti.math.mat4), inv: ti.types.ndarray(ti.math.mat4)):
    for b in range(fwd.shape[0]):
        inv[b][:, :] = inverse_transform_func(fwd[b])


def inverse_transform(matrix: torch.Tensor) -> torch.Tensor:
    """
    matrix: [B, 4, 4],
    """
    ret = torch.zeros_like(matrix)
    inverse_transform_kernel(matrix, ret)
    return ret


@ti.func
def quat_to_axis_angle_func(quat: ti.math.vec4):
    """quat [w, x, y, z]"""
    ret = ti.Vector.zero(float, 3)
    quat_len = quat.norm()
    if quat_len > 0.:
        quat /= quat_len
        axis_len = quat[1:].norm()
        if axis_len > 0.:
            angle = 2. * ti.atan2(axis_len, quat[0])
            ret[:] = angle * (quat[1:] / axis_len)
    return ret


@ti.kernel
def quat_to_axis_angle_kernel(quat: ti.types.ndarray(dtype=ti.math.vec4), aa: ti.types.ndarray(dtype=ti.math.vec3)):
    """quat [w, x, y, z]"""
    for b in range(quat.shape[0]):
        aa[b] = quat_to_axis_angle_func(quat[b])


def quat_to_axis_angle(quat: torch.Tensor):
    B, D = quat.shape
    assert D == 4
    dtype, device = quat.dtype, quat.device
    aa = torch.zeros((B, 3), dtype=dtype, device=device)
    quat_to_axis_angle_kernel(quat.contiguous(), aa)
    return aa


@ti.func
def fwd_vel_func(vel: sim_utils.vec6, origin_xyz: ti.math.vec3) -> sim_utils.vec6:
    ret = ti.Vector.zero(float, 6)
    ret[3:] = vel[3:]
    ret[:3] = vel[:3] + vel[3:].cross(origin_xyz)
    return ret


@ti.func
def update_mat_func(mat: ti.math.mat4, vel: sim_utils.vec6, dt: float) -> ti.math.mat4:
    ret = mat[:, :]
    ret[:3, 3] += vel[:3] * dt
    ret[:3, :3] = axis_angle_to_matrix_func(vel[3:] * dt) @ mat[:3, :3]
    return ret


@ti.func
def skew_func(xyz: ti.math.vec3):
    x, y, z = xyz
    return ti.Matrix([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ])


@ti.kernel
def skew_kernel(xyz: ti.types.ndarray(dtype=ti.math.vec3), ans: ti.types.ndarray(dtype=ti.math.mat3)):
    for b in range(xyz.shape[0]):
        ans[b] = skew_func(xyz[b])


def skew(xyz: torch.Tensor) -> torch.Tensor:
    B, D = xyz.shape
    assert D == 3
    dtype, device = xyz.dtype, xyz.device
    ret = torch.zeros((B, 3, 3), dtype=dtype, device=device)
    skew_kernel(xyz.contiguous(), ret)
    return ret


@ti.func
def angular_velocity_to_drot_dt_func(rot: ti.math.mat3, angular_velocity: ti.math.vec3):
    """given current rotation matrix `rot` and angular velocity `angular_velocity`, 
    return the time derivative of each matrix element. (mat3x3)"""
    x, y, z = angular_velocity
    drot = ti.Matrix.zero(float, 3, 3)
    drot[:, :] = skew_func(angular_velocity) @ rot
    return drot


@ti.kernel
def angular_velocity_to_drot_dt_kernel(drot: ti.types.ndarray(dtype=ti.math.mat3), 
                                       rot: ti.types.ndarray(dtype=ti.math.mat3), 
                                       angular_velocity: ti.types.ndarray(dtype=ti.math.vec3)):
    for b in range(drot.shape[0]):
        drot[b][:, :] = angular_velocity_to_drot_dt_func(rot[b], angular_velocity[b])


@ti.func
def drot_dt_to_angular_velocity_func(rot: ti.math.mat3, drot: ti.math.mat3):
    """given current rotation matrix `rot` and derivative `drot`, 
    return the tangular velocity. (vec3)"""
    vel = ti.Vector.zero(float, 3)
    mat = drot @ rot.transpose()
    vel[0] = (mat[2, 1] - mat[1, 2]) / 2.
    vel[1] = (mat[0, 2] - mat[2, 0]) / 2.
    vel[2] = (mat[1, 0] - mat[0, 1]) / 2.
    return vel


@ti.kernel
def drot_dt_to_angular_velocity_kernel(angular_velocity: ti.types.ndarray(dtype=ti.math.vec3),
                                       drot: ti.types.ndarray(dtype=ti.math.mat3), 
                                       rot: ti.types.ndarray(dtype=ti.math.mat3), 
                                       ):
    for b in range(drot.shape[0]):
        angular_velocity[b][:] = drot_dt_to_angular_velocity_func(rot[b], drot[b])


@ti.func
def project_floating_pos_vel_func(
    par_mat: ti.math.mat4, 
    chd_mat: ti.math.mat4,
    par_vel: sim_utils.vec6,
    chd_vel: sim_utils.vec6,
    origin: ti.math.mat4,
    ):
    """
    Args:
        - par_mat, chd_mat: transformation matrix in world frame
        - par_vel, chd_vel: link velocity in world frame
    
    Return
        - mat4x4: in dof frame, CHD = PAR @ ORI @ MAT
        - vec6: in dof frame,
    """
    mat = ti.Matrix.zero(float, 4, 4)
    vel = ti.Vector.zero(float, 6)
    
    mat[:, :] = inverse_transform_func(par_mat @ origin) @ chd_mat # chd = par @ origin @ mat
    dmatp = angular_velocity_to_drot_dt_func(par_mat[:3, :3], par_vel[3:])
    dmatc = angular_velocity_to_drot_dt_func(chd_mat[:3, :3], chd_vel[3:])

    vel[:3] = origin[:3, :3].transpose() @ (
        dmatp.transpose() @ (chd_mat[:3, 3] - par_mat[:3, 3]) +
        par_mat[:3, :3].transpose() @ (chd_vel[:3] - par_vel[:3])
    )
    vel[3:] = drot_dt_to_angular_velocity_func(
        mat[:3, :3], origin[:3, :3].transpose() @ (
        dmatp.transpose() @ chd_mat[:3, :3] + par_mat[:3, :3].transpose() @ dmatc
    ))
    return mat, vel


@ti.func
def project_revolute_pos_vel_func(
    par_mat: ti.math.mat4, 
    chd_mat: ti.math.mat4,
    par_vel: sim_utils.vec6,
    chd_vel: sim_utils.vec6,
    origin: ti.math.mat4,
    axis: ti.math.vec3,
    ):
    """
    Args:
        - par_mat, chd_mat: transformation matrix in world frame
        - par_vel, chd_vel: link velocity in world frame
        - axis: axis in local frame

    Return: 
        - mat4x4: in dof frame, CHD = PAR @ ORI @ MAT
        - vec6: in dof frame,
    """
    mat, vel = project_floating_pos_vel_func(par_mat, chd_mat, par_vel, chd_vel, origin)
    aa = quat_to_axis_angle_func(mat3_to_quat_func(mat[:3, :3]))
    axis_direction = ti.Vector([0., 0., 0.])
    if axis.norm() > 0:
        axis_direction[:] = axis / axis.norm()
    aa_proj = aa.dot(axis_direction) * axis_direction

    vel_proj = ti.Vector.zero(float, 6)
    vel_proj[3:] = vel[3:].dot(axis_direction) * axis_direction

    mat_proj = ti.Matrix.identity(float, 4)
    mat_proj[:3, :3] = axis_angle_to_matrix_func(aa_proj)

    return mat_proj, vel_proj


def rotation_matrix(angle: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Return matrix to rotate about axis defined by point and direction.

    direction may not be normalized.
    """
    B, = angle.shape
    B_, D = direction.shape
    assert B == B_
    assert D == 3
    dtype, device = angle.dtype, angle.device
    EPS = sim_utils.get_eps(dtype)

    sina = torch.sin(angle)
    cosa = torch.cos(angle)

    direction = direction / torch.clamp_min(direction.norm(dim=-1, keepdim=True), EPS)
    # rotation matrix around unit vector

    M = torch.zeros((B, 4, 4), dtype=dtype, device=device)
    M[:, 3, 3] = 1.
    M[:, 0, 0] = M[:, 1, 1] = M[:, 2, 2] = cosa
    M[:, :3, :3] += direction.view(-1, 3, 1) @ direction.view(-1, 1, 3) * (1.0 - cosa.view(-1, 1, 1))

    direction = direction * sina.view(-1, 1)

    R = torch.stack([-direction[:, 2], +direction[:, 2], +direction[:, 1], -direction[:, 1], -direction[:, 0], +direction[:, 0]], dim=-1)
    M[:, [0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]] += R

    return M


def quaternion_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    B, D = quaternions.shape
    assert D == 4
    dtype, device = quaternions.dtype, quaternions.device
    EPS = sim_utils.get_eps(dtype)

    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / torch.clamp_min((quaternions * quaternions).sum(-1), EPS)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def relative_pos7d_world_frame(pos1: torch.Tensor, pos2: torch.Tensor):
    """
    `ret[:3] = (pos1 - pos2)[:3]`

    `ret[3:] = rel_angle(pos1[3:], pos2[3:])`

    pos1.shape = pos2.shape = (B, 7)
    ret.shape = (B, 6)
    """
    B, D = pos1.shape
    B_, D_ = pos2.shape
    assert B == B_, f"{pos1.shape} {pos2.shape}"
    assert D == D_ and D == 7, f"{pos1.shape} {pos2.shape}"
    dtype, device = pos1.dtype, pos1.device
    M = sim_utils.create_zero_4x4_eye(B, dtype, device)
    M[:, :3, :3] = quaternion_matrix(pos1[:, 3:]) @ quaternion_matrix(pos2[:, 3:]).transpose(1, 2)
    M[:, :3, 3] = pos1[:, :3] - pos2[:, :3]
    ret7d = matrix_to_pos7d(M)
    ret6d = sim_utils.create_zero_6d_vel(B, dtype, device)
    ret6d[:, :3] = ret7d[:, :3]
    ret6d[:, 3:] = quat_to_axis_angle(ret7d[:, 3:])
    return ret6d
