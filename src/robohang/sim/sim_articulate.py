import taichi as ti

from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field

import torch
import numpy as np

import omegaconf
import batch_urdf

from .sim_utils import BaseClass
from .articulate import Articulate
from .sim_rigid import rigid_step_vel_func
from . import sim_utils
from . import so3


@ti.dataclass
class ArticulateSimJoint:
    par_link_idx: int # const
    chd_link_idx: int # const
    joint_type: int # const
    origin: ti.math.mat4 # const, in joint frame
    axis: ti.math.vec3 # const, in dof frame
    dof_mat: ti.math.mat4 # in dof frame
    dof_vel: sim_utils.vec6 # in dof frame
    local_wrench: sim_utils.vec6 # in local frame

    def get_shape(self):
        return (3 + 16 + 3 + 16 + 6 + 6, )


@ti.dataclass
class ArticulateSimLink:
    mass: float # const
    is_static: bool # const
    inertia: ti.math.mat3 # const
    cm_origin: ti.math.mat4 # const, in link frame
    pos_mat: ti.math.mat4 # in world frame
    vel_vec: sim_utils.vec6 # in world frame
    wrench: sim_utils.vec6 # in world frame

    def get_shape(self):
        return (1 + 9 + 16 + 16 + 6 + 6, )
    
    @ti.func
    def inverse_mass_func(self):
        ret = 0.
        if not self.is_static:
            ret = 1. / self.mass
        return ret
        
    @ti.func
    def inverse_inertia_local_func(self):
        ret = ti.Matrix.zero(float, 3, 3)
        if not self.is_static:
            ret[:, :] = self.inertia.inverse()
        return ret
    
    @ti.func
    def inverse_inertia_world_func(self):
        ret = ti.Matrix.zero(float, 3, 3)
        if not self.is_static:
            cm_rot = self.get_center_of_mass_pos_func()[:3, :3]
            ret[:, :] = cm_rot @ self.inertia.inverse() @ cm_rot.transpose()
        return ret
        
    @ti.func
    def get_center_of_mass_pos_func(self):
        return self.pos_mat @ self.cm_origin
    
    @ti.func
    def get_center_of_mass_linear_velocity_func(self):
        return self.vel_vec[:3] + self.vel_vec[3:].cross(
            (self.get_center_of_mass_pos_func() - self.pos_mat)[:3, 3])

    @ti.func
    def get_center_of_mass_angular_velocity_func(self):
        return self.vel_vec[3:]
    
    @ti.func
    def set_linear_velocity_func(self, v):
        self.vel_vec[:3] = v

    @ti.func
    def set_angular_velocity_func(self, v):
        self.vel_vec[3:] = v
    

@dataclass
class SingleLink:
    name: str
    mass: float
    is_static: bool
    inertia: np.ndarray
    """[3, 3]"""
    cm_origin: np.ndarray
    """[4, 4]"""

    parent_joint: Optional[str] = None
    child_joint: List[str] = field(default_factory=list)

    @property
    def cm_xyz(self) -> np.ndarray:
        return self.cm_origin[:3, 3]
    
    @property
    def cm_rot(self) -> np.ndarray:
        return self.cm_origin[:3, :3]


@dataclass
class SingleJoint:
    name: str
    parent: str
    child: str
    type: str
    origin: np.ndarray
    """[4, 4]"""
    axis: np.ndarray
    """[3, ]"""


class MergeLink:
    def __init__(self) -> None:
        self.links: List[SingleLink] = []
        self.names: Tuple[str] = ()

        self.mass = 0.
        self.is_static = False
        self.inertia = np.zeros((3, 3))
        """[3, 3]"""
        self.cm_origin = np.eye(4, dtype=float)
        """[4, 4]"""

    @staticmethod
    def _mass_point_inertia(mass: float, xyz: np.ndarray):
        t_sqr = np.sum(xyz * xyz)
        return mass * (np.eye(3, dtype=float) * t_sqr - np.outer(xyz, xyz))
    
    @property
    def cm_xyz(self) -> np.ndarray:
        return self.cm_origin[:3, 3]
    
    @property
    def cm_rot(self) -> np.ndarray:
        return self.cm_origin[:3, :3]
    
    @property
    def master_link(self) -> str:
        """pos of this dummy link = pos of master_link in articulate"""
        return self.names[0]

    def add_link(self, other: SingleLink):
        self.links.append(other)
        self.names += (other.name, )
        self.is_static = self.is_static or other.is_static

        new_mass = self.mass + other.mass
        if new_mass == 0:
            return
        new_cm_xyz = (self.cm_xyz * self.mass + 
                      other.cm_xyz * other.mass) / new_mass
        other_cm_rot = other.cm_rot
        new_inertia = (
            self.inertia + other_cm_rot @ other.inertia @ other_cm_rot.T+
            self._mass_point_inertia(self.mass, self.cm_xyz - new_cm_xyz) +
            self._mass_point_inertia(other.mass, other.cm_xyz - new_cm_xyz)
        )

        self.mass = new_mass
        self.cm_origin[:3, 3] = new_cm_xyz
        self.inertia[...] = new_inertia


@ti.func
def inv6d_func(a: ti.math.mat3, b: ti.math.mat3, c: ti.math.mat3, d: ti.math.mat3):
    """
    A direct inverse method. Assume `a` and `d` are invertible.
    - `|a b| |a_ b_| = |I 0|`
    - `|c d| |c_ d_| = |0 I|`
    """
    a_ = ti.Matrix.zero(float, 3, 3)
    b_ = ti.Matrix.zero(float, 3, 3)
    c_ = ti.Matrix.zero(float, 3, 3)
    d_ = ti.Matrix.zero(float, 3, 3)
    
    a_inv = a.inverse()
    d_inv = d.inverse()
    cab_d_inv = (c @ a_inv @ b - d).inverse()
    bdc_a_inv = (b @ d_inv @ c - a).inverse()
    
    c_[:, :] = cab_d_inv @ c @ a_inv
    a_[:, :] = a_inv @ (ti.Matrix.identity(float, 3) - b @ c_)
    b_[:, :] = bdc_a_inv @ b @ d_inv
    d_[:, :] = d_inv @ (ti.Matrix.identity(float, 3) - c @ b_)

    return a_, b_, c_, d_


@ti.func
def make_local_frame_func(axis: ti.math.vec3):
    """return mat, mat[0, :] || axis"""
    x = ti.Vector([1., 0., 0.], dt=float)
    y = ti.Vector([0., 1., 0.], dt=float)
    z = ti.Vector([0., 0., 1.], dt=float)
    axis_len = axis.norm()
    if (axis != 0).any():
        x[:] = axis / axis_len
        min_abs = ti.min(*ti.abs(x))
        for i in ti.static(range(3)):
            if ti.abs(x[i]) == min_abs:
                y[:] = 0.
                y[i] = 1.
        y[:] -= y.dot(x) * x
        y /= y.norm()
        z[:] = x.cross(y)
        z /= z.norm()
    return ti.Matrix.cols([x, y, z])


@ti.data_oriented
class ArticulateSim(BaseClass):
    SupportJointType: Dict[str, int] = {
        "floating": 0,
        "revolute": 1,
    }
    _FloatingIdx = SupportJointType["floating"]
    _RevoluteIdx = SupportJointType["revolute"]

    def __init__(self,
                 articulate: Articulate,
                 articulate_sim_cfg: omegaconf.DictConfig,
                 global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(global_cfg)

        self._articulate = articulate
        self._is_dynamic: bool = bool(articulate_sim_cfg.is_dynamic)

        self._default_inertia = float(global_cfg.default_inertia)
        self._default_mass = float(global_cfg.default_mass)

        if self._is_dynamic:
            self._static_links: Set[str] = set(list(articulate_sim_cfg.static_links))
            self._num_velocity_iterations: int = int(articulate_sim_cfg.num_velocity_iterations)
            self._velocity_damping_rate: float = float(articulate_sim_cfg.velocity_damping_rate)

            self._sim_link_list: List[ArticulateSimLink] = []
            self._sim_link_to_idx: Dict[Tuple[str], int] = {}
            self._idx_to_sim_link: Dict[int, Tuple[str]] = {}

            self._sim_joint_list: List[ArticulateSimJoint] = []
            self._sim_joint_to_idx: Dict[str, int] = {}
            self._idx_to_sim_joint: Dict[int, str] = {}

            self._merge_link_map: Dict[Tuple[str], MergeLink]
            self._single_link_to_merge_link_map: Dict[str, Tuple[str]]
            self._single_joint_map: Dict[str, SingleJoint]

            self._build_articulate_tree()

            self._field_link: ti.StructField
            """[B, L], L = number of merged links"""
            self._field_joint: ti.StructField
            """[B, J], J does not include fixed joints"""

    def _parse_sim_link(self, link: MergeLink) -> ArticulateSimLink:
        assert isinstance(link, MergeLink)
        return ArticulateSimLink(
            mass=link.mass,
            is_static=link.is_static,
            inertia=ti.Matrix(link.inertia),
            cm_origin=ti.Matrix(link.cm_origin),
            pos_mat=ti.Matrix(np.eye(4, dtype=float)),
            vel_vec=ti.Vector([0.] * 6),
            wrench=ti.Vector([0.] * 6),
        )

    def _parse_sim_joint(self, joint: SingleJoint) -> Union[ArticulateSimJoint, None]:
        assert isinstance(joint, SingleJoint)
        par_link_idx=self._sim_link_to_idx[self._single_link_to_merge_link_map[joint.parent]]
        chd_link_idx=self._sim_link_to_idx[self._single_link_to_merge_link_map[joint.child]]
        if par_link_idx == chd_link_idx:
            assert joint.type == "fixed"
            return
        else:
            assert joint.type != "fixed"
            return ArticulateSimJoint(
                par_link_idx=par_link_idx,
                chd_link_idx=chd_link_idx,
                joint_type=self.SupportJointType[joint.type],
                origin=ti.Matrix(joint.origin),
                axis=ti.Vector(joint.axis),
                dof_mat=ti.Matrix(np.eye(4, dtype=float)),
                dof_vel=ti.Vector([0.] * 6),
                local_wrench=ti.Vector([0.] * 6),
            )
    
    def _parse_single_link(self, link: batch_urdf.Link) -> SingleLink:
        return SingleLink(
            name=link.name,
            mass=link.inertial.mass if link.inertial.mass is not None else self._default_mass,
            is_static=link.name in self._static_links,
            inertia=sim_utils.torch_to_numpy(link.inertial.inertia)[0, ...] if link.inertial.inertia is not None else np.diag([self._default_inertia] * 3),
            cm_origin=sim_utils.torch_to_numpy(link.inertial.origin)[0, ...] if link.inertial.origin is not None else np.diag([1.] * 4),
        )
    
    def _parse_single_joint(self, joint: batch_urdf.Joint) -> SingleJoint:
        return SingleJoint(
            name=joint.name,
            parent=joint.parent,
            child=joint.child,
            type=joint.type,
            origin=sim_utils.torch_to_numpy(joint.origin)[0, ...],
            axis=sim_utils.torch_to_numpy(joint.axis)[0, ...],
        )
    
    def _merge_links(self, sj: SingleJoint,
                     sj_dict: Dict[str, SingleJoint],
                     ml_dict: Dict[Tuple[str], MergeLink],
                     l2ml: Dict[str, Tuple[str]]):
        assert sj.type == "fixed"
        par_ml = ml_dict.pop(l2ml[sj.parent])
        chd_ml = ml_dict.pop(l2ml[sj.child])
        
        # create MergeLink Object
        new_ml = MergeLink()
        assert len(par_ml.links) > 0
        for sl in par_ml.links:
            new_ml.add_link(SingleLink(
                sl.name,
                sl.mass,
                sl.is_static,
                sl.inertia,
                sl.cm_origin
            )) # in master_link frame
        for sl in chd_ml.links:
            new_ml.add_link(SingleLink(
                sl.name,
                sl.mass,
                sl.is_static,
                sl.inertia,
                sj.origin @ sl.cm_origin
            )) # in master_link frame

        # update ml_dict
        ml_dict[new_ml.names] = new_ml
        
        # update l2ml
        for sl in par_ml.links:
            l2ml[sl.name] = new_ml.names
        for sl in chd_ml.links:
            l2ml[sl.name] = new_ml.names

        # update joints of chd
        for sl in chd_ml.links:
            for cj_str in sl.child_joint:
                cj = sj_dict[cj_str]
                cj.origin[...] = sj.origin @ cj.origin
                # cj.axis[...] = sj.origin[:3, :3] @ cj.axis
                cj.axis[...] = cj.origin[:3, :3] @ cj.axis
                cj.parent = new_ml.master_link
    
    def _build_articulate_tree(self):
        ml_dict: Dict[Tuple[str], MergeLink] = {}
        l2ml: Dict[str, Tuple[str]] = {}
        for link_idx, (link_name, link) in enumerate(self._articulate.link_map.items()):
            ml = MergeLink()
            ml.add_link(self._parse_single_link(link))
            ml_dict[ml.names] = ml
            l2ml[link_name] = ml.names

        sj_dict: Dict[str, SingleJoint] = {}
        for joint_idx, (joint_name, joint) in enumerate(self._articulate.joint_map.items()):
            sj = self._parse_single_joint(joint)
            sj_dict[sj.name] = sj
            par_ml = ml_dict[l2ml[sj.parent]]
            chd_ml = ml_dict[l2ml[sj.child]]
            assert len(par_ml.links) == 1
            par_ml.links[0].child_joint.append(sj.name)
            assert chd_ml.links[0].parent_joint is None
            chd_ml.links[0].parent_joint = sj.name

        for sj in sj_dict.values():
            if sj.type == "fixed":
                self._merge_links(sj, sj_dict, ml_dict, l2ml)

        self._merge_link_map = ml_dict
        self._single_link_to_merge_link_map = l2ml
        self._single_joint_map = sj_dict

        for link_idx, ml in enumerate(self._merge_link_map.values()):
            self._sim_link_list.append(self._parse_sim_link(ml))
            self._sim_link_to_idx[ml.names] = link_idx
            self._idx_to_sim_link[link_idx] = ml.names
            assert ml.is_static or (ml.mass > 0. and (np.linalg.eig(ml.inertia)[0] > 0).all())

        for sj in self._single_joint_map.values():
            sim_joint = self._parse_sim_joint(sj)
            if sim_joint is not None:
                joint_idx = len(self._sim_joint_list)
                self._sim_joint_list.append(sim_joint)
                self._sim_joint_to_idx[sj.name] = joint_idx
                self._idx_to_sim_joint[joint_idx] = sj.name

        self._ln_sim = len(self._sim_link_list)
        self._field_link = sim_utils.GLOBAL_CREATER.StructField(ArticulateSimLink, shape=(self._batch_size, self._ln_sim))
        for sim_link_idx, sim_link in enumerate(self._sim_link_list):
            self._field_link[0, sim_link_idx] = sim_link

        self._jn_sim = len(self._sim_joint_list)
        self._field_joint = sim_utils.GLOBAL_CREATER.StructField(ArticulateSimJoint, shape=(self._batch_size, self._jn_sim))
        for sim_joint_idx, sim_joint in enumerate(self._sim_joint_list):
            self._field_joint[0, sim_joint_idx] = sim_joint

        self._linkcm_impulse_force = sim_utils.GLOBAL_CREATER.VectorField(3, float, shape=(self._batch_size, self._ln_sim))
        """force respect to center of mass"""
        self._linkcm_impulse_torque = sim_utils.GLOBAL_CREATER.VectorField(3, float, shape=(self._batch_size, self._ln_sim))
        """torque respect to center of mass"""

        self._init_sim_kernel()

    @ti.kernel
    def _init_sim_kernel(self):
        for batch_idx, link_idx in ti.ndrange(self._batch_size, self._ln_sim):
            if batch_idx > 0:
                self._field_link[batch_idx, link_idx] = self._field_link[0, link_idx]

        for batch_idx, joint_idx in ti.ndrange(self._batch_size, self._jn_sim):
            if batch_idx > 0:
                self._field_joint[batch_idx, joint_idx] = self._field_joint[0, joint_idx]

    @ti.func
    def _save_link_mat_vel_func(self, batch_idx: int, link_idx: int, pos_mat: ti.math.mat4, vel_vec: sim_utils.vec6):
        self._field_link[batch_idx, link_idx].pos_mat[:, :] = pos_mat
        self._field_link[batch_idx, link_idx].vel_vec[:] = vel_vec

    @ti.func
    def _save_joint_dof_func(self, batch_idx: int, joint_idx: int, dof_mat: ti.math.mat4, dof_vel: sim_utils.vec6):
        "update field_joint.dof_mat and field_joint.dof_vel"
        self._field_joint[batch_idx, joint_idx].dof_mat[:, :] = dof_mat
        self._field_joint[batch_idx, joint_idx].dof_vel[:] = dof_vel

    @ti.kernel
    def _read_link_kernel(self, 
                          link_idx: int, 
                          link_pos: ti.types.ndarray(dtype=sim_utils.vec7),
                          link_vel: ti.types.ndarray(dtype=sim_utils.vec6),):
        for batch_idx in range(self._batch_size):
            self._save_link_mat_vel_func(batch_idx, link_idx, so3.pos7d_to_matrix_func(link_pos[batch_idx]), link_vel[batch_idx])

    @ti.kernel
    def _read_joint_kernel(self,
                           joint_idx: int,
                           dof_mat: ti.types.ndarray(dtype=ti.math.mat4),
                           dof_vel: ti.types.ndarray(dtype=sim_utils.vec6),
                           local_wrench: ti.types.ndarray(dtype=sim_utils.vec6),):
        for batch_idx in range(self._batch_size):
            self._save_joint_dof_func(batch_idx, joint_idx, dof_mat[batch_idx], dof_vel[batch_idx])
            self._field_joint[batch_idx, joint_idx].local_wrench[:] = local_wrench[batch_idx]
    
    @sim_utils.GLOBAL_TIMER.timer
    def _read_link_joint_states(self):
        for link_idx in range(self._ln_sim):
            master_link = self._merge_link_map[self._idx_to_sim_link[link_idx]].master_link
            link_pos = self._articulate._link_pos[master_link]
            link_vel = self._articulate._link_vel[master_link]
            self._read_link_kernel(link_idx, link_pos, link_vel)

        for joint_idx in range(self._jn_sim):
            joint_name = self._idx_to_sim_joint[joint_idx]
            joint = self._articulate._urdf.joint_map[joint_name]
            pos, vel = None, None
            if joint.name in self._articulate._urdf.actuated_joints_map.keys():
                pos = self._articulate._cfg_pos[joint.name]
                vel = self._articulate._cfg_vel[joint.name]
            dof_mat, dof_vel = self._articulate._forward_kinematics_joint(joint, pos, vel)
            local_wrench = self._articulate._wrench_map[joint_name]
            self._read_joint_kernel(joint_idx, dof_mat, dof_vel, local_wrench)

    @ti.kernel
    def _write_floating_joint_kernel(self, joint_idx: int, joint_vel: ti.types.ndarray(dtype=sim_utils.vec6)):
        for batch_idx in range(self._batch_size):
            joint = self._field_joint[batch_idx, joint_idx]
            joint_vel[batch_idx][:] = joint.dof_vel

    @ti.kernel
    def _write_revolute_joint_kernel(self, joint_idx: int, joint_vel: ti.types.ndarray(dtype=float)):
        for batch_idx in range(self._batch_size):
            joint = self._field_joint[batch_idx, joint_idx]
            joint_vel[batch_idx] = joint.dof_vel[3:].dot(joint.axis)

    @sim_utils.GLOBAL_TIMER.timer
    def _write_link_joint_states(self):
        cfg_pos = self._articulate.get_cfg_pos()
        cfg_vel = self._articulate.get_cfg_vel()
        for joint_idx in range(self._jn_sim):
            joint_name = self._idx_to_sim_joint[joint_idx]
            if joint_name in cfg_vel.keys():
                joint_type = self._single_joint_map[joint_name].type
                if joint_type == "floating":
                    self._write_floating_joint_kernel(joint_idx, cfg_vel[joint_name])
                elif joint_type == "revolute":
                    self._write_revolute_joint_kernel(joint_idx, cfg_vel[joint_name])
                else:
                    raise NotImplementedError(joint_type)
        self._articulate.set_cfg_pos_vel(cfg_pos, cfg_vel)

    @ti.func
    def _resolve_revolute_velocity_func(self,
                                        joint: ArticulateSimJoint,
                                        batch_idx: int,
                                        par_link_idx: int,
                                        chd_link_idx: int,
                                        ):
        par_link = self._field_link[batch_idx, par_link_idx]
        chd_link = self._field_link[batch_idx, chd_link_idx]

        joint_pos = par_link.pos_mat @ joint.origin
        axis_world = joint_pos[:3, :3] @ joint.axis
        local_frame = make_local_frame_func(axis_world)
        
        vp = par_link.get_center_of_mass_linear_velocity_func()
        vc = chd_link.get_center_of_mass_linear_velocity_func()
        wp = par_link.get_center_of_mass_angular_velocity_func()
        wc = chd_link.get_center_of_mass_angular_velocity_func()
        xp = par_link.get_center_of_mass_pos_func()[:3, 3]
        xc = chd_link.get_center_of_mass_pos_func()[:3, 3]
        xj = (joint_pos)[:3, 3]

        mp_inv = par_link.inverse_mass_func()
        mc_inv = chd_link.inverse_mass_func()
        ip_inv = par_link.inverse_inertia_world_func()
        ic_inv = chd_link.inverse_inertia_world_func()
        
        xp_xj_skew = so3.skew_func(xp - xj)
        xc_xj_skew = so3.skew_func(xc - xj)

        y = ti.Vector.zero(float, 6)
        a = ti.Matrix.zero(float, 3, 3)
        b = ti.Matrix.zero(float, 3, 3)
        c = ti.Matrix.zero(float, 3, 3)
        d = ti.Matrix.zero(float, 3, 3)

        # a, b, y[0:3] corresponding to linear velocity constraints
        a[:, :] = (
            ti.Matrix.identity(float, 3) * (mp_inv + mc_inv) +
            xc_xj_skew @ ic_inv @ xc_xj_skew.transpose() +
            xp_xj_skew @ ip_inv @ xp_xj_skew.transpose()
        )
        b[:, :] = (
            xc_xj_skew @ ic_inv +
            xp_xj_skew @ ip_inv
        ) @ local_frame
        y[:3] = vp - vc + xp_xj_skew @ wp - xc_xj_skew @ wc

        # c, d, y[3:6] corresponding to angular velocity constraints
        c[:, :] = local_frame.transpose() @ (
            ic_inv @ xc_xj_skew.transpose() +
            ip_inv @ xp_xj_skew.transpose()
        )
        d[:, :] = local_frame.transpose() @ (ip_inv + ic_inv) @ local_frame
        y[3:] = local_frame.transpose() @ (wp - wc)

        force = ti.Vector.zero(float, 3)
        torque = ti.Vector.zero(float, 3)
        if joint.joint_type == self._FloatingIdx:
            pass
        elif joint.joint_type == self._RevoluteIdx:
            # revolute constrain
            c[0, :] = 0.
            d[0, :] = 0.
            d[0, 0] = 1.
            y[3] = 0. # torque along axis = 0

            a_, b_, c_, d_ = inv6d_func(a, b, c, d)
            force[:] = (a_ @ y[:3] + b_ @ y[3:])
            torque[:] = local_frame @ (c_ @ y[:3] + d_ @ y[3:])
        
        parent_force = -force
        parent_torque = -torque + xp_xj_skew.transpose() @ (-force)
        self._linkcm_impulse_force[batch_idx, par_link_idx] += parent_force
        self._linkcm_impulse_torque[batch_idx, par_link_idx] += parent_torque

        child_force = +force
        child_torque = +torque + xc_xj_skew.transpose() @ (+force)
        self._linkcm_impulse_force[batch_idx, chd_link_idx] += child_force
        self._linkcm_impulse_torque[batch_idx, chd_link_idx] += child_torque

    @ti.func
    def _update_link_velocity_use_impulse_func(self, batch_idx: int, link_idx: int):
        link = self._field_link[batch_idx, link_idx]

        link_mat = link.pos_mat
        link_xyz = link_mat[:3, 3]
        link_vel = link.vel_vec

        cm_mat = link.get_center_of_mass_pos_func()
        cm_xyz = cm_mat[:3, 3]
        cm_link_xyz = cm_xyz - link_xyz

        cm_vel = so3.fwd_vel_func(link_vel, cm_link_xyz)
        cm_vel[:3] += link.inverse_mass_func() * self._linkcm_impulse_force[batch_idx, link_idx]
        cm_vel[3:] += link.inverse_inertia_world_func() @ \
            self._linkcm_impulse_torque[batch_idx, link_idx]

        self._field_link[batch_idx, link_idx].vel_vec[:] = so3.fwd_vel_func(cm_vel, -cm_link_xyz)

    @ti.func
    def debug_get_angular_momentum_func(self):
        link = self._field_link[0, 2]
        lin_vel = link.vel_vec[:3]
        ang_vel = link.vel_vec[3:]
        cm_rot = link.get_center_of_mass_pos_func()[:3, :3]
        cm_rel = link.get_center_of_mass_pos_func()[:3, 3] - link.pos_mat[:3, 3]
        ang_mom = (
            cm_rot @ link.inertia @ cm_rot.transpose() @ ang_vel +
            link.mass * cm_rel.cross(lin_vel + ang_vel.cross(cm_rel))
        )
        return ang_mom[1]
    
    @ti.kernel
    def debug_get_angular_momentum_kernel(self) -> float:
        return self.debug_get_angular_momentum_func()

    @ti.kernel
    def _step_vel_impl_kernel(self, dt: float, velocity_damping_rate: float, gravity: ti.types.ndarray(dtype=ti.math.vec3)):
        # assemble rigid wrench
        # gravity
        for batch_idx, link_idx in ti.ndrange(self._batch_size, self._ln_sim):
            link = self._field_link[batch_idx, link_idx]
            wrench = ti.Vector.zero(float, 6)
            wrench[:3] = link.mass * gravity[batch_idx]
            self._field_link[batch_idx, link_idx].wrench = wrench

        # wrench on joint
        for batch_idx, joint_idx in ti.ndrange(self._batch_size, self._jn_sim):
            joint = self._field_joint[batch_idx, joint_idx]
            par_link_idx = joint.par_link_idx
            chd_link_idx = joint.chd_link_idx
            par_link = self._field_link[batch_idx, par_link_idx]
            chd_link = self._field_link[batch_idx, chd_link_idx]
            joint_mat = par_link.pos_mat @ joint.origin
            parcm_mat = par_link.get_center_of_mass_pos_func()
            chdcm_mat = chd_link.get_center_of_mass_pos_func()

            joint_wrench = ti.Vector.zero(float, 6)
            joint_wrench[:3] = joint_mat[:3, :3] @ joint.local_wrench[:3]
            joint_wrench[3:] = joint_mat[:3, :3] @ joint.local_wrench[3:]
            
            par_wrench = ti.Vector.zero(float, 6)
            par_wrench[:3] = -joint_wrench[:3]
            par_wrench[3:] = -joint_wrench[3:] + (joint_mat[:3, 3] - parcm_mat[:3, 3]).cross(par_wrench[:3])
            chd_wrench = ti.Vector.zero(float, 6)
            chd_wrench[:3] = +joint_wrench[:3]
            chd_wrench[3:] = +joint_wrench[3:] + (joint_mat[:3, 3] - chdcm_mat[:3, 3]).cross(chd_wrench[:3])

            self._field_link[batch_idx, par_link_idx].wrench += par_wrench
            self._field_link[batch_idx, chd_link_idx].wrench += chd_wrench

        # update rigid vel using wrench
        for batch_idx, link_idx in ti.ndrange(self._batch_size, self._ln_sim):
            link = self._field_link[batch_idx, link_idx]
            new_vel = rigid_step_vel_func(dt, link.mass, link.inertia, link.cm_origin, 
                                          link.pos_mat, link.vel_vec, link.wrench)
            if not link.is_static:
                self._field_link[batch_idx, link_idx].vel_vec = new_vel

        # resolve constraint
        for vel_iter in ti.static(range(self._num_velocity_iterations)):
            self._linkcm_impulse_force.fill(0.)
            self._linkcm_impulse_torque.fill(0.)
            for batch_idx, joint_idx in ti.ndrange(self._batch_size, self._jn_sim):
                joint = self._field_joint[batch_idx, joint_idx]
                par_link_idx = joint.par_link_idx
                chd_link_idx = joint.chd_link_idx
                self._resolve_revolute_velocity_func(joint, batch_idx, par_link_idx, chd_link_idx)

            # update rigid vel using impulse
            for batch_idx, link_idx in ti.ndrange(self._batch_size, self._ln_sim):
                self._update_link_velocity_use_impulse_func(batch_idx, link_idx)

        # project back
        for batch_idx, joint_idx in ti.ndrange(self._batch_size, self._jn_sim):
            joint = self._field_joint[batch_idx, joint_idx]
            par_link = self._field_link[batch_idx, joint.par_link_idx]
            chd_link = self._field_link[batch_idx, joint.chd_link_idx]
            joint_origin = joint.origin
            axis = joint.axis

            par_mat = par_link.pos_mat
            chd_mat = chd_link.pos_mat
            par_vel = par_link.vel_vec
            chd_vel = chd_link.vel_vec

            if joint.joint_type == self._FloatingIdx:
                dof_mat, dof_vel = so3.project_floating_pos_vel_func(par_mat, chd_mat, par_vel, chd_vel, joint_origin)
                dof_vel *= ti.exp(-velocity_damping_rate * dt)
                self._save_joint_dof_func(batch_idx, joint_idx, dof_mat, dof_vel)
            elif joint.joint_type == self._RevoluteIdx:
                dof_mat, dof_vel = so3.project_revolute_pos_vel_func(par_mat, chd_mat, par_vel, chd_vel, joint_origin, axis)
                dof_vel *= ti.exp(-velocity_damping_rate * dt)
                self._save_joint_dof_func(batch_idx, joint_idx, dof_mat, dof_vel)

    @sim_utils.GLOBAL_TIMER.timer
    def _call_step_vel_impl(self, dt, gravity):
        self._step_vel_impl_kernel(dt, self._velocity_damping_rate, gravity)
        
    def _step_vel_impl(self, dt: float, gravity: torch.Tensor):
        self._read_link_joint_states()
        self._call_step_vel_impl(dt, gravity)
        self._write_link_joint_states()

    def _step_vel_articulate(self, dt: float, gravity: torch.Tensor):
        if self._is_dynamic:
            self._step_vel_impl(dt, gravity)

    @ti.kernel
    def _step_pos_floating_kernel(self,
                                  pos: ti.types.ndarray(dtype=sim_utils.vec7),
                                  vel: ti.types.ndarray(dtype=sim_utils.vec6),
                                  dt: float):
        for batch_idx in range(self._batch_size):
            pos[batch_idx][:3] += vel[batch_idx][:3] * dt
            pos[batch_idx][3:] = so3.mat3_to_quat_func(so3.axis_angle_to_matrix_func(dt * vel[batch_idx][3:]) @ so3.quat_to_mat3_func(pos[batch_idx][3:]))

    def _step_pos_impl(self, dt: float):
        for joint_name, joint in self._articulate.actuated_joints_map.items():
            if joint.type in ["revolute", "prismatic"]:
                self._articulate._cfg_pos[joint_name] += self._articulate._cfg_vel[joint_name] * dt
            elif joint.type == "floating":
                self._step_pos_floating_kernel(self._articulate._cfg_pos[joint_name], self._articulate._cfg_vel[joint_name], dt)
            else:
                raise NotImplementedError(joint.type)
        self._articulate._forward_kinematics(self._articulate._cfg_pos, self._articulate._cfg_vel)

    def _step_pos_articulate(self, dt: float):
        self._step_pos_impl(dt)