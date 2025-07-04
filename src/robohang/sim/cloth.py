import taichi as ti

from typing import Dict, Tuple
import copy

import numpy as np
import torch
import trimesh

import omegaconf
from .sim_utils import BaseClass
from . import maths
from . import sim_utils
from .sparse import SparseMatrix

@ti.data_oriented
class Cloth(BaseClass):
    def __init__(self, mesh: trimesh.Trimesh, cloth_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(global_cfg)
        self._name: str = cloth_cfg.name
        self._mesh = copy.deepcopy(mesh)
        
        # initialize elastic properties
        self._h = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""
        self._h.fill(cloth_cfg.h) # thickness

        self._rho = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""
        self._rho.fill(cloth_cfg.rho) # density

        self._E = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""
        self._E.fill(cloth_cfg.E)  # Young's modulus
        
        self._nu = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""
        self._nu.fill(cloth_cfg.nu) # Poisson's ratio

        self._alpha = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""
        self._alpha.fill(cloth_cfg.alpha) # bending coefficient

        # damping

        # stretch damping coefficient
        self._stretch_relax_t = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""
        self._stretch_relax_t.fill(cloth_cfg.stretch_relax_t)
        # bending damping coefficient
        self._bending_relax_t = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""
        self._bending_relax_t.fill(cloth_cfg.bending_relax_t)

        # Lame parameters
        self._mu = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""
        self._lda = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""
        # stretch coefficient
        self._ks = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""
        # bending coefficient
        self._kb = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B]"""

        self._update_cloth_physics_coefficient()

        # float eps
        self._rel_eps: float = float(sim_utils.get_eps(self._dtype))
        self._dx_eps: float = float(cloth_cfg.get("dx_eps", 1e-6))
        # hessian fix initial displacement
        self._ldlt_loop_init_alpha: float = float(cloth_cfg.E * self._rel_eps)
        self._deform_grad_dsvd_tol: float = float(cloth_cfg.get("deform_grad_dsvd_tol", 1e-5))
        self._ldlt_max_loop_cnt: int = int(cloth_cfg.get("ldlt_max_loop_cnt", 1024))
        self._ldlt_loop_multiplier: float = float(cloth_cfg.get("ldlt_loop_multiplier", 4.))

        self._f2v: ti.MatrixField
        """int, [F][3]"""
        self._f2e: ti.MatrixField
        """int, [F][3]"""
        self._e2v: ti.MatrixField
        """int, [E][2]"""
        self._e2f: ti.MatrixField
        """int, [E][2]"""
        self._v2f_cnt: ti.ScalarField
        """int, [V]"""
        self._v2f: ti.ScalarField
        """int, [V, NVF]"""
        self._v2e_cnt: ti.ScalarField
        """int, [V]"""
        self._v2e: ti.ScalarField
        """int, [V, NVE]"""
        self._pos_rest: ti.MatrixField
        """float, [V][3]"""
        self._pos: ti.MatrixField
        """float, [B, V][3]"""
        self._vel: ti.MatrixField
        """float, [B, V][3]"""
        self._mass: ti.ScalarField
        """float, [B, V]"""
        self._rest_angle: ti.ScalarField
        """float, [E]"""
        self._TTT43: ti.MatrixField
        """float, [F][4, 3]"""
        self._rest_area: ti.ScalarField
        """float, [F]"""

        self._build_topo(mesh)
        self._build_dynamics(mesh)
        self._reset()

        # elastic force
        self._deform_grad: ti.MatrixField = sim_utils.GLOBAL_CREATER.MatrixField(n=3, m=3, dtype=float, shape=(self._batch_size, self._nf))
        """[B, F][3, 3]"""
        self._ddpsi_dFdF: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, self._nf, 9, 9))
        """[B, F, 9, 9]"""
        self._elastic_force: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, self._nv))
        """[B, V][3]"""
        self._E_hessian_stretch: ti.MatrixField = sim_utils.GLOBAL_CREATER.MatrixField(n=3, m=3, dtype=float, shape=(self._batch_size, self._nf, 3, 3))
        """[B, F, 3, 3][3, 3]"""
        self._E_hessian_bending: ti.MatrixField = sim_utils.GLOBAL_CREATER.MatrixField(n=3, m=3, dtype=float, shape=(self._batch_size, self._ne, 4, 4))
        """[B, E, 4, 4][3, 3]"""

        # external force
        self._external_force: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, self._nv))
        """[B, V][3]"""
        self._external_hessian: ti.MatrixField = sim_utils.GLOBAL_CREATER.MatrixField(n=3, m=3, dtype=float, shape=(self._batch_size, self._nv))
        """[B, V][3, 3]"""

        # fix hessian
        self._is_fixed_flag = sim_utils.GLOBAL_CREATER.ScalarField(dtype=bool, shape=(self._batch_size, ))
        """[B, ]"""
        self._is_all_success = sim_utils.GLOBAL_CREATER.ScalarField(dtype=bool, shape=(self._batch_size, ))
        """[B, ]"""

        # derivative of deformation gradient wrt coordinates dF / dx
        self._dF_dx = sim_utils.GLOBAL_CREATER.StructField(maths.tiTensor3333, shape=(self._batch_size, self._nf))
        """[B, F][3, 3, 3, 3]"""
        self._U_f: ti.MatrixField = sim_utils.GLOBAL_CREATER.MatrixField(n=3, m=3, dtype=float, shape=(self._batch_size, self._nf))
        """[B, F][3, 3]"""
        self._S_f: ti.MatrixField = sim_utils.GLOBAL_CREATER.MatrixField(n=3, m=3, dtype=float, shape=(self._batch_size, self._nf))
        """[B, F][3, 3]"""
        self._V_f: ti.MatrixField = sim_utils.GLOBAL_CREATER.MatrixField(n=3, m=3, dtype=float, shape=(self._batch_size, self._nf))
        """[B, F][3, 3]"""
        self._dU_f = sim_utils.GLOBAL_CREATER.StructField(maths.tiTensor3333, shape=(self._batch_size, self._nf))
        """[B, F][3, 3, 3, 3]"""
        self._dS_f = sim_utils.GLOBAL_CREATER.StructField(maths.tiTensor333, shape=(self._batch_size, self._nf))
        """[B, F][3, 3, 3]"""
        self._dV_f = sim_utils.GLOBAL_CREATER.StructField(maths.tiTensor3333, shape=(self._batch_size, self._nf))
        """[B, F][3, 3, 3, 3]"""

        # set constraint
        self._constraint: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, self._nv))
        """[B, V][3]"""
        self._constraint.fill(1.0)

        # sparse matrix
        self._hessian_sparse = SparseMatrix(
            batch_size=self._batch_size,
            nmax_row=self._nv * 3,
            nmax_column=self._nv * 3,
            nmax_triplet=self._nf * (3 * 3) ** 2 + self._ne * (4 * 3) ** 2, 
            store_dense=False
        )

    #######################################################
    # Section: Topo
    #######################################################

    @ti.func
    def _find_opp_vert_on_face_func(self, fid: int, v1id: int, v2id: int) -> int:
        """
        Args:
            - fid, v1id, v2id: int

        Return:
            - v3id: int, where 3 vertices on fid is v1id, v2id, v3id
        """
        assert v1id != v2id, "[ERROR] in find_vertex_3_on_face fid:{} v1id:{} v2id:{} v1id=v2id.".format(
            fid, v1id, v2id)
        v3id = -1
        if fid != -1:
            for i in ti.static(range(3)):
                if v1id != self._f2v[fid][i] and v2id != self._f2v[fid][i]:
                    assert v3id == -1, "[ERROR] find multiple v3id on fid:{} [{},{},{}] with v1id:{} v2id:{}".format(
                        fid, self._f2v[fid][0], self._f2v[fid][1], self._f2v[fid][2], v1id, v2id)
                    v3id = self._f2v[fid][i]
        return v3id
        
    #######################################################
    # Section: Helper Functions
    #######################################################

    @ti.func
    def _get_vertex_volume_func(self, batch_idx, vid: int):
        volume = 0.0
        for i in range(self._v2f_cnt[vid]):
            fid = self._v2f[vid, i]
            cp = self._get_face_rest_cross_product_func(fid)
            volume += cp.norm() / 6 * self._h[batch_idx]
        return volume
    
    @ti.func
    def _get_edge_volume_func(self, batch_idx, eid: int):
        f1id, f2id = self._e2f[eid]
        volume = self._get_face_rest_cross_product_func(f1id).norm() / 6 * self._h[batch_idx]
        if f2id != -1:
            volume += self._get_face_rest_cross_product_func(f2id).norm() / 6 * self._h[batch_idx]
        return volume

    @ti.func
    def _get_face_rest_cross_product_func(self, fid: int) -> ti.Vector:
        ia, ib, ic = self._f2v[fid]
        a, b, c = self._pos_rest[ia], self._pos_rest[ib], self._pos_rest[ic]
        return (a - c).cross(b - c)
    
    @ti.func
    def _get_face_cross_product_func(self, batch_idx, fid: int) -> ti.Vector:
        ia, ib, ic = self._f2v[fid]
        a, b, c = self._pos[batch_idx, ia], self._pos[batch_idx, ib], self._pos[batch_idx, ic]
        return (a - c).cross(b - c)
    
    @ti.func
    def _get_face_rest_area_func(self, fid: int) -> float:
        return (self._get_face_rest_cross_product_func(fid)).norm() / 2.
    
    @ti.func
    def _get_face_rest_volume_func(self, batch_idx, fid: int) -> float:
        return self._get_face_rest_area_func(fid) * self._h[batch_idx]
    
    @ti.func
    def _get_face_normalized_func(self, batch_idx, fid: int) -> ti.Vector:
        cp = self._get_face_cross_product_func(batch_idx, fid)
        return maths.safe_normalized_func(cp, self._dx_eps ** 2)
    
    @ti.func
    def _get_face_max_length_func(self, batch_idx, fid: int) -> float:
        ia, ib, ic = self._f2v[fid]
        a, b, c = self._pos[batch_idx, ia], self._pos[batch_idx, ib], self._pos[batch_idx, ic]
        return ti.max(ti.math.length(a - b), ti.math.length(b - c), ti.math.length(c - a))

    
    @ti.func
    def _get_rest_dihedral_func(self, eid: int) -> float:
        """
        calculate theta = n1 x n2 along (x2 - x1)
        IMPORTANT: consider chirality.
        Example:
        self.edges_vid_fid[1] = v1id, v2id, f1id, f2id = 1, 2, 100, 101
        self._f2v[100] = 1, 2, 3 # chirality = +1
        self._f2v[100] = 2, 1, 3 # chirality = -1
        """
        f1id, f2id = self._e2f[eid]
        ret_val = 0.0
        chirality = 1.0
        if f2id != -1:
            v1id, v2id = self._e2v[eid]
            cp1 = self._get_face_rest_cross_product_func(f1id)
            cp2 = self._get_face_rest_cross_product_func(f2id)
            n1 = maths.safe_normalized_func(cp1, self._dx_eps ** 2)
            n2 = maths.safe_normalized_func(cp2, self._dx_eps ** 2)
            x12 = maths.safe_normalized_func(self._pos_rest[v2id] - self._pos_rest[v1id], self._dx_eps)
            sine = n2.cross(n1).dot(x12)
            cosine = n2.dot(n1)
            theta = ti.atan2(sine, cosine)
            f1_v1id, f1_v2id, f1_v3id = self._f2v[f1id]
            if (v2id == f1_v1id and v1id == f1_v2id) or \
               (v2id == f1_v2id and v1id == f1_v3id) or \
               (v2id == f1_v3id and v1id == f1_v1id):
                chirality = -1.0

            ret_val = theta * chirality
        return ret_val

    @ti.func
    def _get_dihedral_func(self, batch_idx, eid: int) -> float:
        """
        calculate theta = n1 x n2 along (x2 - x1)
        IMPORTANT: consider chirality.
        Example:
        self.edges_vid_fid[1] = v1id, v2id, f1id, f2id = 1, 2, 100, 101
        self._f2v[100] = 1, 2, 3 # chirality = +1
        self._f2v[100] = 2, 1, 3 # chirality = -1
        """
        f1id, f2id = self._e2f[eid]
        ret_val = 0.0
        chirality = 1.0
        if f2id != -1:
            v1id, v2id = self._e2v[eid]
            cp1 = self._get_face_cross_product_func(batch_idx, f1id)
            cp2 = self._get_face_cross_product_func(batch_idx, f2id)
            n1 = maths.safe_normalized_func(cp1, self._dx_eps ** 2)
            n2 = maths.safe_normalized_func(cp2, self._dx_eps ** 2)
            x12 = maths.safe_normalized_func(self._pos[batch_idx, v2id] - self._pos[batch_idx, v1id], self._dx_eps)
            sine = n2.cross(n1).dot(x12)
            cosine = n2.dot(n1)
            theta = ti.atan2(sine, cosine)
            f1_v1id, f1_v2id, f1_v3id = self._f2v[f1id]
            if (v2id == f1_v1id and v1id == f1_v2id) or \
               (v2id == f1_v2id and v1id == f1_v3id) or \
               (v2id == f1_v3id and v1id == f1_v1id):
                chirality = -1.0

            ret_val = theta * chirality
        return ret_val
    
    #######################################################
    # Section: Initialize and Update
    #######################################################

    @ti.kernel
    def _update_cloth_physics_coefficient_kernel(self):
        for batch_idx in range(self._batch_size):
            self._mu[batch_idx] = self._E[batch_idx] / 2. / (1. + self._nu[batch_idx])
            self._lda[batch_idx] = self._E[batch_idx] * self._nu[batch_idx] / \
                (1. + self._nu[batch_idx]) / (1. - 2. * self._nu[batch_idx])
            self._ks[batch_idx] = self._E[batch_idx] * self._h[batch_idx] / (1. - self._nu[batch_idx] ** 2)
            self._kb[batch_idx] = self._ks[batch_idx] * self._h[batch_idx] ** 2 * self._alpha[batch_idx] / 12.

    def _update_cloth_physics_coefficient(self):
        self._update_cloth_physics_coefficient_kernel()

    def _build_topo(self, mesh: trimesh.Trimesh):
        self._nv: int = int(mesh.vertices.shape[0])
        self._nf: int = int(mesh.faces.shape[0])

        self._f2v: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=int, shape=(self._nf, ))
        self._f2v.from_torch(torch.tensor(mesh.faces, dtype=self._dtype_int, device=self._device))

        edges_dict: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for fid in range(self._nf):
            for i, j in zip([0, 1, 2], [1, 2, 0]):
                vi = mesh.faces[fid, i]
                vj = mesh.faces[fid, j]
                if vi > vj:
                    vi, vj = vj, vi
                if (vi, vj) not in edges_dict.keys():
                    edges_dict[(vi, vj)] = [fid, -1]
                else:
                    edges_dict[(vi, vj)][1] = fid
        edges_list = [(k, v) for k, v in edges_dict.items()]

        self._ne: int = int(len(edges_list))
        self._e2v: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=2, dtype=int, shape=(self._ne, ))
        self._e2v.from_torch(torch.tensor([edges[0] for edges in edges_list], dtype=self._dtype_int, device=self._device))
        self._e2f: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=2, dtype=int, shape=(self._ne, ))
        self._e2f.from_torch(torch.tensor([edges[1] for edges in edges_list], dtype=self._dtype_int, device=self._device))

        v2f_list = [[] for _ in range(self._nv)]
        for fid in range(self._nf):
            for i in range(3):
                v2f_list[mesh.faces[fid, i]].append(fid)
        v2f_cnt = [len(x) for x in v2f_list]
        v2f_arr = -np.ones((self._nv, np.max(v2f_cnt)), dtype=int)
        for vid in range(self._nv):
            v2f_arr[vid, :v2f_cnt[vid]] = np.array(v2f_list[vid], dtype=int)
        self._v2f_cnt = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=(self._nv, ))
        self._v2f_cnt.from_torch(torch.tensor(v2f_cnt, dtype=self._dtype_int, device=self._device))
        self._v2f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=v2f_arr.shape)
        self._v2f.from_torch(torch.tensor(v2f_arr, dtype=self._dtype_int, device=self._device))

        v2e_list = [[] for _ in range(self._nv)]
        for eid in range(self._ne):
            for i in range(2):
                v2e_list[edges_list[eid][0][i]].append(eid)
        v2e_cnt = [len(x) for x in v2e_list]
        v2e_arr = -np.ones((self._nv, np.max(v2e_cnt)), dtype=int)
        for vid in range(self._nv):
            v2e_arr[vid, :v2e_cnt[vid]] = np.array(v2e_list[vid], dtype=int)
        self._v2e_cnt = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=(self._nv, ))
        self._v2e_cnt.from_torch(torch.tensor(v2e_cnt, dtype=self._dtype_int, device=self._device))
        self._v2e = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=v2e_arr.shape)
        self._v2e.from_torch(torch.tensor(v2e_arr, dtype=self._dtype_int, device=self._device))
        
        f2e = [[] for _ in range(self._nf)]
        for edge_idx, edge in enumerate(edges_list):
            f2e[edge[1][0]].append(edge_idx)
            if edge[1][1] != -1:
                f2e[edge[1][1]].append(edge_idx)
        self._f2e: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=int, shape=(self._nf, ))
        self._f2e.from_torch(torch.tensor(f2e, dtype=self._dtype_int, device=self._device))

    def _build_dynamics(self, mesh: trimesh.Trimesh):
        self._pos_rest:ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._nv, ))
        self._pos_rest.from_torch(torch.tensor(mesh.vertices, dtype=self._dtype, device=self._device))
        self._pos:ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, self._nv))
        self._vel:ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, self._nv))
        
        self._mass = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, self._nv))
        self._rest_angle = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._ne, ))
        self._TTT43 = sim_utils.GLOBAL_CREATER.MatrixField(n=4, m=3, dtype=float, shape=(self._nf, ))
        self._rest_area = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._nf, ))

        self._update_vfe_property_kernel()

    @ti.func
    def _update_face_property_func(self, fid: int):
        """update face's TTT43-matrix and rest area"""
        cp = self._get_face_rest_cross_product_func(fid)
        self._rest_area[fid] = cp.norm() / 2.

        ia, ib, ic = self._f2v[fid]
        a, b, c = self._pos_rest[ia], self._pos_rest[ib], self._pos_rest[ic]

        tmp_T = ti.Matrix.cols([b - a, c - a])
        tmp_TT = tmp_T.transpose()

        tmp_TTT = tmp_TT @ tmp_T # [2, 2]
        if ti.abs(tmp_TTT.determinant()) > tmp_TTT.norm_sqr() * self._rel_eps:
            TTT_inv = tmp_TTT.inverse()
            tmp_N = maths.safe_normalized_func(cp, self._dx_eps ** 2)

            tmp_TTT43_0 = -ti.Matrix([[1, 1]]) @ TTT_inv @ tmp_TT
            tmp_TTT43_12 = TTT_inv @ tmp_TT
            tmp_TTT43_3 = tmp_N

            self._TTT43[fid][0, :] = tmp_TTT43_0[0, :]
            self._TTT43[fid][1:3, :] = tmp_TTT43_12
            self._TTT43[fid][3, :] = tmp_TTT43_3
        else:
            self._TTT43[fid] = ti.Matrix.zero(float, 4, 3)

    @ti.func
    def _update_vertex_property_func(self, batch_idx: int, vid: int):
        """calculate vertex mass and update"""
        self._mass[batch_idx, vid] = self._rho[batch_idx] * \
            self._get_vertex_volume_func(batch_idx, vid)

    @ti.func
    def _update_edge_property_func(self, eid: int):
        """update rest angle"""
        self._rest_angle[eid] = self._get_rest_dihedral_func(eid)

    @ti.kernel
    def _update_vfe_property_kernel(self):
        for batch_idx, vid in ti.ndrange(self._batch_size, self._nv):
            self._update_vertex_property_func(batch_idx, vid)
        for fid in range(self._nf):
            self._update_face_property_func(fid)
        for eid in range(self._ne):
            self._update_edge_property_func(eid)

    #######################################################
    # Section: Physics
    #######################################################

    @ti.func
    def _get_dF_dx_func(self, batch_idx, fid, w, i, j, k) -> float:
        """Return dF_{fid}^{j, k} / dx_{vid}^{i} with vid=f2v[fid][w]"""
        # for implicit integral
        return self._dF_dx[batch_idx, fid].get(w, i, j, k)

    @ti.func
    def _get_deformation_gradient_func(self, batch_idx, fid: int) -> ti.Matrix:
        v0id, v1id, v2id = self._f2v[fid]
        x0 = self._pos[batch_idx, v0id]
        x1 = self._pos[batch_idx, v1id]
        x2 = self._pos[batch_idx, v2id]
        normalized = self._get_face_normalized_func(batch_idx, fid)
        tmp_X = ti.Matrix.cols([x0, x1, x2, normalized])
        ret_val = ti.Matrix.identity(float, 3)
        if self._TTT43[fid].any():
            ret_val = tmp_X @ self._TTT43[fid]
        return ret_val

    @ti.func
    def _get_dtheta_dX_func(self, batch_idx, eid: int) -> ti.Matrix:
        """return a 4x3 matrix"""
        ret_val = ti.Matrix.zero(dt=float, n=4, m=3)
        f1id, f2id = self._e2f[eid]
        v1id, v2id = self._e2v[eid]
        if f2id != -1:
            v3id = self._find_opp_vert_on_face_func(f1id, v1id, v2id)
            v4id = self._find_opp_vert_on_face_func(f2id, v1id, v2id)

            x1 = self._pos[batch_idx, v1id]
            x2 = self._pos[batch_idx, v2id]
            x3 = self._pos[batch_idx, v3id]
            x4 = self._pos[batch_idx, v4id]

            h1 = maths.get_distance_func(x3, x1, x2, False, self._dx_eps)
            h2 = maths.get_distance_func(x4, x1, x2, False, self._dx_eps)

            n1 = self._get_face_normalized_func(batch_idx, f1id)
            n2 = self._get_face_normalized_func(batch_idx, f2id)

            w_f1 = maths.get_2D_barycentric_weights_func(x3, x1, x2, self._dx_eps)
            w_f2 = maths.get_2D_barycentric_weights_func(x4, x1, x2, self._dx_eps)

            dtheta_dX1 = - w_f1[0] * n1 / ti.max(h1, self._dx_eps) - \
                w_f2[0] * n2 / ti.max(h2, self._dx_eps)
            dtheta_dX2 = - w_f1[1] * n1 / ti.max(h1, self._dx_eps) - \
                w_f2[1] * n2 / ti.max(h2, self._dx_eps)
            dtheta_dX3 = n1 / ti.max(h1, self._dx_eps)
            dtheta_dX4 = n2 / ti.max(h2, self._dx_eps)

            ret_val[0, :] = dtheta_dX1
            ret_val[1, :] = dtheta_dX2
            ret_val[2, :] = dtheta_dX3
            ret_val[3, :] = dtheta_dX4

        return ret_val

    @ti.func
    def _get_edge_bending_coeff_func(self, batch_idx, eid: int) -> float:
        """E_b = 0.5 * coeff * theta ^ 2"""
        ret_val = 0.0
        f1id, f2id = self._e2f[eid]
        v1id, v2id = self._e2v[eid]
        if f2id != -1:
            area12 = self._rest_area[f1id] + self._rest_area[f2id]
            edge_length = (self._pos_rest[v1id] - self._pos_rest[v2id]).norm()
            ret_val = self._kb[batch_idx] * edge_length ** 2 / \
                (4.0 * ti.max(area12, self._dx_eps ** 2))
        return ret_val

    @ti.kernel
    def _calculate_derivative_kernel(self):
        # calculate dF / dx
        for batch_idx, fid in ti.ndrange(self._batch_size, self._nf):
            n = self._get_face_normalized_func(batch_idx, fid)
            x0 = self._pos[batch_idx, self._f2v[fid][0]]
            x1 = self._pos[batch_idx, self._f2v[fid][1]]
            x2 = self._pos[batch_idx, self._f2v[fid][2]]

            h0 = maths.get_distance_vec_func(x0, x1, x2, False, self._dx_eps)
            h1 = maths.get_distance_vec_func(x1, x2, x0, False, self._dx_eps)
            h2 = maths.get_distance_vec_func(x2, x0, x1, False, self._dx_eps)

            for j, k in ti.ndrange(3, 3):
                for i in range(3):
                    v0 = self._TTT43[fid][3, k] * h0[j] * \
                        n[i] / ti.max(h0.norm_sqr(), self._dx_eps ** 2)
                    v1 = self._TTT43[fid][3, k] * h1[j] * \
                        n[i] / ti.max(h1.norm_sqr(), self._dx_eps ** 2)
                    v2 = self._TTT43[fid][3, k] * h2[j] * \
                        n[i] / ti.max(h2.norm_sqr(), self._dx_eps ** 2)

                    if i == j:
                        v0 += self._TTT43[fid][0, k]
                        v1 += self._TTT43[fid][1, k]
                        v2 += self._TTT43[fid][2, k]

                    self._dF_dx[batch_idx, fid].set(0, i, j, k, v0)
                    self._dF_dx[batch_idx, fid].set(1, i, j, k, v1)
                    self._dF_dx[batch_idx, fid].set(2, i, j, k, v2)

            self._deform_grad[batch_idx, fid] = self._get_deformation_gradient_func(batch_idx, fid)
            self._U_f[batch_idx, fid], self._S_f[batch_idx, fid], self._V_f[batch_idx, fid], \
                self._dU_f[batch_idx, fid], self._dS_f[batch_idx, fid], self._dV_f[batch_idx, fid] = \
                    maths.dsvd_func(self._deform_grad[batch_idx, fid], self._deform_grad_dsvd_tol)

    @ti.kernel
    def _calculate_ddpsi_dFdF_kernel(self):
        for batch_idx, fid in ti.ndrange(self._batch_size, self._nf):
            # Deformation Gradient F_f
            # F_f = self.get_deformation_gradient_func(fid)
            U_f, S_f, V_f, dU_f, dS_f, dV_f = self._U_f[batch_idx, fid], self._S_f[batch_idx, fid], self._V_f[batch_idx, fid], self._dU_f[batch_idx, fid], \
                self._dS_f[batch_idx, fid], self._dV_f[batch_idx, fid]

            # dSi = d psi / d Si
            dS = ti.Matrix.zero(float, n=3, m=3)
            for s in range(3):
                dS[s, s] = 2 * self._mu[batch_idx] * (S_f[s, s] - 1.0) + \
                    self._lda[batch_idx] * (S_f[0, 0] + S_f[1, 1] + S_f[2, 2] - 3.0)

            U_dS_V = U_f @ dS @ V_f.transpose()  # d psi / d F
            for k, l in ti.ndrange(3, 3):
                for w, i in ti.ndrange(3, 3):
                    coeff = U_dS_V[k, l] * self._h[batch_idx] * self._rest_area[fid]
                    self._elastic_force[batch_idx, self._f2v[fid][w]][i] -= coeff * \
                        self._get_dF_dx_func(batch_idx, fid, w, i, k, l)

            for m, n in ti.ndrange(3, 3):
                # calculate d / d Fmn (d psi / d F)
                dU_dF_mn = ti.Matrix.zero(float, n=3, m=3)
                dV_dF_mn = ti.Matrix.zero(float, n=3, m=3)

                for k, l in ti.ndrange(3, 3):
                    dU_dF_mn[k, l] = dU_f.get(k, l, m, n)
                    dV_dF_mn[k, l] = dV_f.get(k, l, m, n)

                ds_mn = ti.Matrix.zero(float, n=3, m=3)

                for j in range(3):
                    ds_mn[j, j] = dS_f.get(j, m, n) * 2.0 * self._mu[batch_idx] + \
                        (dS_f.get(0, m, n) + dS_f.get(1, m, n) +
                        dS_f.get(2, m, n)) * self._lda[batch_idx]

                ddpsi_dFdF_mn = U_f @ ds_mn @ V_f.transpose() + \
                    U_f @ dS @ dV_dF_mn.transpose() + \
                    dU_dF_mn @ dS @ V_f.transpose()

                for k, l in ti.ndrange(3, 3):
                    self._ddpsi_dFdF[batch_idx, fid, 3 * m + n, 3 * k + l] = \
                        ddpsi_dFdF_mn[k, l]
                        
        for batch_idx, fid, i, j in ti.ndrange(self._batch_size, self._nf, 9, 9):
            if i <= j:
                self._ddpsi_dFdF[batch_idx, fid, i, j] = (
                    self._ddpsi_dFdF[batch_idx, fid, i, j] +
                    self._ddpsi_dFdF[batch_idx, fid, j, i]) / 2
                self._ddpsi_dFdF[batch_idx, fid, j, i] = self._ddpsi_dFdF[batch_idx, fid, i, j]

    @sim_utils.GLOBAL_TIMER.timer
    def _calculate_derivative(self):
        self._calculate_derivative_kernel()
        self._calculate_ddpsi_dFdF_kernel()

    @ti.kernel
    def _fix_ddpsi_dFdF_kernel(self, alpha: float) -> bool:
        assert alpha >= 0.0, "[ERROR] in fix ddpsi_dFdF, displacement alpha:{} should not be negative.".format(
            alpha)
        self._is_all_success.fill(True)

        for batch_idx, fid in ti.ndrange(self._batch_size, self._nf):
            if not self._is_fixed_flag[batch_idx]:
                A_mat = maths.tiMatrix9x9()
                for i, j in ti.ndrange(9, 9):
                    A_mat.set(i, j, self._ddpsi_dFdF[batch_idx, fid, i, j])
                    assert ti.abs(
                        self._ddpsi_dFdF[batch_idx, fid, i, j] - self._ddpsi_dFdF[batch_idx, fid, j, i]) < 1e-6 * self._E[batch_idx]
                is_success, L_mat, D_vec = maths.ldlt_decompose_9x9_func(
                    A_mat, self._E[batch_idx] * self._rel_eps)
                if not is_success:
                    self._is_all_success[batch_idx] = False
                    for i in range(9):
                        self._ddpsi_dFdF[batch_idx, fid, i, i] += alpha

        # update is fixed flag:
        is_all_batch_fixed = True
        for batch_idx in range(self._batch_size):
            if self._is_all_success[batch_idx]:
                self._is_fixed_flag[batch_idx] = True
            if not self._is_fixed_flag[batch_idx]:
                is_all_batch_fixed = False
        return is_all_batch_fixed

    @sim_utils.GLOBAL_TIMER.timer
    def _fix_stretch_hessian(self):
        self._is_fixed_flag.fill(False)
        alpha0 = self._ldlt_loop_init_alpha
        for i in range(self._ldlt_max_loop_cnt):
            is_all_batch_fixed = self._fix_ddpsi_dFdF_kernel(float(alpha0 * (self._ldlt_loop_multiplier ** i)))
            if is_all_batch_fixed:
                break
            if i == self._ldlt_max_loop_cnt - 1:
                print("[ERROR] ldlt fix failed, there may exist some bugs.")

    @ti.kernel
    def _add_stretch_with_hessian_fixed_kernel(self):
        for batch_idx, fid, m, n, k, l in ti.ndrange(self._batch_size, self._nf, 3, 3, 3, 3):
            coeff = self._ddpsi_dFdF[batch_idx, fid, 3 * m + n, 3 * k + l] * \
                self._rest_area[fid] * self._h[batch_idx]
            for a, i in ti.ndrange(3, 3):
                for b, j in ti.ndrange(3, 3):
                    self._E_hessian_stretch[batch_idx, fid, a, b][i, j] += coeff * \
                        self._get_dF_dx_func(batch_idx, fid, a, i, m, n) * \
                        self._get_dF_dx_func(batch_idx, fid, b, j, k, l)

    @sim_utils.GLOBAL_TIMER.timer
    def _add_stretch_with_hessian(self):
        self._add_stretch_with_hessian_fixed_kernel()

    @ti.kernel
    def _add_bending_with_hessian_kernel(self):
        for batch_idx, eid in ti.ndrange(self._batch_size, self._ne):
            f1id, f2id = self._e2f[eid]
            v1id, v2id = self._e2v[eid]
            if f2id != -1:
                v3id = self._find_opp_vert_on_face_func(f1id, v1id, v2id)
                v4id = self._find_opp_vert_on_face_func(f2id, v1id, v2id)
                vert = ti.Vector([v1id, v2id, v3id, v4id])

                theta = self._get_dihedral_func(batch_idx, eid)
                theta_rest = self._rest_angle[eid]
                dtheta_dX = self._get_dtheta_dX_func(batch_idx, eid)
                coeff = self._get_edge_bending_coeff_func(batch_idx, eid)
                assert coeff >= 0.0, "[ERROR] in add bending hessian, coeff={} < 0.0".format(
                    coeff)
                for i, j in ti.ndrange(3, 4):
                    self._elastic_force[batch_idx, vert[j]][i] -= coeff * \
                        (theta - theta_rest) * dtheta_dX[j, i]
                    for k, l in ti.ndrange(3, 4):
                        self._E_hessian_bending[batch_idx, eid, j, l][i, k] += coeff * \
                            dtheta_dX[j, i] * dtheta_dX[l, k]
    
    @sim_utils.GLOBAL_TIMER.timer
    def _add_bending_with_hessian(self):
        self._add_bending_with_hessian_kernel()

    @ti.kernel
    def _assemble_hessian_kernel(self):
        # set zero
        self._hessian_sparse.set_zero_func()

        # assemble hessian
        for batch_idx, fid, i, j in ti.ndrange(self._batch_size, self._nf, 3, 3):
            vi, vj = self._f2v[fid][i], self._f2v[fid][j]
            for k, l in ti.ndrange(3, 3):
                self._hessian_sparse.add_value_func(
                    batch_idx, 3 * vi + k, 3 * vj + l, self._E_hessian_stretch[batch_idx, fid, i, j][k, l])

        for batch_idx, eid in ti.ndrange(self._batch_size, self._ne):
            f1id, f2id = self._e2f[eid]
            v1id, v2id = self._e2v[eid]
            if f2id != -1:
                v3id = self._find_opp_vert_on_face_func(f1id, v1id, v2id)
                v4id = self._find_opp_vert_on_face_func(f2id, v1id, v2id)
                vert = ti.Vector([v1id, v2id, v3id, v4id])
                for i, j in ti.ndrange(4, 4):
                    vi, vj = vert[i], vert[j]
                    for k, l in ti.ndrange(3, 3):
                        self._hessian_sparse.add_value_func(
                            batch_idx, 3 * vi + k, 3 * vj + l, self._E_hessian_bending[batch_idx, eid, i, j][k, l])
    
    @sim_utils.GLOBAL_TIMER.timer
    def _assemble_hessian(self):
        self._assemble_hessian_kernel()
                        
    @ti.kernel
    def _add_stretch_damping_force_kernel(self):
        for batch_idx, eid in ti.ndrange(self._batch_size, self._ne):
            v1id, v2id = self._e2v[eid]

            r1 = self._pos[batch_idx, v1id]
            r2 = self._pos[batch_idx, v2id]
            r12_normalized = maths.safe_normalized_func(r2 - r1, self._dx_eps)

            v1 = self._vel[batch_idx, v1id]
            v2 = self._vel[batch_idx, v2id]
            v1p = v1.dot(r12_normalized) * r12_normalized
            v2p = v2.dot(r12_normalized) * r12_normalized
            vp = v1p - v2p

            m_reduced = 1.0 / (1.0 / self._mass[batch_idx, v1id] +
                               1.0 / self._mass[batch_idx, v2id])
            f0 = m_reduced / self._stretch_relax_t[batch_idx]

            self._elastic_force[batch_idx, v1id] += -f0 * vp
            self._elastic_force[batch_idx, v2id] += +f0 * vp

    @ti.kernel
    def _add_bending_damping_force_kernel(self):
        for batch_idx, eid in ti.ndrange(self._batch_size, self._ne):
            v1id, v2id = self._e2v[eid]
            f1id, f2id = self._e2f[eid]
            if f2id != -1:
                v3id = self._find_opp_vert_on_face_func(f1id, v1id, v2id)
                v4id = self._find_opp_vert_on_face_func(f2id, v1id, v2id)

                x1 = self._pos[batch_idx, v1id]
                x2 = self._pos[batch_idx, v2id]
                x3 = self._pos[batch_idx, v3id]
                x4 = self._pos[batch_idx, v4id]

                h1 = maths.get_distance_func(x3, x1, x2, False, self._dx_eps)
                h2 = maths.get_distance_func(x4, x1, x2, False, self._dx_eps)

                n1 = self._get_face_normalized_func(batch_idx, f1id)
                n2 = self._get_face_normalized_func(batch_idx, f2id)

                w_f1 = maths.get_2D_barycentric_weights_func(x3, x1, x2, self._dx_eps)
                w_f2 = maths.get_2D_barycentric_weights_func(x4, x1, x2, self._dx_eps)

                dtheta_dX1 = - w_f1[0] * n1 / ti.max(h1, self._dx_eps) - \
                    w_f2[0] * n2 / ti.max(h2, self._dx_eps)
                dtheta_dX2 = - w_f1[1] * n1 / ti.max(h1, self._dx_eps) - \
                    w_f2[1] * n2 / ti.max(h2, self._dx_eps)
                dtheta_dX3 = n1 / ti.max(h1, self._dx_eps)
                dtheta_dX4 = n2 / ti.max(h2, self._dx_eps)

                omega = \
                    dtheta_dX1.dot(self._vel[batch_idx, v1id]) + \
                    dtheta_dX2.dot(self._vel[batch_idx, v2id]) + \
                    dtheta_dX3.dot(self._vel[batch_idx, v3id]) + \
                    dtheta_dX4.dot(self._vel[batch_idx, v4id])

                m_reduced = 1.0 / (1.0 / self._mass[batch_idx, v1id] +
                                   1.0 / self._mass[batch_idx, v2id] +
                                   1.0 / self._mass[batch_idx, v3id] +
                                   1.0 / self._mass[batch_idx, v4id])
                h_reduced = 1.0 / (1.0 / h1 + 1.0 / h2)
                edge_inertia = h_reduced ** 2 * m_reduced
                torq = -omega * edge_inertia / self._bending_relax_t[batch_idx]

                self._elastic_force[batch_idx, v3id] += torq * n1 / ti.max(h1, self._dx_eps)
                self._elastic_force[batch_idx, v4id] += torq * n2 / ti.max(h2, self._dx_eps)
                self._elastic_force[batch_idx, v1id] += torq * (-w_f1[0] * n1 / ti.max(h1, self._dx_eps) +
                                                                -w_f2[0] * n2 / ti.max(h2, self._dx_eps))
                self._elastic_force[batch_idx, v2id] += torq * (-w_f1[1] * n1 / ti.max(h1, self._dx_eps) +
                                                                -w_f2[1] * n2 / ti.max(h2, self._dx_eps))
                
    @sim_utils.GLOBAL_TIMER.timer
    def _add_damping_force(self):
        self._add_stretch_damping_force_kernel()
        self._add_bending_damping_force_kernel()

    def _calculate_elastic_force(self):
        """
        calculate elastic force and store answers in 
            - `self._elastic_force`
            - `self._hessian_sparse`
        """
        self._elastic_force.fill(0.0)
        self._E_hessian_stretch.fill(0.0)
        self._E_hessian_bending.fill(0.0)
        self._calculate_derivative()
        self._fix_stretch_hessian()
        self._add_stretch_with_hessian()
        self._add_bending_with_hessian()
        self._assemble_hessian()
        self._add_damping_force()

    #######################################################
    # Section: Miscellaneous
    #######################################################

    @ti.kernel
    def _calculate_vert_norm_kernel(self, batch_idx: int, vert_norm: ti.types.ndarray(dtype=ti.math.vec3)):
        for vid in range(self._nv):
            norm = ti.Vector.zero(float, 3)
            for fid_idx in range(self._v2f_cnt[vid]):
                fid = self._v2f[vid, fid_idx]
                norm += self._get_face_cross_product_func(batch_idx, fid)
            norm /= ti.max(norm.norm(), self._dx_eps)
            vert_norm[vid] = norm

    def _get_vert_norm(self, batch_idx: int) -> np.ndarray:
        vert_norm = torch.zeros((self._nv, 3), device=self._device, dtype=self._dtype)
        self._calculate_vert_norm_kernel(batch_idx, vert_norm)
        return sim_utils.torch_to_numpy(vert_norm)

    def _get_mesh(self, batch_idx: int, double_side: bool=False, vert_norm: bool=False) -> trimesh.Trimesh:
        if not double_side:
            mesh = trimesh.Trimesh(
                vertices=self._pos.to_numpy()[batch_idx], 
                faces=self._f2v.to_numpy(),
                vertex_normals=None if not vert_norm else self._get_vert_norm(batch_idx))
        else:
            vert = self._pos.to_numpy()[batch_idx]
            face = self._f2v.to_numpy()
            norm = self._get_vert_norm(batch_idx)
            mesh = trimesh.Trimesh(
                vertices=np.concatenate([vert, vert], axis=0), 
                faces=np.concatenate([face, face[:, [0, 2, 1]] + vert.shape[0]], axis=0),
                vertex_normals=None if not vert_norm else np.concatenate([+norm, -norm], axis=0))
        return mesh
    
    def _reset(self):
        self._pos.from_numpy(np.repeat(self._pos_rest.to_numpy()[None, ...], axis=0, repeats=self._batch_size))
        self._vel.fill(0.0)

    #######################################################
    # Section: APIs
    #######################################################

    @property
    def nv(self):
        return self._nv
    
    @property
    def nf(self):
        return self._nf
    
    @property
    def ne(self):
        return self._ne
    
    @property
    def name(self):
        return self._name

    @property
    def rest_mesh(self):
        return self._mesh
    
    @property
    def dx_eps(self):
        return self._dx_eps

    def get_f2v(self):
        """int, [F][3]"""
        return self._f2v.to_torch(device=self.device)
    
    def get_constraint(self) -> torch.Tensor:
        """[B, V, 3], 1.0 is free, 0.0 is fixed"""
        return self._constraint.to_torch(self._device)
    
    def set_constraint(self, constraint: torch.Tensor) -> None:
        """[B, V, 3]"""
        assert constraint.shape == (self._batch_size, self._nv, 3)
        constraint[torch.where(constraint != 0)] = 1.0
        return self._constraint.from_torch(constraint)
    
    def get_pos(self) -> torch.Tensor:
        """[B, V, 3]"""
        return self._pos.to_torch(self._device)
    
    def set_pos(self, pos: torch.Tensor) -> None:
        """[B, V, 3]"""
        assert pos.shape == (self._batch_size, self._nv, 3)
        return self._pos.from_torch(pos)
    
    def get_vel(self) -> torch.Tensor:
        """[B, V, 3]"""
        return self._vel.to_torch(self._device)
    
    def set_vel(self, vel: torch.Tensor) -> None:
        """[B, V, 3]"""
        assert vel.shape == (self._batch_size, self._nv, 3)
        return self._vel.from_torch(vel)
    
    def get_mesh(self, batch_idx: int, double_side: bool=False, vert_norm: bool=False, **kwargs) -> trimesh.Trimesh:
        return self._get_mesh(batch_idx, double_side, vert_norm)
    
    def get_external_force(self):
        """[B, V, 3] Get external force. Simulator will consume this force every step."""
        return self._external_force.to_torch(device=self.device)
    
    def get_external_hessian(self):
        """[B, V, 3, 3] Get external hessian. Simulator will consume this hessian every step."""
        return self._external_hessian.to_torch(device=self.device)
    
    def set_external_force(self, external_force: torch.Tensor):
        """[B, V, 3] Set external force. Simulator will consume this force every step."""
        assert external_force.shape == (self._batch_size, self._nv, 3)
        self._external_force.from_torch(external_force)
    
    def set_external_hessian(self, external_hessian: torch.Tensor):
        """[B, V, 3, 3] Set external hessian. Simulator will consume this hessian every step."""
        assert external_hessian.shape == (self._batch_size, self._nv, 3, 3)
        self._external_hessian.from_torch(external_hessian)

    def get_state(self) -> dict:
        state = {
            "pos": self._pos.to_numpy(),
            "vel": self._vel.to_numpy(),
            "constraint": self._constraint.to_numpy(),
        }
        return state
    
    def set_state(self, state: dict):
        assert isinstance(state, dict)
        self._pos.from_numpy(state["pos"])
        self._vel.from_numpy(state["vel"])
        self._constraint.from_numpy(state["constraint"])

    def reset(self):
        self._reset()