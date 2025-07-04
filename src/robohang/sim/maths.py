import taichi as ti


@ti.dataclass
class tiTensor333:
    """3x3x3 tensor. data stored as one 9x3 matrix"""
    data: ti.types.matrix(9, 3, float)

    @ti.func
    def get(self, k, i, j):
        return self.data[i * 3 + j, k]

    @ti.func
    def set(self, k, i, j, v):
        self.data[i * 3 + j, k] = v

    def get_shape(self):
        return (3, 3, 3)


@ti.dataclass
class tiTensor3333:
    """3x3x3x3 tensor. data stored as three 3x9 matrices"""
    data0: ti.types.matrix(3, 9, float)
    data1: ti.types.matrix(3, 9, float)
    data2: ti.types.matrix(3, 9, float)

    @ti.func
    def get(self, k, l, i, j):
        ret_val = 0.0
        if i == 0:
            ret_val = self.data0[j, k * 3 + l]
        elif i == 1:
            ret_val = self.data1[j, k * 3 + l]
        elif i == 2:
            ret_val = self.data2[j, k * 3 + l]
        else:
            assert False, "i={} out of range".format(i)
        return ret_val

    @ti.func
    def set(self, k, l, i, j, v):
        if i == 0:
            self.data0[j, k * 3 + l] = v
        elif i == 1:
            self.data1[j, k * 3 + l] = v
        elif i == 2:
            self.data2[j, k * 3 + l] = v
        else:
            assert False, "i={} out of range".format(i)

    def get_shape(self):
        return (3, 3, 3, 3)


@ti.dataclass
class tiMatrix9x9:
    """9x9 matrix. data stored as three 3x9 matrices"""
    data0: ti.types.matrix(3, 9, float)
    data1: ti.types.matrix(3, 9, float)
    data2: ti.types.matrix(3, 9, float)

    @ti.func
    def get(self, i, j):
        assert 0 <= i and i < 9 and 0 <= j and j < 9
        ret_val = 0.0
        i1 = i // 3
        i2 = i - i1 * 3
        if i1 == 0:
            ret_val = self.data0[i2, j]
        elif i1 == 1:
            ret_val = self.data1[i2, j]
        elif i1 == 2:
            ret_val = self.data2[i2, j]
        else:
            assert False, "[ERROR] tiMatrix9x9 error"
        return ret_val

    @ti.func
    def set(self, i, j, v):
        assert 0 <= i and i < 9 and 0 <= j and j < 9
        i1 = i // 3
        i2 = i - i1 * 3
        if i1 == 0:
            self.data0[i2, j] = v
        elif i1 == 1:
            self.data1[i2, j] = v
        elif i1 == 2:
            self.data2[i2, j] = v
        else:
            assert False, "[ERROR] tiMatrix9x9 error"


@ti.func
def ssvd_func(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V


@ti.func
def dsvd_func(F: ti.math.mat3, tol: float):
    """
    Calculate svd of F and derivative of U, S, V.
        - F = U @ S @ VT
        - dU[k,l,i,j] = partial Ukl / partial Fij
        - dS[k,i,j] = partial Sk / partial Fij
        - dV[k,l,i,j] = partial Vkl / partial Fij

    Args:
        - F: 3x3 ti.Matrix
        - tol: positive numerical error tolerance

    Return:
        - U: 3x3 ti.Matrix
        - S: 3x3 ti.Matrix
        - V: 3x3 ti.Matrix
        - dU: tiTensor3333
        - dS: tiTensor333
        - dV: tiTensor3333
    """
    U, S, V = ssvd_func(F)
    dU = tiTensor3333(0.0)
    dS = tiTensor333(0.0)
    dV = tiTensor3333(0.0)

    if ti.abs(S[0, 0] - S[1, 1]) < tol or ti.abs(S[1, 1] - S[2, 2]) < tol or \
            ti.abs(S[2, 2] - S[0, 0]) < tol:
        for i in range(3):
            for j in range(3):
                F[i, j] += ti.random() * tol
        U, S, V = ssvd_func(F)

    w01 = 0.0
    w02 = 0.0
    w12 = 0.0

    d01 = S[1, 1] * S[1, 1] - S[0, 0] * S[0, 0]
    d02 = S[2, 2] * S[2, 2] - S[0, 0] * S[0, 0]
    d12 = S[2, 2] * S[2, 2] - S[1, 1] * S[1, 1]

    if ti.abs(d01) < tol:
        d01 = 0.0
    else:
        d01 = 1.0 / d01

    if ti.abs(d02) < tol:
        d02 = 0.0
    else:
        d02 = 1.0 / d02

    if ti.abs(d12) < tol:
        d12 = 0.0
    else:
        d12 = 1.0 / d12

    for r in range(3):
        for s in range(3):
            Ur = ti.Vector([U[r, 0], U[r, 1], U[r, 2]])
            Vs = ti.Vector([V[s, 0], V[s, 1], V[s, 2]])
            UVT = Ur.outer_product(Vs)

            # Compute dS
            for i in range(3):
                dS.set(i, r, s, UVT[i, i])

            for i in range(3):
                UVT[i, i] -= dS.get(r, s, i)

            tmp = S @ UVT + UVT.transpose() @ S
            w01 = tmp[0, 1] * d01
            w02 = tmp[0, 2] * d02
            w12 = tmp[1, 2] * d12

            tmp = ti.Matrix(
                [[0.0, w01, w02], [-w01, 0.0, w12], [-w02, -w12, 0.0]])
            V_tmp = V @ tmp
            for i in range(3):
                for j in range(3):
                    dV.set(i, j, r, s, V_tmp[i, j])

            tmp = UVT @ S + S @ UVT.transpose()
            w01 = tmp[0, 1] * d01
            w02 = tmp[0, 2] * d02
            w12 = tmp[1, 2] * d12

            tmp = ti.Matrix(
                [[0.0, w01, w02], [-w01, 0.0, w12], [-w02, -w12, 0.0]])
            U_tmp = U @ tmp
            for i in range(3):
                for j in range(3):
                    dU.set(i, j, r, s, U_tmp[i, j])
    return U, S, V, dU, dS, dV


@ti.func
def ldlt_decompose_9x9_func(A_mat: tiMatrix9x9, tol: float):
    """
    LDLT decompose for 9x9 matrix.

    Args:
        - mat: tiMatrix9x9
        - tol: positive numerical error tolerance

    Return:
        - is_success: bool
        - L_mat: tiMatrix9x9
        - D_vec: 9D ti.Vector
    """
    M = ti.static(9)

    L_mat = tiMatrix9x9()
    D_vec = ti.Vector.zero(float, M)
    is_success = True
    for j in ti.static(range(M)):
        # calculate Dj
        tmp_sum = 0.0
        for k in ti.static(range(j)):
            tmp_sum += L_mat.get(j, k) ** 2 * D_vec[k]
        D_vec[j] = A_mat.get(j, j) - tmp_sum

        # calculate Lij
        L_mat.set(j, j, 1.0)
        if D_vec[j] <= tol:
            D_vec[j] = 0.0
            for i in ti.static(range(j + 1, M)):
                L_mat.set(i, j, 0.0)
            is_success = False

        else:
            for i in ti.static(range(j + 1, M)):
                tmp_sum = 0.0
                for k in ti.static(range(j)):
                    tmp_sum += L_mat.get(i, k) * L_mat.get(j, k) * D_vec[k]
                lij = (A_mat.get(i, j) - tmp_sum) / D_vec[j]
                L_mat.set(i, j, lij)

    return is_success, L_mat, D_vec


@ti.func
def safe_normalized_func(a, eps) -> ti.Vector:
    """
    Args:
        - a: ti.Vector 3D
        - eps: float

    Return:
        - a / ti.max(a.norm(), eps)"""
    return a / ti.max(a.norm(), eps)


@ti.func
def get_2D_barycentric_weights_func(x, a, b, dx_eps) -> ti.Vector:
    """
    Args:
        - x, a, b: ti.Vector
        - dx_eps: float

    Return:
        - 2D ti.Vector (u, v)
            - u + v = 1.0
            - u * a + v * b = proj(x)
    """
    e = b - a
    t = e.dot(x - a) / ti.max(e.norm_sqr(), dx_eps ** 2)
    return ti.Vector([1.0 - t, t])


@ti.func
def get_3D_barycentric_weights_func(x, a, b, c, dx_eps) -> ti.Vector:
    """
    Args:
        - x, a, b, c: ti.Vector
        - dx_eps: float

    Return:
        - 3D ti.Vector (u, v, w)
            - u + v + w = 1.0
            - u * a + v * b + w * c = proj(x)
    """
    n = safe_normalized_func((b - a).cross(c - a), dx_eps)

    u = (b - x).cross(c - x).dot(n)
    v = (c - x).cross(a - x).dot(n)
    w = (a - x).cross(b - x).dot(n)

    uvw = u + v + w
    dx_eps_sqr = dx_eps ** 2
    if uvw >= 0.0 and uvw < dx_eps_sqr:
        uvw = dx_eps_sqr
    elif uvw < 0.0 and uvw > - dx_eps_sqr:
        uvw = -dx_eps_sqr
    return ti.Vector([u / uvw, v / uvw, 1.0 - (u / uvw + v / uvw)])


@ti.func
def get_distance_func(x, a, b, is_segment, dx_eps) -> float:
    """
    Args:
        - x, a, b: ti.Vector
        - dx_eps: float

    Return:
        - float
    """
    bc = get_2D_barycentric_weights_func(x, a, b, dx_eps)
    if is_segment:
        bc = ti.math.clamp(bc, 0.0, 1.0)
    xp = bc[0] * a + bc[1] * b
    return (x - xp).norm()


@ti.func
def get_distance_vec_func(x, a, b, is_segment, dx_eps) -> ti.Vector:
    """
    Args:
        - x, a, b: ti.Vector
        - dx_eps: float

    Return:
        - ti.Vector, point from x to edge (proj(x) - x)
    """
    bc = get_2D_barycentric_weights_func(x, a, b, dx_eps)
    if is_segment:
        bc = ti.math.clamp(bc, 0.0, 1.0)
    xp = bc[0] * a + bc[1] * b
    return xp - x


@ti.func
def get_distance_to_triangle_func(x, a, b, c, dx_eps) -> ti.Vector:
    """
    Args:
        - x, a, b, c: ti.Vector
        - dx_eps: float

    Return: (l, u, v, w)
        - l = min(||x-d||), where d is in triangle abc
        - (u, v, w): barycentric coordinate of d
    """
    dist_a = ti.math.length(x - a)
    dist_b = ti.math.length(x - b)
    dist_c = ti.math.length(x - c)

    dist_ab = get_distance_func(x, a, b, True, dx_eps)
    dist_bc = get_distance_func(x, b, c, True, dx_eps)
    dist_ca = get_distance_func(x, c, a, True, dx_eps)

    dist = ti.min(dist_a, dist_b, dist_c, dist_ab, dist_bc, dist_ca)
    ret_bc = ti.Vector.zero(float, 3)
    if dist == dist_a:
        ret_bc = ti.Vector([1.0, 0.0, 0.0], float)
    elif dist == dist_b:
        ret_bc = ti.Vector([0.0, 1.0, 0.0], float)
    elif dist == dist_c:
        ret_bc = ti.Vector([0.0, 0.0, 1.0], float)
    elif dist == dist_ab:
        uu, vv = ti.math.clamp(get_2D_barycentric_weights_func(
            x, a, b, dx_eps), 0.0, 1.0)
        ret_bc = ti.Vector([uu, vv, 0.0], float)
    elif dist == dist_bc:
        vv, ww = ti.math.clamp(get_2D_barycentric_weights_func(
            x, b, c, dx_eps), 0.0, 1.0)
        ret_bc = ti.Vector([0.0, vv, ww], float)
    elif dist == dist_ca:
        ww, uu = ti.math.clamp(get_2D_barycentric_weights_func(
            x, c, a, dx_eps), 0.0, 1.0)
        ret_bc = ti.Vector([uu, 0.0, ww], float)
    u, v, w = get_3D_barycentric_weights_func(x, a, b, c, dx_eps)

    proj = u * a + v * b + w * c
    dist_p = ti.math.length(proj - x)

    if dist_p < dist and \
            0.0 < u and u < 1.0 and \
            0.0 < v and v < 1.0 and \
            0.0 < w and w < 1.0:
        dist = dist_p
        ret_bc = ti.Vector([u, v, w], float)

    return ti.Vector([dist, ret_bc[0], ret_bc[1], ret_bc[2]], float)


@ti.func
def get_edge_edge_barycentric_weights_func(a, b, c, d, dx_eps) -> ti.Vector:
    """
    [ua + (1 - u)b] - [vc + (1 - v)d] is perpendicular to (a - b) and (c - d)

    loss = [(a - b)u + (d - c)v + b - d]
    """
    x = a - b
    y = d - c
    z = b - d
    xx = x.dot(x)
    yy = y.dot(y)
    xy = x.dot(y)
    xz = x.dot(z)
    yz = y.dot(z)

    mat = ti.Matrix(
        [[xx, xy],
         [xy, yy]],
         dt=float,
    )
    vec = ti.Vector([-xz, -yz], dt=float)
    det = mat.determinant()

    uv = ti.Vector([.5, .5], dt=float)
    if det > dx_eps * dx_eps:
        uv = mat.inverse() @ vec
    else:
        uv = (mat + ti.Matrix.identity(dt=float, n=2) * dx_eps * dx_eps).inverse() @ vec

    return uv


@ti.func
def get_distance_edge_edge_func(a, b, c, d, dx_eps) -> ti.Vector:
    """
    Args:
        - a, b, c, d: ti.Vector
        - dx_eps: float

    Return: (l, u, v)
        - l = min(||x-y||), x is on ab, y is on cd
        - (u, v): barycentric coordinate of x and y:
            - x = [ua + (1 - u)b]
            - y = [vc + (1 - v)d]
    """
    dist_a = get_distance_func(a, c, d, True, dx_eps)
    dist_b = get_distance_func(b, c, d, True, dx_eps)
    dist_c = get_distance_func(c, a, b, True, dx_eps)
    dist_d = get_distance_func(d, a, b, True, dx_eps)

    u, v = ti.math.clamp(get_edge_edge_barycentric_weights_func(a, b, c, d, dx_eps), 0., 1.)
    dist_p = ti.math.length((u * a + (1. - u) * b) - (v * c + (1. - v) * d))

    dist = ti.min(dist_a, dist_b, dist_c, dist_d, dist_p)
    ret_bc = ti.Vector([u, v], float)
    if dist == dist_a:
        ret_bc = ti.Vector([1., ti.math.clamp(get_2D_barycentric_weights_func(a, c, d, dx_eps)[0], 0., 1.)], float)
    elif dist == dist_b:
        ret_bc = ti.Vector([0., ti.math.clamp(get_2D_barycentric_weights_func(b, c, d, dx_eps)[0], 0., 1.)], float)
    elif dist == dist_c:
        ret_bc = ti.Vector([ti.math.clamp(get_2D_barycentric_weights_func(c, a, b, dx_eps)[0], 0., 1.), 1.], float)
    elif dist == dist_d:
        ret_bc = ti.Vector([ti.math.clamp(get_2D_barycentric_weights_func(d, a, b, dx_eps)[0], 0., 1.), 0.], float)

    return ti.Vector([dist, ret_bc[0], ret_bc[1]], float)


@ti.func
def trilinear_4D_func(coor: ti.math.vec3, val: ti.types.matrix(8, 4, float)) -> ti.math.vec4:
    """trilinear interpolation.

    Args:
        - coor: 3x1 vector
        - val: 8x4 matrix

    Return:
        4D vector"""
    c = ti.Matrix.zero(dt=float, n=4, m=4)
    for j in ti.static(range(2)):
        for k in ti.static(range(2)):
            tmp = 2 * j + k
            c[tmp, :] = val[tmp, :] * (1 - coor[0]) + val[4 + tmp, :] * coor[0]

    d = ti.Matrix.zero(dt=float, n=2, m=4)
    for k in ti.static(range(2)):
        d[k, :] = c[k, :] * (1 - coor[1]) + c[2 + k, :] * coor[1]

    return d[0, :] * (1 - coor[2]) + d[1, :] * coor[2]


@ti.func
def vec_max_func(f: ti.template(), n: int):
    ret_val = f[0]
    for i in range(1, n):
        _ = ti.atomic_max(ret_val, f[i])
    return ret_val


@ti.kernel
def vec_max_int_kernel(f: ti.template(), n: int) -> int:
    return vec_max_func(f, n)


@ti.func
def vec_mul_vec_batch_func(batch_size, batch_mask, ans, a, b, n_field):
    """ans = a * b (element wise)"""
    max_n = vec_max_func(n_field, batch_size)
    for batch_idx, i in ti.ndrange(batch_size, max_n):
        if i < n_field[batch_idx] and batch_mask[batch_idx] != 0.0:
            ans[batch_idx, i] = a[batch_idx, i] * b[batch_idx, i]


@ti.kernel
def vec_mul_vec_batch_kernel(batch_size: int, batch_mask: ti.template(), ans: ti.template(), a: ti.template(), b: ti.template(), n_field: ti.template()):
    """ans = a * b (element wise)"""
    vec_mul_vec_batch_func(batch_size, batch_mask, ans, a, b, n_field)


@ti.func
def vec_dot_vec_batch_func(batch_size, batch_mask, a, b, ans, n_field):
    """ans = aT @ b"""
    for batch_idx in range(batch_size):
        if batch_mask[batch_idx] != 0.0:
            ans[batch_idx] = 0.0

    max_n = vec_max_func(n_field, batch_size)
    for batch_idx, i in ti.ndrange(batch_size, max_n):
        if i < n_field[batch_idx] and batch_mask[batch_idx] != 0.0:
            ans[batch_idx] += a[batch_idx, i] * b[batch_idx, i]


@ti.kernel
def vec_dot_vec_batch_kernel(batch_size: int, batch_mask: ti.template(), a: ti.template(), b: ti.template(), ans: ti.template(), n_field: ti.template()):
    """ans = aT @ b"""
    return vec_dot_vec_batch_func(batch_size, batch_mask, a, b, ans, n_field)


@ti.func
def block_mul_vec_batch_func(batch_size, batch_mask, ans, block, vec, n_field, block_dim):
    """
    M = block_dim

    ans[b, i * M + j] += block[b, i, j, k] * vec[b, j * M + k]
    """
    max_n = vec_max_func(n_field, batch_size)
    for batch_idx, i in ti.ndrange(batch_size, max_n):
        if i < n_field[batch_idx] and batch_mask[batch_idx] != 0.0:
            ans[batch_idx, i] = 0.0

    num_block = (max_n + block_dim - 1) // block_dim
    for batch_idx, i, j, k in ti.ndrange(batch_size, num_block, block_dim, block_dim):
        if batch_mask[batch_idx] != 0.0:
            jj = i * block_dim + j
            kk = i * block_dim + k
            if jj < n_field[batch_idx] and kk < n_field[batch_idx]:
                ans[batch_idx, jj] += block[batch_idx, i, j, k] * vec[batch_idx, kk]


@ti.kernel
def block_mul_vec_batch_kernel(batch_size: int, batch_mask: ti.template(), ans: ti.template(), block: ti.template(), vec: ti.template(), n_field: ti.template(), block_dim: int):
    """
    M = block_dim

    ans[b, i * M + j] += block[b, i, j, k] * vec[b, j * M + k]
    """
    block_mul_vec_batch_func(batch_size, batch_mask, ans, block, vec, n_field, block_dim)


@ti.func
def vec3_pad0_func(vec3: ti.math.vec3):
    vec4 = ti.Vector.zero(float, n=4)
    vec4[:3] = vec3
    return vec4


@ti.func
def vec3_pad1_func(vec3: ti.math.vec3):
    vec4 = ti.Vector.one(float, n=4)
    vec4[:3] = vec3
    return vec4