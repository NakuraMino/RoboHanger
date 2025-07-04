import taichi as ti


from . import maths


@ti.func
def stp_func(u: ti.math.vec3, v: ti.math.vec3, w: ti.math.vec3) -> float:
    return ti.math.dot(u, ti.math.cross(v, w))


@ti.func
def solve_quadratic_func(a: float, b: float, c: float, eps: float):
    x = ti.Vector.zero(float, 2)
    nsol = 0
    '''int solve_quadratic (double a, double b, double c, double x[2]) {
        // http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
        double d = b*b - 4*a*c;
        if (d < 0) {
            x[0] = -b/(2*a);
            return 0;
        }
        double q = -(b + sgn(b)*sqrt(d))/2;
        int i = 0;
        if (abs(a) > 1e-12*abs(q))
            x[i++] = q/a;
        if (abs(q) > 1e-12*abs(c))
            x[i++] = c/q;
        if (i==2 && x[0] > x[1])
            swap(x[0], x[1]);
        return i;
    }'''
    d = b * b - 4 * a * c
    if d < 0:
        x[0] = - b / (2 * a)
        nsol = 0
    else:
        q = -(b + ti.math.sign(b) * ti.sqrt(d)) / 2.
        i = 0
        if ti.abs(a) > eps * eps * ti.abs(q):
            x[0] = q / a
            i += 1
        if ti.abs(q) > eps * eps * ti.abs(c):
            if i == 0:
                x[0] = c / q
            else:
                x[1] = c / q
            i += 1
        if i == 2 and x[0] > x[1]:
            tmp = x[0]
            x[0] = x[1]
            x[1] = tmp
        nsol = i
    return x, nsol


@ti.func
def newtons_method_func(a: float, b: float, c: float, d: float, x0: float, init_dir: int, eps: float, iter_num: int) -> float:
    '''double newtons_method (double a, double b, double c, double d, double x0,
                        int init_dir) {
        if (init_dir != 0) {
            // quadratic approximation around x0, assuming y' = 0
            double y0 = d + x0*(c + x0*(b + x0*a)),
                ddy0 = 2*b + x0*(6*a);
            x0 += init_dir*sqrt(abs(2*y0/ddy0));
        }
        for (int iter = 0; iter < 100; iter++) {
            double y = d + x0*(c + x0*(b + x0*a));
            double dy = c + x0*(2*b + x0*3*a);
            if (dy == 0)
                return x0;
            double x1 = x0 - y/dy;
            if (abs(x0 - x1) < 1e-6)
                return x0;
            x0 = x1;
        }
        return x0;
    }'''
    if init_dir != 0:
        y0 = d + x0 * (c + x0 * (b + x0 * a))
        ddy0 = 2*b + x0*(6*a)
        x0 += init_dir * ti.sqrt(ti.abs(2 * y0 / ddy0))
    ti.loop_config(serialize=True)
    for _ in range(iter_num):
        y = d + x0 * (c + x0 * (b + x0 * a))
        dy = c + x0 * (2 * b + x0 * 3 * a)
        if dy == 0:
            break
        x1 = x0 - y / dy
        if ti.abs(x0 - x1) < eps:
            break
        x0 = x1
    return x0


@ti.func
def solve_cubic_func(a: float, b: float, c: float, d: float, eps: float, iter_num: int):
    x = ti.Vector.zero(float, 3)
    nsol = 0
    '''int solve_cubic (double a, double b, double c, double d, double x[3]) {
        double xc[2];
        int ncrit = solve_quadratic(3*a, 2*b, c, xc);
        if (ncrit == 0) {
            x[0] = newtons_method(a, b, c, d, xc[0], 0);
            return 1;
        } else if (ncrit == 1) {// cubic is actually quadratic
            return solve_quadratic(b, c, d, x);
        } else {
            double yc[2] = {d + xc[0]*(c + xc[0]*(b + xc[0]*a)),
                            d + xc[1]*(c + xc[1]*(b + xc[1]*a))};
            int i = 0;
            if (yc[0]*a >= 0)
                x[i++] = newtons_method(a, b, c, d, xc[0], -1);
            if (yc[0]*yc[1] <= 0) {
                int closer = abs(yc[0])<abs(yc[1]) ? 0 : 1;
                x[i++] = newtons_method(a, b, c, d, xc[closer], closer==0?1:-1);
            }
            if (yc[1]*a <= 0)
                x[i++] = newtons_method(a, b, c, d, xc[1], 1);
            return i;
        }
    }'''
    xc, ncrit = solve_quadratic_func(3 * a, 2 * b, c, eps)
    if ncrit == 0:
        x[0] = newtons_method_func(a, b, c, d, xc[0], 0, eps, iter_num)
        nsol = 1
    elif ncrit == 1:
        tmpx, tmpnsol = solve_quadratic_func(b, c, d, eps)
        x[:2] = tmpx
        nsol = tmpnsol
    else:
        yc = ti.Vector([d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
                        d + xc[1] * (c + xc[1] * (b + xc[1] * a))])
        i = 0
        if yc[0] * a >= 0:
            x[0] = newtons_method_func(a, b, c, d, xc[0], -1, eps, iter_num)
            i += 1
        if yc[0] * yc[1] <= 0:
            tmp = 0.
            if ti.abs(yc[0]) < ti.abs(yc[1]): # closer == 0
                tmp = newtons_method_func(a, b, c, d, xc[0], 1, eps, iter_num)
            else:
                tmp = newtons_method_func(a, b, c, d, xc[1], -1, eps, iter_num)
            if i == 0:
                x[0] = tmp
            else:
                x[1] = tmp
            i += 1
        if yc[1] * a <= 0:
            tmp = newtons_method_func(a, b, c, d, xc[1], 1, eps, iter_num)
            if i == 0:
                x[0] = tmp
            elif i == 1:
                x[1] = tmp
            else:
                x[2] = tmp
            i += 1
        nsol = i
    return x, nsol
    

'''bool collision_test (Impact::Type type, const Node *node0, const Node *node1,
                     const Node *node2, const Node *node3, Impact &impact) {
    impact.type = type;
    impact.nodes[0] = (Node*)node0;
    impact.nodes[1] = (Node*)node1;
    impact.nodes[2] = (Node*)node2;
    impact.nodes[3] = (Node*)node3;
    const Vec3 &x0 = node0->x0, v0 = node0->x - x0;
    Vec3 x1 = node1->x0 - x0, x2 = node2->x0 - x0, x3 = node3->x0 - x0;
    Vec3 v1 = (node1->x - node1->x0) - v0, v2 = (node2->x - node2->x0) - v0,
         v3 = (node3->x - node3->x0) - v0;
    double a0 = stp(x1, x2, x3),
           a1 = stp(v1, x2, x3) + stp(x1, v2, x3) + stp(x1, x2, v3),
           a2 = stp(x1, v2, v3) + stp(v1, x2, v3) + stp(v1, v2, x3),
           a3 = stp(v1, v2, v3);
    if (abs(a0) < 1e-6*norm(x1)*norm(x2)*norm(x3))
        return false; // initially coplanar
    double t[4];
    int nsol = solve_cubic(a3, a2, a1, a0, t);
    t[nsol] = 1; // also check at end of timestep
    for (int i = 0; i < nsol; i++) {
        if (t[i] < 0 || t[i] > 1)
            continue;
        impact.t = t[i];
        Vec3 x0 = pos(node0,t[i]), x1 = pos(node1,t[i]),
             x2 = pos(node2,t[i]), x3 = pos(node3,t[i]);
        Vec3 &n = impact.n;
        double *w = impact.w;
        double d;
        bool inside;
        if (type == Impact::VF) {
            d = signed_vf_distance(x0, x1, x2, x3, &n, w);
            inside = (min(-w[1], -w[2], -w[3]) >= -1e-6);
        } else {// Impact::EE
            d = signed_ee_distance(x0, x1, x2, x3, &n, w);
            inside = (min(w[0], w[1], -w[2], -w[3]) >= -1e-6);
        }
        if (dot(n, w[1]*v1 + w[2]*v2 + w[3]*v3) > 0)
            n = -n;
        if (abs(d) < 1e-6 && inside)
            return true;
    }
    return false;
}'''


@ti.func
def vert_face_ccd_func(pa: ti.math.vec3, pb: ti.math.vec3, pc: ti.math.vec3, pd: ti.math.vec3,
                       va: ti.math.vec3, vb: ti.math.vec3, vc: ti.math.vec3, vd: ti.math.vec3,
                       dt: float, eps: float, iter_num: int) -> float:
    """vert [a] -> face [b, c, d]"""
    x1 = pb - pa
    v1 = (vb - va) * dt
    x2 = pc - pa
    v2 = (vc - va) * dt
    x3 = pd - pa
    v3 = (vd - va) * dt
    a3 = stp_func(v3, v1, v2)
    a2 = (stp_func(x3, v1, v2) + 
          stp_func(v3, x1, v2) +
          stp_func(v3, v1, x2))
    a1 = (stp_func(x3, x1, v2) + 
          stp_func(v3, x1, x2) +
          stp_func(x3, v1, x2))
    a0 = stp_func(x3, x1, x2)
    collision_time = 1.
    if ti.abs(a0) > eps * (x1.norm_sqr() * x2.norm_sqr() * x3.norm_sqr()) * (1. / 3.):
        t, nsol = solve_cubic_func(a3, a2, a1, a0, eps, iter_num)
        ti.loop_config(serialize=True)
        for i in range(3):
            if i < nsol:
                if (t[i] < 0. or t[i] > 1.):
                    continue
                d, u, v, w = maths.get_distance_to_triangle_func(
                    pa + va * dt * t[i],
                    pb + vb * dt * t[i],
                    pc + vc * dt * t[i],
                    pd + vd * dt * t[i],
                    eps,
                )
                inside = (ti.min(u, v, w) > -eps)
                if ti.abs(d) < eps and inside:
                    collision_time = t[i]
                    break
    return collision_time


@ti.func
def edge_edge_ccd_func(pa: ti.math.vec3, pb: ti.math.vec3, pc: ti.math.vec3, pd: ti.math.vec3,
                       va: ti.math.vec3, vb: ti.math.vec3, vc: ti.math.vec3, vd: ti.math.vec3,
                       dt: float, eps: float, iter_num: int) -> float:
    """edge [a, b] -> edge [c, d]"""
    x1 = pb - pa
    v1 = (vb - va) * dt
    x2 = pc - pa
    v2 = (vc - va) * dt
    x3 = pd - pa
    v3 = (vd - va) * dt
    a3 = stp_func(v3, v1, v2)
    a2 = (stp_func(x3, v1, v2) + 
          stp_func(v3, x1, v2) +
          stp_func(v3, v1, x2))
    a1 = (stp_func(x3, x1, v2) + 
          stp_func(v3, x1, x2) +
          stp_func(x3, v1, x2))
    a0 = stp_func(x3, x1, x2)
    collision_time = 1.
    if ti.abs(a0) > eps * (x1.norm_sqr() * x2.norm_sqr() * x3.norm_sqr()) * (1. / 3.):
        t, nsol = solve_cubic_func(a3, a2, a1, a0, eps, iter_num)
        ti.loop_config(serialize=True)
        for i in range(3):
            if i < nsol:
                if (t[i] < 0. or t[i] > 1.):
                    continue
                d, u, v = maths.get_distance_edge_edge_func(
                    pa + va * dt * t[i],
                    pb + vb * dt * t[i],
                    pc + vc * dt * t[i],
                    pd + vd * dt * t[i],
                    eps,
                )
                inside = (ti.min(u, v) > -eps and ti.max(u, v) < 1. + eps)
                if ti.abs(d) < eps and inside:
                    collision_time = t[i]
                    break
    return collision_time
