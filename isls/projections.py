import numpy as np
import scipy
from copy import deepcopy


########## PRIMITIVE PROJECTIONS ###############
def project_bound(x, l, u):
    """
    Projects the vector x between upper and lower bounds u and l such that l<=P(x)<=u
    """
    return np.clip(x, l, u)

def project_linear(x, a, l, u):
    """
    Projects the vector x such that l<= a.T @ x <= u
    """
    if x.ndim == 2:
        return project_linear_batch(x, a, l, u)
    else:
        aTx = a.dot(x)
        aTa = a.dot(a) + 1e-30
        if aTx > u:
            mu = aTx - u
        elif aTx < l:
            mu = aTx - l
        else:
            mu = 0.
        return x - mu*a/aTa

def project_linear_batch(x, a, l, u):
    """
    Batch version of project_linear
    """
    aTx = np.sum(x * a, axis=-1)
    aTa = np.sum(a * a, axis=-1) + 1e-30

    z = x.copy()
    cond1 = np.where(aTx > u)
    cond2 = np.where(aTx < l)
    tmp = a/aTa[:,None]
    z[cond1] = z[cond1] - (aTx - u)[cond1,None] * tmp[cond1]
    z[cond2] = z[cond2] - (aTx - l)[cond2,None] * tmp[cond2]
    return z


def project_multilinear(x,A,l,u):
    """
    Projects the vector x such that l<= A @ x <= u, where l and u are vectors, and A is a matrix
    Gives a projection that is on the boundary and not necessarily minimizing the norm as you would get
    from the projection onto a convex set
    """

    Ax = A.dot(x)
    tmp = Ax.copy()
    AAT_inv = np.linalg.inv(A@A.T)
    cond1 = np.where(Ax > u)
    cond2 = np.where(Ax < l)
    tmp[cond1] = u[cond1]
    tmp[cond2] = l[cond2]
    mu = AAT_inv @ (Ax - tmp)
    return x - A.T@mu


def project_affine(x, a, b, l, u):
    """
    Projects the vector x such that l<= a.T @ x + b <= u
    """
    return project_linear(x, a, l-b, u-b)

def project_quadratic(x,l,u):
    """
    Projects the vector x such that l<= 0.5 * (x.T @ x) <= u
    """
    if x.ndim == 2:
        return project_quadratic_batch(x, l, u)
    else:
        raise NotImplementedError
        # val = 0.5 * (x.T.dot(x))
        #
        # if val > u:
        #     z =  x * np.sqrt(2 * u)/ np.linalg.norm(x)
        #     zs = [z, -z]
        #     return zs[np.argmin([np.linalg.norm(z-x), np.linalg.norm(-z-x)])]
        # elif l > val:
        #     z = x * np.sqrt(2 * l) / np.linalg.norm(x)
        #     zs = [z, -z]
        #     return zs[np.argmin([np.linalg.norm(z - x), np.linalg.norm(-z - x)])]
        # else:
        #     return  x

def project_quadratic_batch(x,l,u):
    """
    Batch version of project_quadratic, x is a matrix
    """
    z = x.copy()
    val = 0.5 * np.sum(x * x, axis=-1)
    cond1 = np.where(val > u)
    cond2 = np.where(l > val)
    # x_norm  = np.linalg.norm(x, axis=-1)

    # cond_close = np.where(x_norm<1e-3)
    # x[cond_close] += np.random.normal(scale=1e-3,size=x[cond_close].shape)
    z[cond1] = x[cond1] * np.sqrt(2 * u)/ np.linalg.norm(x[cond1], axis=-1)[:,None]
    z[cond2] = x[cond2] * np.sqrt(2 * l)/ np.linalg.norm(x[cond2], axis=-1)[:,None]
    return z

def project_quadratic_b(x, b, l, u):
    """
    Projects the vector x such that l<= 0.5 * (x.T @ x) + b.T @ x <= u
    b : vector
    """
    z = x + b
    const = 0.5 * b.T.dot(b)
    p_z = project_quadratic(z, l + const, u + const )
    return p_z - b


def project_soc_unit(zt):
    """
    Projects the vector z and t onto the second order cone (soc) such that ||z||<= t
    """
    z = zt[...,:-1]
    t = zt[...,-1]
    if z.ndim == 2:
        z_, t_ = project_soc_unit_batch(z=z, t=t)
        return np.concatenate([z_, t_[:, None]], axis=1)

    else:
        z_norm = np.linalg.norm(z)
        if z_norm <= t:
            z_, t_ = z,t
        elif z_norm <= -t:
            z_, t_ = z*0, t*0
        else:
            tmp = (z_norm + t)/2
            z_, t_ = tmp*z/(z_norm+1e-30), tmp
        return np.append(z_, t_)


def project_soc_unit_batch(z, t):
    """
    Batch version of project_soc_unit
    """
    z_norm = np.linalg.norm(z, axis=-1)
    z_ = z.copy()
    t_ = t.copy()

    cond1 = np.logical_or(z_norm <= -t, t < 0)
    cond2 = np.logical_or(z_norm  >  t, z_norm > -t)
    cond3 = z_norm <= t

    tmp = (z_norm + t_) / 2

    z_[cond2] = tmp[cond2, None] * z[cond2] / (z_norm[cond2, None]+1e-30)
    t_[cond2] = tmp[cond2].copy()

    z_[cond1] = 0.
    t_[cond1] = 0.

    z_[cond3] = z[cond3]
    t_[cond3] = t[cond3]
    return z_, t_
def project_soc(z0, A, b, rho=1e0, max_iter=100, tol=1e-5, verbose=False):
    """
    Projects the vector x0 onto a second order cone s.t. Ax + b is in SOC.
    x0 = [nb_size, dim_x] or [dim_x]
    As = [dim_i, dim_x]
    bs = [dim_i]
    max_iter = 100
    threshold = 1e-5
    verbose = False
    """
    if z0.ndim == 1:
        nb_size = 1
        z0 = z0[None]
    else:
        nb_size = z0.shape[0]
    z = z0.T.copy()
    x = A @ z + b[:,None]
    lmb = np.zeros_like(x)

    l_side = np.eye(z0.shape[-1]) + rho * A.T @ A
    l_side_inv = np.linalg.inv(l_side)

    prim_res_norm_ = 1e5
    dual_res_norm_ = 1e5
    for j in range(max_iter):
        # print("Iteration", j)
        Az_b = A @ z + b[:,None]
        x = project_soc_unit((Az_b + lmb).T).T

        z_prev = z.copy()
        z = l_side_inv @ (z0.T + rho * A.T @ (- b[:,None] + x - lmb))

        Az_b = A @ z + b[:, None]
        prim_res = Az_b - x
        dual_res = rho*(z - z_prev)
        lmb = lmb + prim_res

        prim_res_norm = np.linalg.norm(prim_res, axis=0)
        dual_res_norm = np.linalg.norm(dual_res, axis=0)
        prev_prim_res_norm = np.copy(prim_res_norm_)
        prev_dual_res_norm = np.copy(dual_res_norm_)

        prim_res_norm_ = np.max(prim_res_norm)
        dual_res_norm_ = np.max(dual_res_norm)

        if prim_res_norm_ < tol and dual_res_norm_ < tol:
            if verbose:
                print("Project set convex converged at iteration ", j, "!")
                print("Residual is ", "{:.2e}".format(prim_res_norm_),
                      "{:.2e}".format(dual_res_norm_))
            break
        else:
            if j == max_iter - 1:
                if verbose:
                    print("Project set convex max iteration reached.")
                    print("Residual is ", "{:.2e}".format(prim_res_norm_),
                          "{:.2e}".format(dual_res_norm_))

            else:
                prim_change = np.abs(prev_prim_res_norm - prim_res_norm_) / (prev_prim_res_norm + 1e-30)
                dual_change = np.abs(prev_dual_res_norm - dual_res_norm_) / (prev_dual_res_norm + 1e-30)
                if prim_change < 1e-5 and dual_change < 1e-5:
                    if verbose:
                        print("Project set convex can't improve anymore at iteration ", j, "!")
                        print("Residual is ", "{:.2e}".format(prim_res_norm_),
                              "{:.2e}".format(dual_res_norm_))
                    break

    if nb_size == 1:
        return z.T[0]
    else:
        return z.T
def project_unit_ball(x):
    """
    Projects the vector x into the unit ball
    """
    x_norm = np.linalg.norm(x)
    if x_norm <= 1:
        return x
    else:
        return x/x_norm

def project_square(x, l, u):
    """
    Projects the vector x such that l <= ||x||_{\infty} <= u
    This defines the region defined by between two square regions centered at 0.
    """
    z = x.copy()
    j = np.argmax(np.abs(x))
    if np.abs(x[j]) < l:
        z[j] = l*np.sign(x[j])
    return np.maximum(np.minimum(z, u), -u)

def project_square_batch(x, l, u):
    """
    Projects the vector x such that l <= ||x||_{\infty} <= u
    This defines the region defined by between two square regions centered at 0.
    """
    z = x.copy()
    j = np.argmax(np.abs(x), axis=-1)
    x_inf_norm = np.linalg.norm(x, ord=np.inf, axis=-1)
    cond = np.where(x_inf_norm < l)
    z[(cond, j[cond])] = l * np.sign(x[(cond, j[cond])])
    return np.maximum(np.minimum(z, u), -u)

def project_square_c(x, c, l, u):
    """
    Projects the vector x such that l <= ||x-c||_{\infty} <= u
    This defines the region defined by between two square regions centered at 0.
    """
    z = x - c
    z_ = project_square(z, l, u)
    return z_ + c

def project_block_lower_triangular(z, x_dim, u_dim, N):
    """
    """
    for i in range(N):
        z[i * u_dim, i * x_dim:(i + 1) * x_dim] = 0.
    return z


projections = {"SOC": project_soc_unit, "bound": project_bound, "linear": project_linear,
               "quadratic": project_quadratic, "square":project_square}


def project_set_convex(x0, As=[], bs=[], projections=[], rho=1, max_iter=200, threshold=1e-4, verbose=False):
    """
    Projects the vector x0 onto the intersection of a set of second order cones.
    x0 = [nb_size, dim_x] or [dim_x]
    As = list of [dim_i, dim_x]
    bs = list of [dim_i]
    projections = a list of projection functions "bound", "linear", "quadratic", "SOC".
    max_iter = 100
    threshold = 1e-5
    verbose = False
    """
    nb_proj = len(projections)
    if x0.ndim == 1:
        nb_size = 1
        x0 = x0[None]
    else:
        nb_size = x0.shape[0]
    x=x0.T.copy()
    z = []
    lmb = []
    l_side = np.eye(x0.shape[-1])
    l_side_add = 0.
    for i in range(nb_proj):
        z += [As[i] @ x + bs[i][:,None]]
        lmb += [z[-1].copy()*0]
        l_side_add += As[i].T @ As[i]

    l_side_inv = np.linalg.inv(l_side + rho * l_side_add)

    prim_res_norm = np.ones((nb_proj, nb_size))*1e5
    dual_res_norm = np.ones((nb_proj, nb_size))*1e5
    prim_res_norm_ = 1e5
    dual_res_norm_ = 1e5
    for j in range(max_iter):
        r_side = 0.
        for i in range(nb_proj):
            r_side += As[i].T @ (- bs[i][:,None] + z[i] - lmb[i])
        x = l_side_inv @ (x0.T + rho * r_side)

        z_prev = z.copy()

        prev_prim_res_norm = np.copy(prim_res_norm_)
        prev_dual_res_norm = np.copy(dual_res_norm_)

        for i in range(nb_proj):
            Ax_b = As[i] @ x + bs[i][:,None]
            z[i] = projections[i]((Ax_b + lmb[i]).T).T

            prim_res = Ax_b - z[i]
            dual_res = rho*As[i].T @ (z[i] - z_prev[i])
            lmb[i] = lmb[i] + prim_res
            prim_res_norm[i] = np.linalg.norm(prim_res, axis=0)
            dual_res_norm[i] = np.linalg.norm(dual_res, axis=0)

        prim_res_norm_ = np.max(prim_res_norm)
        dual_res_norm_ = np.max(dual_res_norm)
        # prim_res_norm_ = np.linalg.norm(prim_res_norm)
        # dual_res_norm_ = np.linalg.norm(dual_res_norm)
        # if verbose: print(prim_res_norm,"\n", dual_res_norm, "\n",)
        if prim_res_norm_ < threshold and dual_res_norm_ < threshold:
            if verbose:
                print("Project set convex converged at iteration ", j, "!")
                print("Residual is ", "{:.2e}".format(prim_res_norm_),
                      "{:.2e}".format(dual_res_norm_))
            break
        else:
            if j == max_iter - 1:
                if verbose:
                    print("Project set convex max iteration reached.")
                    print("Residual is ", "{:.2e}".format(prim_res_norm_),
                          "{:.2e}".format(dual_res_norm_))

            else:
                prim_change = np.abs(prev_prim_res_norm - prim_res_norm_) / (prev_prim_res_norm + 1e-30)
                dual_change = np.abs(prev_dual_res_norm - dual_res_norm_) / (prev_dual_res_norm + 1e-30)
                if prim_change < 1e-5 and dual_change < 1e-5:
                    if verbose:
                        print("Project set convex can't improve anymore at iteration ", j, "!")
                        print("Residual is ", "{:.2e}".format(prim_res_norm_),
                              "{:.2e}".format(dual_res_norm_))
                    break

    if nb_size == 1:
        return x.T[0]
    else:
        return x.T
#
# def project_set_convex2(z0, As=[], bs=[], projections=[], rho=1, max_iter=200, threshold=1e-4, verbose=False):
#     """
#     Projects the vector x0 onto the intersection of a set of second order cones.
#     x0 = [nb_size, dim_x] or [dim_x]
#     As = list of [dim_i, dim_x]
#     bs = list of [dim_i]
#     projections = a list of projection functions "bound", "linear", "quadratic", "SOC".
#     max_iter = 100
#     threshold = 1e-5
#     verbose = False
#     """
#     nb_proj = len(projections)
#     if z0.ndim == 1:
#         nb_size = 1
#         z0 = z0[None]
#     else:
#         nb_size = z0.shape[0]
#     z=z0.T.copy()
#     x = []
#     lmb = []
#     l_side = np.eye(z0.shape[-1])
#     l_side_add = 0.
#     for i in range(nb_proj):
#         # x += [np.zeros((As[i].shape[0], nb_size))]
#         x += [As[i] @ z + bs[i][:,None]]
#         lmb += [x[-1].copy()*0.]
#         l_side_add += As[i].T @ As[i]
#
#     l_side_inv = np.linalg.inv(l_side + rho * l_side_add)
#
#     prim_res_norm = np.ones((nb_proj, nb_size))*1e5
#     dual_res_norm = np.ones((nb_proj, nb_size))*1e5
#     prim_res_norm_ = 1e5
#     dual_res_norm_ = 1e5
#     for j in range(max_iter):
#         # print("Iteration", j)
#         for i in range(nb_proj):
#             Az_b = As[i] @ z + bs[i][:,None]
#             x[i] = projections[i]((Az_b + lmb[i]).T).T
#
#         z_prev = z.copy()
#         r_side = 0.
#         for i in range(nb_proj):
#             r_side += As[i].T @ (- bs[i][:,None] + x[i] - lmb[i])
#         z = l_side_inv @ (z0.T + rho * r_side)
#         for i in range(nb_proj):
#             Az_b = As[i] @ z + bs[i][:, None]
#             prim_res = Az_b - x[i]
#             dual_res = rho * As[i].T @ (z - z_prev)
#             lmb[i] = lmb[i] + prim_res
#
#             prim_res_norm[i] = np.linalg.norm(prim_res, axis=0)
#             dual_res_norm[i] = np.linalg.norm(dual_res, axis=0)
#         prev_prim_res_norm = np.copy(prim_res_norm_)
#         prev_dual_res_norm = np.copy(dual_res_norm_)
#
#         prim_res_norm_ = np.max(prim_res_norm)
#         dual_res_norm_ = np.max(dual_res_norm)
#
#         if prim_res_norm_ < threshold and dual_res_norm_ < threshold:
#             if verbose:
#                 print("Project set convex converged at iteration ", j, "!")
#                 print("Residual is ", "{:.2e}".format(prim_res_norm_),
#                       "{:.2e}".format(dual_res_norm_))
#             break
#         else:
#             if j == max_iter - 1:
#                 if verbose:
#                     print("Project set convex max iteration reached.")
#                     print("Residual is ", "{:.2e}".format(prim_res_norm_),
#                           "{:.2e}".format(dual_res_norm_))
#
#             else:
#                 prim_change = np.abs(prev_prim_res_norm - prim_res_norm_) / (prev_prim_res_norm + 1e-30)
#                 dual_change = np.abs(prev_dual_res_norm - dual_res_norm_) / (prev_dual_res_norm + 1e-30)
#                 if prim_change < 1e-5 and dual_change < 1e-5:
#                     if verbose:
#                         print("Project set convex can't improve anymore at iteration ", j, "!")
#                         print("Residual is ", "{:.2e}".format(prim_res_norm_),
#                               "{:.2e}".format(dual_res_norm_))
#                     break
#
#     if nb_size == 1:
#         return z.T[0], x
#     else:
#         return z.T, x



def project_set_convex_dykstra(x0, projections=[],  max_iter=200, tol=1e-4, verbose=False):
    """
    Dykstra's projection algorithm
    Projects the vector x0 onto the intersection of a set of second order cones.
    x0 = [nb_size, dim_x] or [dim_x]
    As = list of [dim_i, dim_x]
    bs = list of [dim_i]
    projections = a list of projection functions "bound", "linear", "quadratic", "SOC".
    max_iter = 100
    threshold = 1e-5
    verbose = False
    """
    d = len(projections)
    if x0.ndim == 1:
        nb_size = 1
        x0 = x0[None]
    else:
        nb_size = x0.shape[0]
    u = x0.copy()
    z = np.zeros((d, nb_size, x0.shape[-1]))

    k = 0
    cI = np.stack([10.]*nb_size)
    while k<= max_iter and np.any(cI >= tol):
        # print("Iteration", n)
        cI = cI*0
        for i in range(d):
            prev_u = u.copy()
            u = projections[i](prev_u - z[i])
            prev_z = z[i].copy()
            z[i] = u - (prev_u - prev_z)

            # Stop condition
            cI += np.linalg.norm(prev_z - z[i], axis=-1)**2
        k += 1
    if k == max_iter+1:
        if verbose: print("Max iteration achieved.")
    else:
        if verbose: print("Converged at iteration", k)
    if verbose: print("Residual:","{:.2e}".format(np.max(cI)))
    return u