import numpy as np
import scipy
from copy import deepcopy

def project_bound(x, l, u):
    """
    Projects the vector x between upper and lower bounds u and l such that l<=P(x)<=u
    """
    return np.clip(x, l, u)

def project_linear(x, a, l, u):
    """
    Projects the vector x such that l<= a.T @ x <= u
    """
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


def project_set_convex(x, list_of_proj, max_iter=50, threshold=1e-5, verbose=False):
    """
    Projects onto the intersection of a set of convex functions.
    list_of_proj contains the projection functions in the form f(x), i.e.,
    all the other parameters need to be already defined for simplicity.
    """
    prim_res_norm = 1e6
    dual_res_norm = 1e6
    z = 0.
    nb_proj = len(list_of_proj)
    if nb_proj == 0:
        return x
    elif nb_proj == 1:
        return list_of_proj[0](x)
    else:
        x_p = np.repeat(x[np.newaxis], nb_proj, axis=0)
        lmb = np.zeros_like(x_p)

        for j in range(max_iter):
            x_p_bar = np.mean(x_p, axis=0)
            for i in range(nb_proj):
                x_p[i] = list_of_proj[i](x_p_bar - lmb[i])
            z_prev = deepcopy(z)
            z = np.mean(x_p, axis=0)
            prim_res = x_p - z
            prev_prim_res_norm = np.copy(prim_res_norm)
            prev_dual_res_norm = np.copy(dual_res_norm)
            prim_res_norm = np.linalg.norm(prim_res)**2
            dual_res_norm = np.linalg.norm(z - z_prev) ** 2

            lmb += prim_res
            if prim_res_norm < threshold and dual_res_norm < threshold:
                if verbose:
                    print("Converged at iteration ", j, "!")
                    print("Residual is ", "{:.2e}".format(prim_res_norm), "{:.2e}".format(dual_res_norm))
                break
            else:
                prim_change = np.abs(prev_prim_res_norm - prim_res_norm) / (prev_prim_res_norm + 1e-30)
                dual_change = np.abs(prev_dual_res_norm - dual_res_norm) / (prev_dual_res_norm + 1e-30)
                if prim_change < 1e-5 and dual_change < 1e-5:
                    if verbose:
                        print("Can't improve anymore at iteration ", j, "!")
                        print("Residual is ", "{:.2e}".format(prim_res_norm), "{:.2e}".format(dual_res_norm))
                        print("Residual change is ", "{:.2e}".format(prim_change), "{:.2e}".format(dual_change))
                    break
            if j == max_iter - 1:
                if verbose:
                    print("Residual is ", "{:.2e}".format(prim_res_norm), "{:.2e}".format(dual_res_norm))
                    print("Max iteration reached.")
        return z


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
        val = 0.5 * (x.T.dot(x))

        if val > u:
            z =  x * np.sqrt(2 * u)/ np.linalg.norm(x)
            zs = [z, -z]
            return zs[np.argmin([np.linalg.norm(z-x), np.linalg.norm(-z-x)])]
        elif l > val:
            z = x * np.sqrt(2 * l) / np.linalg.norm(x)
            zs = [z, -z]
            return zs[np.argmin([np.linalg.norm(z - x), np.linalg.norm(-z - x)])]
        else:
            return  x

def project_quadratic_batch(x,l,u):
    """
    Batch version of project_quadratic, x is a matrix
    """
    z = x.copy()
    val = 0.5 * np.sum(x * x, -1)
    cond1 = np.where(val > u)
    cond2 = np.where(l > val)
    x_norm  = np.linalg.norm(x, axis=-1)

    cond_close = np.where(x_norm<1e-3)
    x[cond_close] += np.random.normal(scale=1e-3,size=x[cond_close].shape)

    z[cond1] = x[cond1] * np.sqrt(2 * u)/ x_norm[cond1,None]
    z[cond2] = x[cond2] * np.sqrt(2 * l) / x_norm[cond2,None]
    # print(z)
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


def project_quadratic_A(x, A, l, u):
    """
    Projects the vector x such that l<= 0.5 * (x.T @ A @ x) <= u
    A : positive definite matrix
    """
    L = np.linalg.cholesky(A).T
    z = L @ x
    p_z = project_quadratic(z, l, u)
    return scipy.linalg.solve(L, p_z)

def project_quadratic_Ab(x, A, b, l, u):
    """
    Projects the vector x such that l<= 0.5 * (x.T @ A @ x) + b.T @ x <= u
    A : positive definite matrix
    b : vector
    """
    L = np.linalg.cholesky(A).T
    A_inv = np.linalg.inv(A)
    A_inv_b = A_inv @ b
    z = L @ (x +  A_inv_b)
    const = 0.5 * b.T.dot(A_inv).dot(b)
    p_z = project_quadratic(z, l + const, u + const )
    return np.linalg.solve(L, p_z) - A_inv_b




def project_soc(x, A, b, c, d):
    """
    Projects the vector x onto the second order cone (soc) such that ||Ax-b||<= c.Tx + d
    """
    if x.ndim == 2:
        return project_soc_batch(x, A, b, c, d)
    else:
        A_ = np.concatenate([A, c[None]], 0)
        b_ = np.append(b, d)

        z = x @ A.T + b
        t = c.T.dot(x) + d
        z_, t_ = project_soc_unit(z, t)
        proj = np.append(z_, t_)

        return scipy.linalg.lstsq(A_, proj-b_)[0]

def project_soc_batch(x, A, b, c, d):
    """
    Batch version of project_soc
    """
    A_ = np.concatenate([A, c[None]], 0)
    b_ = np.append(b, d)

    z = x @ A.T + b
    t = x @ c.T + d

    z_, t_ = project_soc_unit(z, t)
    proj = np.concatenate([z_, t_[:,None]], axis=1)
    return scipy.linalg.lstsq(A_, proj.T-b_[:,None])[0].T


def project_soc_unit(z,t):
    """
    Projects the vector z and t onto the second order cone (soc) such that ||z||<= t
    """
    if z.ndim == 2:
        return project_soc_unit_batch(z, t)
    else:

        z_norm = np.linalg.norm(z)

        if z_norm <= t:
            return z,t
        elif z_norm <= -t:
            return z*0, t*0
        else:
            tmp = (z_norm + t)/2
            return tmp*z/(z_norm+1e-30), tmp

def project_soc_unit_batch(z, t):
    """
    Batch version of project_soc_unit
    """
    z_norm = np.linalg.norm(z, axis=-1)

    cond1 = z_norm <= t
    cond2 = z_norm <= -t
    cond3 = (1-cond1-cond2).astype(bool)

    tmp = (z_norm + t) / 2

    z[cond2] = 0.
    t[cond2] = 0.

    z[cond3] = tmp[cond3, None] * z[cond3] / (z_norm[cond3, None]+1e-30)
    t[cond3] = tmp[cond3]
    return z, t


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

def project_square2(x, c, l, u):
    """
    Projects the vector x such that l <= ||x-c||_{\infty} <= u
    This defines the region defined by between two square regions centered at 0.
    """
    z = x - c
    z_ = project_square(z, l, u)
    return z_ + c

# Specialized functions
# def project_soc_batch(x, A_, b_, A_inv_):
#     """
#     Projects the batch vector x onto the second order cone (soc) such that ||Ax-b||<= c.Tx + d
#     A_ = [A ,c], b_ = [ b,d]
#     """
#     z_ = x.dot(A_.T) + b_
#     z__ = z_[:, :-1]
#     t__ = z_[:, -1]
#     z_norm = np.linalg.norm(z__, axis=-1)
#     z_[np.where(z_norm <= -t__), :] = 0.
#     ind = np.where(z_norm >= t__)
#     tmp = (z_norm + t__) / 2
#     z_[ind, -1] = tmp[ind]
#     # print(tmp.shape, z__.shape, z_.shape)
#     z_[ind, :-1] = (tmp[ind, None] * z__[ind] / z_norm[ind, None])
#     return (z_ - b_).dot(A_inv_.T)



def project_soc_batch2(x, sigma_sq, upper, psi):
    z =  x[:,1:] * sigma_sq
    t = (upper - x[:,0])/psi
    z_norm = np.linalg.norm(z, axis=-1)

    cond1 = z_norm <= t
    cond2 = z_norm <= -t
    cond3 = (1-cond1-cond2).astype(bool)

    tmp = (z_norm + t) / 2

    z[cond2] = 0.
    t[cond2] = 0.

    z[cond3] = tmp[cond3, None] * z[cond3] / (z_norm[cond3, None]+1e-30)
    t[cond3] = tmp[cond3]

    return np.concatenate([upper-t[:,None]*psi, z/(sigma_sq+1e-30)], axis=-1)

def project_soc_batch3(x, sigma_sq, lower, psi):
    z =  x[:,1:] * sigma_sq
    t = (-lower + x[:,0])/psi
    z_norm = np.linalg.norm(z, axis=-1)

    cond1 = z_norm <= t
    cond2 = z_norm <= -t
    cond3 = (1-cond1-cond2).astype(bool)

    tmp = (z_norm + t) / 2

    z[cond2] = 0.
    t[cond2] = 0.

    z[cond3] = tmp[cond3, None] * z[cond3] / (z_norm[cond3, None] + 1e-30)
    t[cond3] = tmp[cond3]

    return np.concatenate([t[:,None]*psi+lower, z/(sigma_sq+1e-30)], axis=-1)




# def project_set_convex_batch(x, list_of_proj, max_iter=50):
#     """
#     Batch version of project_set_convex
#     Projects onto the intersection of a set of convex functions.
#     list_of_proj contains the projection functions in the form f(x), i.e.,
#     all the other parameters need to be already defined for simplicity.
#     """
#     nb_proj = len(list_of_proj)
#     x_p = np.tile(np.copy(x)[None], (nb_proj, 1, 1))
#     lmb = np.zeros_like(x_p)
#
#     for j in range(max_iter):
#         x_p_bar = np.mean(x_p, axis=0)
#         for i in range(nb_proj):
#             x_p[i] = list_of_proj[i](x_p_bar - lmb[i])
#         z = np.mean(x_p + lmb, 0)
#         prim_res = x_p - z
#         lmb = lmb + prim_res
#         # norm_prim_res_prev = np.copy(norm_prim_res)
#         norm_prim_res  = np.linalg.norm(prim_res)
#         if norm_prim_res < 1e-3 or j == max_iter - 1:
#         # if np.abs(norm_prim_res - norm_prim_res_prev) < 1e-5 or j == max_iter - 1: #if the change is small
#         #     print(norm_prim_res)
#             break
#     return z

def project_block_lower_triangular(z, x_dim, u_dim, N):
    for i in range(N):
        z[i * u_dim, i * x_dim:(i + 1) * x_dim] = 0.
    return z

import torch
def project_soc_batch_torch(x, A_, b_, A_inv_):
    """
    Projects the batch vector x onto the second order cone (soc) such that ||Ax-b||<= c.Tx + d
    A_ = [A ,c], b_ = [ b,d]
    """
    z_ = x @ A_.T + b_
    z__ = z_[:, :-1]
    t__ = z_[:, -1]
    z_norm = torch.linalg.norm(z__, dim=-1)
    z_[torch.where(z_norm < -t__, True, False), :] = 0.
    ind = torch.where(z_norm > t__, True, False)
    tmp = (z_norm + t__) / 2
    z_[ind, -1] = tmp[ind]
    z_[ind, :-1] = (tmp[ind, None] * z__[ind] / z_norm[ind, None])
    return (z_ - b_) @ A_inv_.T
