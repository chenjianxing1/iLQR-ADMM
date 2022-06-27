import numpy as np

from .projections import *


def ADMM(shape_x, shape_u, f_argmin, project_x=False, project_u=False,
         z_x_init=None, z_u_init=None, lmb_x_init=None, lmb_u_init=None, Qr=None, Rr=None,
         max_iter=20,alpha=1., threshold=1e-3, verbose=False, return_lmb=False, log=False):

    logs = []

    z_x = np.zeros(shape_x) if z_x_init is None else z_x_init
    z_u = np.zeros(shape_u) if z_u_init is None else z_u_init

    lmb_x = np.zeros(shape_x) if lmb_x_init is None else lmb_x_init
    lmb_u = np.zeros(shape_u) if lmb_u_init is None else lmb_u_init

    if not project_x:
        z_x = None
        lmb_x = None
    if not project_u:
        z_u = None
        lmb_u = None

    prim_res_norm = 1e6
    dual_res_norm = 1e6
    prim_res_x = 0.
    prim_res_u = 0.
    reg_x = None
    reg_u = None
    for j in range(max_iter):
        if project_x : reg_x = z_x - lmb_x
        if project_u : reg_u = z_u - lmb_u
        _ = f_argmin(reg_x, reg_u)
        if not _:
            _ = (_,)
            print("unsuccesful first step of ADMM at iteration", j)
            break

        x_x = _[0]
        x_u = _[1]

        prev_prim_res_norm = np.copy(prim_res_norm)
        prev_dual_res_norm = np.copy(dual_res_norm)

        if project_x:
            z_prev_x = z_x.copy()
            z_x_ = alpha * x_x + (1 - alpha) * z_x
            z_x = project_x(z_x_ + lmb_x)

            prim_res_x = x_x - z_x
            lmb_x += prim_res_x

        if project_u:
            z_prev_u = z_u.copy()
            z_u_ = alpha * x_u + (1 - alpha) * z_u
            z_u = project_u(z_u_ + lmb_u)
            prim_res_u = x_u - z_u
            lmb_u += prim_res_u


        prim_res_norm = 0.
        dual_res_norm = 0.
        if project_x:
            dual_res_norm += np.linalg.norm(z_x - z_prev_x)
            prim_res_norm += np.linalg.norm(prim_res_x)
        if project_u:
            dual_res_norm += np.linalg.norm(z_u - z_prev_u)
            prim_res_norm += + np.linalg.norm(prim_res_u)

        logs += [np.array([prim_res_norm,dual_res_norm])]
        if prim_res_norm < threshold and dual_res_norm < threshold:  # or np.abs(prim_res_norm-prim_res_norm_prev) < 1e-5:
            if verbose:
                print("ADMM converged at iteration ", j, "!")
                print("ADMM residual is ", "{:.2e}".format(prim_res_norm), "{:.2e}".format(dual_res_norm))
            break
        else:
            prim_change = np.abs(prev_prim_res_norm - prim_res_norm)/(prev_prim_res_norm+1e-30)
            dual_change = np.abs(prev_dual_res_norm - dual_res_norm)/(prev_dual_res_norm+1e-30)
            if  prim_change < threshold and dual_change < threshold:
                if verbose:
                    print("ADMM can't improve anymore at iteration ", j, "!")
                    print("ADMM residual is ", "{:.2e}".format(prim_res_norm), "{:.2e}".format(dual_res_norm))
                    print("ADMM residual change is ", "{:.2e}".format(prim_change), "{:.2e}".format(dual_change))
                break
            else:
                pass
                # if j >= 10:
                #     prim_osc = np.abs(logs[-4:] - np.mean(logs[-8:-4]))
                #     if np.abs(np.mean(logs[-4:]) - np.mean(logs[-8:-4])) < 1e-4:
                #         print("Cost is oscillating at iteration", j)
                #         break

        if j == max_iter - 1:
            if verbose:
                print("ADMM residuals-> primal:", "{:.2e}".format(prim_res_norm), "dual:","{:.2e}".format(dual_res_norm))
                print("ADMM: Max iteration reached.")

    return_vals = _

    if return_lmb:
        return_vals += (lmb_x, lmb_u, z_x, z_u)
    if log:
        return_vals += (logs, )

    return return_vals




