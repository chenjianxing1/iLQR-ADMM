from .projections import *

def ADMM(dim_x, dim_u, f_argmin, list_of_proj_x=[], list_of_proj_u=[],
         z_x_init=None, z_u_init=None, lmb_x_init=None, lmb_u_init=None,
         max_iter=20,alpha=1., threshold=1e-3, verbose=False, return_lmb=False, log=False):

    if log: logs = []

    z_x = np.zeros(dim_x) if z_x_init is None else z_x_init
    z_u = np.zeros(dim_u) if z_u_init is None else z_u_init

    lmb_x = np.zeros(dim_x) if lmb_x_init is None else lmb_x_init
    lmb_u = np.zeros(dim_u) if lmb_u_init is None else lmb_u_init

    prim_res_norm = 1e6
    dual_res_norm = 1e6
    for j in range(max_iter):

        reg_x = z_x - lmb_x
        reg_u = z_u - lmb_u
        _ = f_argmin(reg_x, reg_u)
        if not _:
            _ = (_,)
            print("unsuccesful first step of ADMM at iteration", j)
            break

        x_x = _[0]
        x_u = _[1]

        z_prev_x = z_x.copy()
        z_prev_u = z_u.copy()

        z_x_ = alpha * x_x + (1 - alpha) * z_x
        z_u_ = alpha * x_u + (1 - alpha) * z_u

        z_u = project_set_convex(z_u_ + lmb_u, list_of_proj_u)
        z_x = project_set_convex(z_x_ + lmb_x, list_of_proj_x)
        # print(x_x - z_x)
        # Dual update
        prim_res_x = z_x_ - z_x
        prim_res_u = z_u_ - z_u

        lmb_x += prim_res_x
        lmb_u += prim_res_u

        prev_prim_res_norm = np.copy(prim_res_norm)
        prev_dual_res_norm = np.copy(dual_res_norm)

        prim_res_norm = np.linalg.norm(prim_res_x) ** 2 + 0*np.linalg.norm(prim_res_u) ** 2
        dual_res_norm = np.linalg.norm(z_x - z_prev_x) ** 2 + np.linalg.norm(z_u - z_prev_u) ** 2

        if log: logs += [np.array([prim_res_norm,dual_res_norm])]
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
