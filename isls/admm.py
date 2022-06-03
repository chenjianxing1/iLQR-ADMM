from .projections import *


def ADMM(dim_x, dim_u, f_argmin, list_of_proj_x=[], list_of_proj_u=[],z_x_init=None, max_iter=20,alpha=1.,
            threshold=1e-3, verbose=False, log=False):

    if log: logs = []
    if z_x_init is None:
        z_x = np.zeros(dim_x)
    else:
        z_x = z_x_init
    lmb_x = np.zeros(dim_x)

    z_u = np.zeros(dim_u)
    lmb_u = np.zeros(dim_u)
    add_params = ()
    x_x = 0.
    x_u = 0.
    prim_res_norm = 1e6
    dual_res_norm = 1e6
    for j in range(max_iter):

        reg_x = z_x - lmb_x
        reg_u = z_u - lmb_u
        _ = f_argmin(reg_x, reg_u)

        add_params = _[2:]
        x_x_tilde = _[0]
        x_u_tilde = _[1]

        x_x = alpha * x_x_tilde + (1 - alpha) * x_x
        x_u = alpha * x_u_tilde + (1 - alpha) * x_u

        z_prev_x = z_x
        z_prev_u = z_u

        z_x_ = alpha * x_x + (1 - alpha) * z_x
        z_u_ = alpha * x_u + (1 - alpha) * z_u

        z_x = project_set_convex(z_x_ + lmb_x , list_of_proj_x)
        z_u = project_set_convex(z_u_ + lmb_u , list_of_proj_u)

        # Dual update
        prim_res_x = z_x_ - z_x
        prim_res_u = z_u_ - z_u

        lmb_x += prim_res_x
        lmb_u += prim_res_u

        prev_prim_res_norm = np.copy(prim_res_norm)
        prev_dual_res_norm = np.copy(dual_res_norm)

        prim_res_norm = np.linalg.norm(prim_res_x) ** 2 + np.linalg.norm(prim_res_u) ** 2
        dual_res_norm = np.linalg.norm(z_x - z_prev_x) ** 2 + np.linalg.norm(z_u - z_prev_u) ** 2

        if log: logs += [np.array([prim_res_norm,dual_res_norm])]
        if verbose: print(prim_res_norm, dual_res_norm)
        if prim_res_norm < threshold and dual_res_norm < threshold:  # or np.abs(prim_res_norm-prim_res_norm_prev) < 1e-5:
            print("Converged at iteration ", j, "!")
            print("Residual is ", "{:.2e}".format(prim_res_norm), "{:.2e}".format(dual_res_norm))
            break
        else:
            prim_change = np.abs(prev_prim_res_norm - prim_res_norm)/(prev_prim_res_norm+1e-30)
            dual_change = np.abs(prev_dual_res_norm - dual_res_norm)/(prev_dual_res_norm+1e-30)
            if  prim_change < 1e-5 and dual_change < 1e-5:
                print("Can't improve anymore at iteration ", j, "!")
                print("Residual is ", "{:.2e}".format(prim_res_norm), "{:.2e}".format(dual_res_norm))
                print("Residual change is ", "{:.2e}".format(prim_change), "{:.2e}".format(dual_change))
                break
        if j == max_iter - 1:
            print("Residuals-> primal:", "{:.2e}".format(prim_res_norm), "dual:","{:.2e}".format(dual_res_norm))
            print("Max iteration reached.")
    if log:
        return x_u,  logs, *add_params
    else:
        return x_u, add_params

