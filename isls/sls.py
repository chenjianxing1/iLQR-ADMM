import scipy.sparse as sp

from .projections import *
from .sls_base import SLSBase
from .admm import ADMM


class SLS(SLSBase):
    def __init__(self, x_dim, u_dim, N, dtype=np.float64):
        """
        System Level Synthesis module for practical use.
        """
        super().__init__(x_dim, u_dim, N, dtype=dtype)




    def solve_batch(self, x0):
        """
        Computes

        Parameters
        ----------
        x0: initial state

        Returns
        -------
        x_opt, u_opt: optimal values for tbe state and the control
        """
        if type(self.Q) == sp.csc.csc_matrix:
            self.Q = self.Q.toarray()

        if self.l_side_invs is None:
            self.l_side_invs = self.compute_inverses(np.array(self.D.T @ self.Q @ self.D + self.R))

        C_x0 = self.C[:, :self.x_dim] @ x0
        u_opt = self.l_side_invs[0] @ self.D.T @ (self.Q @ self.xd - C_x0)
        x_opt = C_x0 + self.D @ u_opt
        return x_opt.reshape(self.N, -1), u_opt.reshape(self.N, -1)

    def solve_DP(self, Qr=None, Rr=None, ur=None, xr=None, return_Qs=False):
        """
        Solving LQT problem with dynamic programming,
        assuming that there is no cost multiplying x and u -> Cux = 0
        and no cost on in between timesteps.
        zt = [xt, ut]
        x{t+1} = Ft @ zt + ft
        cost_t = 0.5* zt.T @ Ct @ zt + zt.T @ ct
        """

        assert self.A is not None, "Set the linear dynamics model by self.AB = [A,B] before calling this method."
        assert self.Q is not None, "Set the quadratic cost model by self.set_cost_variables() before calling this method."

        K = np.zeros((self.N, self.u_dim, self.x_dim))
        k = np.zeros((self.N, self.u_dim))

        # Final timestep computations
        Q = self.Qs[self.seq[-1]]
        V = 2 * Q
        v = -2 * Q.dot(self.zs[self.seq[-1]])

        if Qr is not None:
            # Regularized problem for ADMM
            assert xr is not None
            xr_ = xr.reshape(self.N, -1)
            V += 2 * Qr[-1]
            v += -2 * Qr[-1].dot(xr_[-1])

        if Rr is not None:
            assert ur is not None
            ur_ = ur.reshape(self.N, -1)

        if return_Qs:
            Quu_log = np.zeros((self.N, self.u_dim, self.u_dim))
            Quu_inv_log = np.zeros((self.N, self.u_dim, self.u_dim))
            Qux_log = np.zeros((self.N, self.u_dim, self.x_dim))

        for t in range(self.N - 2, -1, -1):
            Q = self.Qs[self.seq[t]]

            Cxx = 2*Q
            Cuu = 2*self.Rt
            Cux = 0.

            cx = -2*Q.dot(self.zs[self.seq[t]])
            cu = 0.

            if Qr is not None:
                Cxx += 2 * Qr[t]
                cx  += - 2 * Qr[t].dot(xr_[t])
            if Rr is not None:
                Cuu += 2 * Rr[t]
                cu  += - 2 * Rr[t].dot(ur_[t])

            A = self.A
            B = self.B

            qx = cx + A.T.dot(v)
            qu = cu + B.T.dot(v)

            Qxx = Cxx + A.T.dot(V).dot(A)
            Qux = Cux + B.T.dot(V).dot(A)
            Quu = Cuu + B.T.dot(V).dot(B)

            Quu_inv = np.linalg.inv(Quu)
            Kt = -Quu_inv.dot(Qux)
            kt = -Quu_inv.dot(qu)

            V = Qxx + Qux.T.dot(Kt) + Kt.T.dot(Qux) + Kt.T.dot(Quu).dot(Kt)
            v = qx + Qux.T.dot(kt) + Kt.T.dot(qu) + Kt.T.dot(Quu).dot(kt)

            K[t] = Kt
            k[t] = kt

            if return_Qs:
                Quu_log[t] = Quu
                Quu_inv_log[t] = Quu_inv
                Qux_log[t] = Qux
        if return_Qs:
            return K, k, Quu_log, Quu_inv_log, Qux_log
        else:
            return K, k

    def solve_DP_ff(self, K, Quu, Qux, Quu_inv, Qr=None, Rr=None, ur=None, xr=None):

        k = np.zeros((self.N, self.u_dim))

        # Final timestep computations
        Q = self.Qs[self.seq[-1]]
        v = -2 * Q.dot(self.zs[self.seq[-1]])

        if Qr is not None:
            # Regularized problem for ADMM
            assert xr is not None
            xr_ = xr.reshape(self.N, -1)
            v += -2 * Qr[-1].dot(xr_[-1])

        if Rr is not None:
            assert ur is not None
            ur_ = ur.reshape(self.N, -1)

        for t in range(self.N - 2, -1, -1):
            Q = self.Qs[self.seq[t]]
            cx = -2 * Q.dot(self.zs[self.seq[t]])
            cu = 0.

            if Qr is not None:
                cx += - 2 * Qr[t].dot(xr_[t])
            if Rr is not None:
                cu += - 2 * Rr[t].dot(ur_[t])

            qx = cx + self.A.T.dot(v)
            qu = cu + self.B.T.dot(v)
            kt = -Quu_inv[t].dot(qu)
            v = qx + Qux[t].T.dot(kt) + K[t].T.dot(qu) + K[t].T.dot(Quu[t]).dot(kt)
            k[t] = kt

        return k

    ############################################### SLS ##############################################################
    def solve_sls(self, verbose=False):
        """
        Solving System Level Synthesis problem.
        """
        assert self.A is not None, "Set the linear dynamics model by self.AB = [A,B] before calling this method."
        assert self.Q is not None, "Set the quadratic cost model by self.set_cost_variables() before calling this method."
        PHI_U = np.zeros((self.N * self.u_dim, self.N * self.x_dim))

        if type(self.Q) == sp.csc.csc_matrix:
            self.Q = self.Q.toarray()

        DTQ = self.D.T @ self.Q

        if self.l_side_invs is None:
            self.l_side_invs = self.compute_inverses(np.array(DTQ @ self.D + self.R))

        du = self.l_side_invs[0] @ DTQ @ self.xd
        # self.dx = self.D @ self.du

        if verbose: print("Feedforward computed.")
        r_side = - DTQ @ self.C
        for i in range(self.N):
            if verbose: print("Computing feedback timestep ", i)
            phi_u = self.l_side_invs[i] @ r_side[i * self.u_dim:, i * self.x_dim:(i + 1) * self.x_dim]
            PHI_U[i * self.u_dim:, i * self.x_dim:(i + 1) * self.x_dim] = phi_u

        # self.PHI_X = self.C + self.D @ self.PHI_U
        if verbose: print("Feedback computed.")
        return PHI_U, du

    def controller(self, PHI_U, du):
        """
        Computes the feedback gains K and the feedforward control commands k.
        """
        PHI_X = self.C + self.D @ PHI_U
        K = PHI_U @ np.linalg.inv(PHI_X)
        k = (np.eye(self.D.shape[-1]) - K @ self.D) @ du
        return K, k

    def initialize_replanning_procedure(self, K ):
        self.replan_matrix = (np.eye(self.D.shape[-1]) - K @ self.D)@np.linalg.solve(self.D.T @ self.Q @ self.D + self.R, self.D.T @ self.Q)

    def replan_feedforward(self, k, xd):
        return k + self.replan_matrix.dot(xd - self.xd)


    ######################################### Inequalities #########################################################

    def ADMM_LQT_Batch(self, x0, list_of_proj_x=[], list_of_proj_u=[], max_iter=20, rho_x=None, rho_u=None, alpha=1.,
                       z_x_init=None, z_u_init=None,
                       threshold=1e-3,  verbose=False, log=False):
        """
        Solves LQT-ADMM in batch form.
        x0: initial state
        """

        Qr, Rr = self.compute_Rr_Qr(rho_x=rho_x, rho_u=rho_u, dp=False)
        # Initialize some values
        DQ = self.D.T @ self.Q
        DQD = DQ @ self.D
        l_side = DQD + self.R
        if Qr is not None:
            l_side += self.D.T @ Qr @ self.D
        if Rr is not None:
            l_side += Rr
        l_side_inv = np.array(np.linalg.inv(l_side))


        Sx_x0 = self.C[:, :self.x_dim] @ x0
        r_side = DQ.dot(self.xd - Sx_x0)
        if Qr is not None:
            r_side +=  - self.D.T @ Qr @ Sx_x0


        def f_argmin(x, u):
            r_side_ = r_side.copy()
            if Qr is not None: r_side_ += self.D.T @ Qr @ x
            if Rr is not None: r_side_ += Rr @ u
            u_hat = l_side_inv @ r_side_
            x_hat = Sx_x0 + self.D @ u_hat
            return x_hat, u_hat

        return ADMM(self.x_dim * self.N, self.u_dim * self.N, f_argmin, list_of_proj_x, list_of_proj_u,alpha=alpha,
                    z_x_init=z_x_init, z_u_init=z_u_init,
                        max_iter=max_iter, threshold=threshold, verbose=verbose, log=log)


    def ADMM_LQT_DP(self, x0, list_of_proj_x=[], list_of_proj_u=[], max_iter=2000, rho_x=None, rho_u=None,alpha=1.,
                    threshold=1e-3, verbose=False, log=False):
        """
        Solves LQT-ADMM with dynamic programming.
        """

        Qr, Rr = self.compute_Rr_Qr(rho_x=rho_x, rho_u=rho_u, dp=True)
        K, _, Quu_log, Quu_inv_log, Qux_log = self.solve_DP(Rr=Rr, Qr=Qr,
                                                                 xr=np.zeros(self.N * self.x_dim),
                                                                 ur=np.zeros(self.N * self.u_dim), return_Qs=True)

        def f_argmin(x, u):
            k = self.solve_DP_ff(K=K, Quu=Quu_log, Quu_inv=Quu_inv_log, Qux=Qux_log,
                                 Rr=Rr, Qr=Qr, xr=x, ur=u)
            x_nom, u_nom = self.get_trajectory_dp(x0, K, k)
            return x_nom.flatten(), u_nom.flatten(), K, k

        return ADMM(self.x_dim * self.N, self.u_dim * self.N, f_argmin, list_of_proj_x, list_of_proj_u, alpha=alpha,
                       max_iter=max_iter, threshold=threshold, verbose=verbose, log=log)

    def ADMM(self, list_of_proj_x=[], list_of_proj_u=[], max_iter=5000, rho_x=0., rho_u=0., alpha=1.,threshold=1e-3, verbose=False,log=False):
        """
        Solves system level synthesis problem with ADMM.
        Robustness of control commands being in bounds with respect to a initial position distribution only.
        """
        Qr, Rr = self.compute_Rr_Qr(rho_x=rho_x, rho_u=rho_u, dp=False)

        # Initialize some values
        DTQ = self.D.T @ self.Q
        DTQr = self.D.T @ Qr
        DTD =  DTQr @ self.D
        DTQD = DTQ @ self.D
        I = np.eye(self.R.shape[0])
        r_side = - DTQ @ self.C - DTQr @ self.C
        l_side_fb = np.array(DTQD + self.R) + rho_u * I + DTD

        q_d = DTQ @ self.xd
        l_side_invs = self.compute_inverses(l_side_fb)
        AA_inv = l_side_invs[0]

        PHI_U, du = self.solve_sls()

        # Init ADMM
        if log: logs = []

        dim_x = (self.x_dim*self.N, self.x_dim//2+1) # shape[1] is self.x_dim+1 because we have robustness only wrt x0 position
        dim_u = (self.u_dim*self.N, self.x_dim//2+1) # shape[1] is self.x_dim+1 because we have robustness only wrt x0 position
        z_x = np.zeros(dim_x)
        z_u = np.zeros(dim_u)


        z_u[:, 1:] = PHI_U[:, :self.x_dim//2]
        z_u[:, 0] = du
        z_x[:, 1:] = (self.C + self.D @ PHI_U)[:, :self.x_dim//2]
        z_x[:, 0] = self.D @ du

        lmb_x = 0
        lmb_u = 0

        x_x = z_x.copy()
        x_u = z_u.copy()

        prim_res_norm = 1e6
        dual_res_norm = 1e6
        print("Start iterating..")

        def f_argmin(x, u):
            x_r = DTQr @ x
            u_r = rho_u * u
            x_u[:, 0] = AA_inv @ ( q_d + x_r[:,0] + u_r[:, 0])
            x_x[:, 0] = self.D @ x_u[:, 0]
            x_u[:, 1:] = np.matmul(AA_inv, r_side[:, :self.x_dim//2] + x_r[:, 1:] + u_r[:, 1:])
            x_x[:, 1:] = self.C[:, :dim_x[1]-1] + self.D @ x_u[:, 1:]
            return x_x, x_u

        for j in range(max_iter):

            ## First step
            x_x, x_u = f_argmin(z_x - lmb_x, z_u - lmb_u)

            ## Projection step
            z_prev_x = z_x
            z_prev_u = z_u

            z_x_ = alpha * x_x + (1 - alpha) * z_x
            z_u_ = alpha * x_u + (1 - alpha) * z_u

            z_x = project_set_convex(z_x_  + lmb_x , list_of_proj_x, threshold=1e-2)
            z_u = project_set_convex(z_u_  + lmb_u , list_of_proj_u, threshold=1e-2)

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
                prim_change = np.abs(prev_prim_res_norm - prim_res_norm) / (prev_prim_res_norm + 1e-30)
                dual_change = np.abs(prev_dual_res_norm - dual_res_norm) / (prev_dual_res_norm + 1e-30)
                if prim_change < 1e-5 and dual_change < 1e-5:
                    print("Can't improve anymore at iteration ", j, "!")
                    print("Residual is ", "{:.2e}".format(prim_res_norm), "{:.2e}".format(dual_res_norm))
                    print("Residual change is ", "{:.2e}".format(prim_change), "{:.2e}".format(dual_change))
                    break
            if j == max_iter - 1:
                print("Residual is ", "{:.2e}".format(prim_res_norm), "{:.2e}".format(dual_res_norm))
                print("Max iteration reached.")

        du = x_u[:,0]
        phi_u = np.concatenate([x_u[:,1:dim_u[1]], PHI_U[:, dim_u[1]-1:] ], axis=-1)
        if log:
            return du, phi_u, logs
        else:
            return du, phi_u







