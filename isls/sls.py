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




    def solve(self, verbose=False):
        """
        Solving System Level Synthesis problem.
        """
        assert self.A is not None, "Set the linear dynamics model by self.AB = [A,B] before calling this method."
        assert self.Q is not None, "Set the quadratic cost model by self.set_cost_variables() before calling this method."

        if type(self.Q) == sp.csc.csc_matrix:
            self.Q = self.Q.toarray()

        DTQ = self.D.T @ self.Q

        if self.l_side_invs is None:
            self.l_side_invs = self.compute_inverses(np.array(DTQ @ self.D + self.R))

        self.du = self.l_side_invs[0] @ DTQ @ self.xd
        self.dx = self.D @ self.du

        if verbose: print("Feedforward computed.")
        r_side = - DTQ @ self.C
        for i in range(self.N):
            if verbose: print("Computing feedback timestep ", i)
            phi_u = self.l_side_invs[i] @ r_side[i * self.u_dim:, i * self.x_dim:(i + 1) * self.x_dim]
            self.PHI_U[i * self.u_dim:, i * self.x_dim:(i + 1) * self.x_dim] = phi_u

        self.PHI_X = self.C + self.D @ self.PHI_U
        if verbose: print("Feedback computed.")

    def controller(self):
        """
        Computes the feedback gains K and the feedforward control commands k.
        """
        assert self.PHI_X is not None, "Run self.solve() before calling this method."

        K = np.array(self.PHI_U @ np.linalg.inv(self.PHI_X))
        k = (np.eye(self.D.shape[-1]) - K @ self.D) @ self.du
        return K, k


    def initialize_replanning_procedure(self, K ):
        self.replan_matrix = (np.eye(self.D.shape[-1]) - K @ self.D)@np.linalg.solve(self.D.T @ self.Q @ self.D + self.R, self.D.T @ self.Q)

    def replan_feedforward(self, k, xd):
        return k + self.replan_matrix.dot(xd - self.xd)

    def solve_DP(self):
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

        dim = self.x_dim + self.u_dim
        Ct = np.zeros((dim, dim))
        ct = np.zeros(dim)
        ft = np.zeros(self.x_dim)

        # Final timestep computations
        Q = self.Qs[self.seq[-1]]
        V = 2 * Q
        v = -2 * Q.dot(self.zs[self.seq[-1]])
        for t in range(self.N - 2, -1, -1):
            Q = self.Qs[self.seq[t]]

            Ct[:self.x_dim, :self.x_dim] = 2*Q
            Ct[self.x_dim:, self.x_dim:] = 2*self.Rt

            ct[:self.x_dim] = -2*Q.dot(self.zs[self.seq[t]])
            ct[self.x_dim:] = 0  # this part is nonzero in the case of regularized ADMM

            A = self.A
            B = self.B
            Ft = np.hstack([A, B])

            FtV = Ft.T.dot(V)
            Qt = Ct + FtV.dot(Ft)
            qt = ct + FtV.dot(ft) + Ft.T.dot(v)

            Quu = Qt[self.x_dim:, self.x_dim:]
            Qux = Qt[self.x_dim:, :self.x_dim]
            Qxx = Qt[:self.x_dim, :self.x_dim]
            Quu_inv = np.linalg.inv(Quu)

            qx = qt[:self.x_dim]
            qu = qt[self.x_dim:]

            Kt = -Quu_inv.dot(Qux)
            kt = -Quu_inv.dot(qu)
            K[t] = Kt
            k[t] = kt

            V = Qxx + Qux.T.dot(Kt) + Kt.T.dot(Qux) + Kt.T.dot(Quu).dot(Kt)
            v = qx + Qux.T.dot(kt) + Kt.T.dot(qu) + Kt.T.dot(Quu).dot(kt)

        return K, k


    #################################### Inequalities ######################################

    def ADMM_LQT_Batch(self, x0, list_of_proj_x=[], list_of_proj_u=[], max_iter=20, rho_x=0., rho_u=0., alpha=1.,
                       threshold=1e-3, warm_start=False, init_guess=None, verbose=False, log=False):
        """
        Solves LQT-ADMM in batch form.
        x0: initial state
        """

        if type(rho_x) == float:
            Qr = np.kron(np.eye(self.N), np.eye(self.x_dim)*rho_x)
        elif rho_x.shape[0] == self.x_dim:
            Qr = np.kron(np.eye(self.N), rho_x)
        else:
            Qr = rho_x

        # Initialize some values
        I = np.eye(self.R.shape[0])

        DQ = self.D.T @ self.Q
        DTD = self.D.T @ Qr @ self.D
        DQD = DQ @ self.D
        T = np.linalg.inv(DQD + self.R + I * rho_u + DTD)
        Sx_x0 = self.C[:, :self.x_dim] @ x0
        q = DQ.dot(self.xd - Sx_x0) - self.D.T @ Qr @ Sx_x0
        if warm_start:
            if init_guess is None:
                z_init = Sx_x0 + self.D @ np.linalg.solve(DQD + self.R, DQ.dot(self.xd - Sx_x0))
            else:
                z_init = init_guess
        else:
            z_init = None

        def f_argmin(x, u):
            u_hat = np.array(T.dot(q + rho_u * u + self.D.T @ Qr @ x))[0]
            x_hat = Sx_x0 + self.D @ u_hat
            return x_hat, u_hat

        return ADMM(self.x_dim * self.N, self.u_dim * self.N, f_argmin, list_of_proj_x, list_of_proj_u,alpha=alpha,
                       z_x_init=z_init, max_iter=max_iter, threshold=threshold, verbose=verbose, log=log)


    def ADMM_LQT_DP(self, x0, list_of_proj_x=[], list_of_proj_u=[], max_iter=2000,  rho_x=0., rho_u=0.,alpha=1.,
                    threshold=1e-3, verbose=False, log=False):
        """
        Solves LQT-ADMM with dynamic programming.
        """

        if type(rho_x) == float:
            Qr = np.tile(rho_x*np.eye(self.x_dim)[None], (self.N,1,1))
        elif rho_x.shape[0] == self.x_dim:
            Qr = np.tile(rho_x[None], (self.N,1,1))
        else:
            Qr = rho_x

        Rr = rho_u*np.eye(self.u_dim)

        dim = self.x_dim + self.u_dim
        A = self.A
        B = self.B
        Ft = np.hstack([A, B])

        # Compute static variables first (4 variables)
        V = np.zeros((self.N, self.x_dim, self.x_dim))
        Quu_inv = np.zeros((self.N, self.u_dim, self.u_dim))
        Qts = np.zeros((self.N, dim, dim))
        K = np.zeros((self.N, self.u_dim, self.x_dim))
        var1 = np.zeros((self.N, self.x_dim, self.u_dim))
        var2 = np.zeros((self.N, self.x_dim))

        Ct = np.zeros((dim, dim))
        Q = self.Qs[self.seq[-1]]
        V[-1] = 2 * (Q + Qr[-1])
        var2[-1] = -2 * self.Qs[self.seq[-1]].dot(self.zs[self.seq[-1]])
        for t in range(self.N - 2, -1, -1):
            Q = self.Qs[self.seq[t]]
            Ct[:self.x_dim, :self.x_dim] = 2 * (Q + Qr[t])
            Ct[self.x_dim:, self.x_dim:] = 2 * (self.Rt + Rr)

            FtV = Ft.T.dot(V[t+1])
            Qts[t] = Ct + FtV.dot(Ft)

            Quu = Qts[t, self.x_dim:, self.x_dim:]
            Qux = Qts[t, self.x_dim:, :self.x_dim]
            Qxx = Qts[t, :self.x_dim, :self.x_dim]
            Quu_inv[t] = np.linalg.inv(Quu)

            K[t] = -Quu_inv[t].dot(Qux)
            var1[t] = Qux.T + K[t].T.dot(Quu)
            V[t] = Qxx + var1[t].dot(K[t]) + K[t].T.dot(Qux)
            var2[t] = -2 * Q.dot(self.zs[self.seq[t]])

        k = np.zeros((self.N, self.u_dim))
        ct = np.zeros(dim)
        u_log = np.zeros((self.N, self.u_dim))
        x_log = np.zeros((self.N + 1, self.x_dim))
        x_log[0] = x0
        def f_argmin(x, u):
            xr = x.reshape(-1, self.x_dim)
            ur = u.reshape(-1, self.u_dim)

            v = var2[-1] - 2 * Qr[-1].dot(xr[-1])

            for t in range(self.N - 2, -1, -1):
                ct[:self.x_dim] = var2[t] - 2 * Qr[t].dot(xr[t])
                ct[self.x_dim:] = -2 * Rr.dot(ur[t])

                qt = ct + Ft.T.dot(v)

                k[t] = -Quu_inv[t].dot(qt[self.x_dim:])
                v = qt[:self.x_dim] + K[t].T.dot(qt[self.x_dim:]) + var1[t].dot(k[t])

            for i in range(self.N):
                u_log[i] = K[i].dot(x_log[i]) + k[i]
                x_log[i + 1] = self.A.dot(x_log[i]) + self.B.dot( u_log[i])
            return x_log[:-1].flatten(), u_log.flatten(), K, k
        return ADMM(self.x_dim * self.N, self.u_dim * self.N, f_argmin, list_of_proj_x, list_of_proj_u, alpha=alpha,
                       max_iter=max_iter, threshold=threshold, verbose=verbose, log=log)

    def ADMM(self, list_of_proj_x=[], list_of_proj_u=[], max_iter=5000, rho_x=0., rho_u=0., alpha=1.,threshold=1e-3, verbose=False,log=False):
        """
        Solves system level synthesis problem with ADMM.
        Robustness of control commands being in bounds with respect to a initial position distribution only.
        """
        if type(rho_x) == float:
            Qr = np.eye(self.x_dim * self.N) * rho_x
        elif rho_x.shape[0] == self.x_dim:
            Qr = np.kron(np.eye(self.N), rho_x)
        else:
            Qr = rho_x

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

        self.solve()
        # Init ADMM
        if log: logs = []

        dim_x = (self.x_dim*self.N, self.x_dim//2+1) # shape[1] is self.x_dim+1 because we have robustness only wrt x0 position
        dim_u = (self.u_dim*self.N, self.x_dim//2+1) # shape[1] is self.x_dim+1 because we have robustness only wrt x0 position
        z_x = np.zeros(dim_x)
        z_u = np.zeros(dim_u)
        lmb_x = 0
        lmb_u = 0

        x_x_tilde = np.zeros(dim_x)
        x_u_tilde = np.zeros(dim_u)

        x_x = 0.
        x_u = 0.

        prim_res_norm = 1e6
        dual_res_norm = 1e6
        print("Start iterating..")
        for j in range(max_iter):

            ## First step
            x_r = DTQr @ (z_x - lmb_x)
            u_r = rho_u * (z_u - lmb_u)

            # Optimize for feedforward terms
            x_u_tilde[:, 0] = AA_inv @ ( q_d + x_r[:,0] + u_r[:, 0])
            x_x_tilde[:, 0] = self.D @ x_u_tilde[:, 0]

            # Optimize for feedback terms
            # since we want robustness only wrt x0 ,we can only update the first block column.
            x_u_tilde[:, 1:] = np.matmul(AA_inv, r_side[:, :self.x_dim//2] + x_r[:, 1:] + u_r[:, 1:])
            x_x_tilde[:, 1:] = self.C[:, :dim_x[1]-1] + self.D @ x_u_tilde[:, 1:]

            x_x = alpha * x_x_tilde + (1 - alpha) * x_x
            x_u = alpha * x_u_tilde + (1 - alpha) * x_u

            ## Projection step
            z_prev_x = z_x
            z_prev_u = z_u

            z_x_ = alpha * x_x + (1 - alpha) * z_x
            z_u_ = alpha * x_u + (1 - alpha) * z_u

            z_x = project_set_convex(z_x_  + lmb_x , list_of_proj_x,max_iter=10, threshold=1e-3)
            z_u = project_set_convex(z_u_  + lmb_u , list_of_proj_u,max_iter=10, threshold=1e-3)

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
        phi_u = np.concatenate([x_u[:,1:dim_u[1]], self.PHI_U[:, dim_u[1]-1:] ], axis=-1)
        if log:
            return du, phi_u, logs
        else:
            return du, phi_u







