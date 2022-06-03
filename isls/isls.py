import scipy.sparse as sp
from .utils import find_precs, find_mus
from .admm import ADMM
from .projections import *
from .isls_base import iSLSBase
import matplotlib.pyplot as plt

class iSLS(iSLSBase):
    def __init__(self, x_dim, u_dim, N):
        super().__init__(x_dim, u_dim, N)

    def solve_ilqr(self, get_AB, max_ilqr_iter=50, max_line_search_iter=10, dp=True, verbose=False):
        for i in range(max_ilqr_iter):
            As, Bs = get_AB(self.x_nom, self.u_nom)
            self.AB = As, Bs

            if dp:
                success = self.iterate_once_dp_ilqr(max_iter=max_line_search_iter, verbose=verbose)
            else:
                success = self.iterate_once_batch_ilqr(max_iter=max_line_search_iter, verbose=verbose)

            if np.abs(np.diff(self.cost_log[-2:])) < 1e-5:
                print("Cost change is too low, cannot improve anymore at iteration", i+1, ".")
                break
            if not success:
                print("Forward pass failed, cannot improve anymore at iteration", i+1, ".")
                break
        if i == max_ilqr_iter - 1:
            print("Maximum iterations reached.")

    ######################################### Batch iLQR #################################################
    def batch_rollout_open_loop(self, x_nom, u_nom):
        """

        Parameters
        ----------
        x_nom : [N, x_dim]
        u_nom : [nb_batch, N, u_dim]

        Returns
        -------

        """

        nb_batch = u_nom.shape[0]
        x = np.tile(x_nom[0],(nb_batch, 1))
        x_log = np.zeros((nb_batch, self.N, self.x_dim))
        for i in range(self.N):
            x_log[:,i] = x
            x = self._forward_model(x, u_nom[:,i])
        return x_log, u_nom

    def iterate_once_batch_ilqr(self, max_iter=10, verbose=False):
        if type(self.Q) != sp.csc.csc_matrix:
            self.Q = sp.csc_matrix(self.Q)

        success = False
        # Backward Pass
        xd = self.xd - self.x_nom.flatten()
        ud = -self.u_nom.flatten()
        DTQ = self.D.T @ self.Q

        delta_u_opt = scipy.linalg.solve( DTQ @ self.D + self.R,
                                DTQ @ xd + self.R @ ud, assume_a='sym').reshape(self.N,-1)
        if verbose:
            print("Backward pass finished.")

        # Line Search
        delta_u = delta_u_opt[None]*self.alphas[:max_iter,None,None]
        x_noms, u_noms = self.batch_rollout_open_loop(self.x_nom, self.u_nom + delta_u)
        costs = self.cost_function(x_noms, u_noms)
        ind = np.argmin(costs)
        cost_optimal = costs[ind]

        # Convergence check
        if cost_optimal < self.cost:
            if verbose:
                print("We found a better solution with alpha ", self.alphas[ind], " with a new cost of ",
                      "{:.2e}".format(cost_optimal),
                      "which is smaller than the old cost of", "{:.2e}".format(self.cost))
            self.nominal_values = x_noms[ind], u_noms[ind]
            success = True

        return success


    ######################################### DP-iLQR #################################################

    def solve_DP(self):
        """
        Assuming that there is no cost multiplying x and u -> Cux = 0
        zt = [xt, ut]
        x{t+1} = Ft @ zt + ft
        cost_t = 0.5* zt.T @ Ct @ zt + zt.T @ ct
        """

        K = np.zeros((self.N, self.u_dim, self.x_dim))
        k = np.zeros((self.N, self.u_dim))

        dim = self.x_dim + self.u_dim
        Ct = np.zeros((dim, dim))
        ct = np.zeros(dim)
        ft = np.zeros(self.x_dim)

        # Final timestep computations
        Q = self.Qs[self.seq[-1]]
        V = 2 * Q
        v = -2 * Q.dot(self.zs[self.seq[-1]] - self.x_nom[-1])
        for t in range(self.N - 2, -1, -1):
            Q = self.Qs[self.seq[t]]

            Ct[:self.x_dim, :self.x_dim] = 2 * Q
            Ct[self.x_dim:, self.x_dim:] = 2 * self.Rt

            ct[:self.x_dim] = -2 * Q.dot(self.zs[self.seq[t]] - self.x_nom[t])
            ct[self.x_dim:] = -2 * self.Rt.dot(-self.u_nom[t])

            A = self.A[t]
            B = self.B[t]
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

    def batch_DP_rollout(self, K, k):
        """
        k : [nb_batch, N, u_dim]
        x_nom : [N, x_dim]
        u_nom : [N, u_dim]
        """
        nb_batch = k.shape[0]
        x = np.tile(self.x_nom[0],(nb_batch, 1))
        x_log = np.zeros((nb_batch, self.N, self.x_dim))
        u_log = np.zeros((nb_batch, self.N, self.u_dim))
        for i in range(self.N):
            dx_vec = x - self.x_nom[i]
            u = dx_vec @ K[i].T
            u += k[:,i] + self.u_nom[i]
            u_log[:, i] = u
            x_log[:, i] = x
            x = self.forward_model(x, u)

        return x_log, u_log

    def iterate_once_dp_ilqr(self, max_iter=15, verbose=False):

        success = False

        # Backward Pass
        K,k = self.solve_DP()
        if verbose: print("Backward pass finished.")

        # Line Search
        k_new = k[None] * self.alphas[:max_iter,None,None]
        x_noms, u_noms = self.batch_DP_rollout(K, k_new)

        # Convergence check
        costs = self.cost_function(x_noms, u_noms)
        ind = np.argmin(costs)
        if costs[ind] < self.cost:
            if verbose:
                print("We found a better solution with alpha ", self.alphas[ind], " with a new cost of ",
                      "{:.2e}".format(costs[ind]),
                      "which is smaller than the old cost of", "{:.2e}".format(self.cost))
            self.nominal_values = x_noms[ind], u_noms[ind]
            success = True

        return success




    ################################## ADMM #########################################
    def iterate_once_batch_ilqr_reg(self, rho_u, Qr, x_reg, u_reg, max_iter=15):
        if type(self.Q) != sp.csc.csc_matrix:
            self.Q = sp.csc_matrix(self.Q)

        I = np.eye(self.R.shape[0])
        x_nom_ = self.x_nom.flatten()
        u_nom_ = self.u_nom.flatten()
        success = False

        # Backward Pass
        xd = self.xd - x_nom_
        ud = -u_nom_
        x_reg_ = x_reg - x_nom_
        u_reg_ = u_reg - u_nom_
        DTQ = self.D.T @ self.Q
        DTD = self.D.T @ Qr @ self.D

        delta_u_opt = scipy.linalg.solve(DTQ @ self.D + self.R + I * rho_u + DTD,
                                         DTQ @ xd + self.R @ ud + rho_u * u_reg_ + self.D.T @ Qr @ x_reg_,
                                         assume_a='sym').reshape(self.N, -1)

        # Line Search
        delta_u = delta_u_opt[None] * self.alphas[:max_iter, None, None]
        x_noms, u_noms = self.batch_rollout_open_loop(self.x_nom, self.u_nom + delta_u)
        costs = self.cost_function(x_noms, u_noms)

        # Convergence check
        ind = np.argmin(costs)
        if costs[ind] < self.cost:
            self.nominal_values = x_noms[ind], u_noms[ind]
            success = True

        return success

    def ilqr_admm(self, get_AB, list_of_proj_x=[], list_of_proj_u=[], z_init=None, max_admm_iter=20,
                        max_line_search=20, rho_x=0., rho_u=0., alpha=1, threshold=1e-3, verbose=False, log=False):

        if type(rho_x) == float:
            rho_x = np.eye(self.x_dim*self.N)*rho_x
        elif rho_x.shape[0] == self.x_dim:
            rho_x = np.kron(np.eye(self.N), rho_x)

        def f_argmin(x, u):
            As, Bs = get_AB(self.x_nom, self.u_nom)

            self.AB = As, Bs
            success = self.iterate_once_batch_ilqr_reg(rho_u=rho_u, Qr=rho_x, x_reg=x, u_reg=u, max_iter=max_line_search)
            return self.x_nom.flatten(), self.u_nom.flatten()

        _ = ADMM(self.x_dim * self.N, self.u_dim * self.N, f_argmin, list_of_proj_x, list_of_proj_u,alpha=alpha,
                z_x_init=z_init, max_iter=max_admm_iter, threshold=threshold, verbose=verbose, log=log)
        if log:
            return _[-1]
        else:
            return 1

    ######################################### iSLS #################################################


    def rollout(self, K, k, x_nom, u_nom, verbose=False):
        x = x_nom[0]
        x_log = []
        u_log = []
        dx_vec = np.zeros(self.N*self.x_dim)
        for i in range(self.N):
            dx_vec[i * self.x_dim:(i + 1) * self.x_dim] = x - x_nom[i]
            u = K[i*self.u_dim:(i+1)*self.u_dim, :(i+1)*self.x_dim].dot(dx_vec[:(i+1)*self.x_dim])
            u += k[i * self.u_dim:(i + 1) * self.u_dim] + u_nom[i]

            if verbose:
                print(x, u)

            x_log += [x]
            u_log += [u]
            x = self._forward_model(x[None], u[None])[0]

        x_log = np.stack(x_log)
        u_log = np.stack(u_log)
        return x_log, u_log

    def batch_rollout(self, K, k, x_nom, u_nom):
        """
        k : [nb_batch, N*u_dim]
        x_nom : [N, x_dim]
        u_nom : [N, u_dim]
        """
        nb_batch = k.shape[0]
        x = np.tile(x_nom[0],(nb_batch, 1))
        x_log = np.zeros((nb_batch, self.N, self.x_dim))
        u_log = np.zeros((nb_batch, self.N, self.u_dim))
        dx_vec = np.zeros((nb_batch, self.N*self.x_dim))
        for i in range(self.N):
            dx_vec[:, i * self.x_dim:(i + 1) * self.x_dim] = x - x_nom[i]
            u = dx_vec[:, :(i+1)*self.x_dim] @ K[i*self.u_dim:(i+1)*self.u_dim, :(i+1)*self.x_dim].T
            u += k[:, i * self.u_dim:(i + 1) * self.u_dim] + u_nom[i]
            x_log[:, i] = x
            u_log[:, i] = u
            x = self._forward_model(x, u)

        return x_log, u_log


    def backward_pass(self, cx=None, Cxx=None):
        if cx is None: # means quadratic cost
            xd = self.xd-self._x_nom.flatten()
            ud = -self._u_nom.flatten()
            DTQ = self.D.T @ self.Q
            l_side = DTQ @ self.D + self.R
            r_side = - DTQ @ self.C
            self.du = scipy.linalg.solve(l_side, DTQ @ xd + self.R @ ud, assume_a='sym')
            print("Feedforward computed")
            for i in range(self.N):
                AA = l_side[i * self.u_dim:, i * self.u_dim:]
                BB = r_side[i * self.u_dim:, i * self.x_dim:(i + 1) * self.x_dim]
                phi_u = scipy.linalg.solve(AA, BB, assume_a='sym')
                self.PHI_U[i * self.u_dim:, i * self.x_dim:(i + 1) * self.x_dim] = phi_u

        else:
            self.Q = Cxx
            self.solve(only_feedback=True)
            self.du = sp.linalg.spsolve(self.D.T @ self.Q @ self.D + self.R, - self.D.T @ cx - self.R @ self._u_nom.flatten())
            self.dx = self.D @ self.du
        print("Backward pass done.")
        # PHI_X = self.C + self.D @ self.PHI_U
        self.K = scipy.linalg.solve_triangular((self.C + self.D @ self.PHI_U).T, self.PHI_U.T).T
        self.k = self.du -self.K.dot(self.D.dot(self.du))
        print("Controller computed.")



    def forward_pass(self, max_iter=15):
        # Line Search
        k_new = self.k[None] * self.alphas[:max_iter,None]
        x_noms, u_noms = self.batch_rollout(self.K, k_new, self._x_nom, self._u_nom)
        costs = self.cost_function(x_noms, u_noms)
        ind = np.argmin(costs)
        success = False
        # print(costs-self.cur_cost)
        if costs[ind] < self.cur_cost:
            print("We found a better solution with alpha ", self.alphas[ind], " with a new cost of ",
                  "{:.2e}".format(costs[ind]),
                  "which is smaller than the old cost of", "{:.2e}".format(self.cur_cost))
            self.nominal_values = x_noms[ind], u_noms[ind]
            self.k = k_new[ind]
            self.du = self.du*self.alphas[ind]
            success = True
        if not success:
            print("Forward pass failed.")
            return 0
        else:
            return 1

    def iterate_once(self, cx=None, Cxx=None, max_iter=15):
        self.backward_pass(cx, Cxx)
        success = self.forward_pass(max_iter=max_iter)
        return success

    def rollout_open_loop(self, x_nom, u_nom):
        x = x_nom[0]
        x_log = []
        u_log = []
        for i in range(self.N):
            u =  u_nom[i]
            x_log += [x]
            u_log += [u]
            x = self._forward_model(x[None], u[None])[0]
        x_log = np.stack(x_log)
        u_log = np.stack(u_log)
        return x_log, u_log


    def initialize_replanning_procedure(self):
        DTQ = self.D.T @ self.Q
        self.replan_matrix = (np.eye(self.D.shape[-1]) - self.K @ self.D)@np.linalg.solve(DTQ @ self.D + self.R, DTQ)
        self.replan_matrix_u = (np.eye(self.D.shape[-1]) - self.K @ self.D)@np.linalg.solve(DTQ @ self.D + self.R, self.R.toarray())


    def replan_feedforward(self, xd=None, delta_xd=None):
        if delta_xd is None:
            delta_k = self.replan_matrix.dot(xd-self.xd)
        else:
            delta_k = self.replan_matrix.dot(delta_xd)
        # delta_k[-self.u_dim*2:] = 0.
        return self.k + delta_k



