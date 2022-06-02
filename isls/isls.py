from .sls import SLS
import scipy.sparse as sp
from .utils import find_precs, find_mus
from .admm import ADMM
from .projections import *

class iSLS(SLS):
    def __init__(self, x_dim, u_dim, N, quadratic_cost=False, linear_model=False):
        super().__init__(x_dim, u_dim, N)
        self._forward_model = None
        self._cost_function = None
        self._secondary_cost_function = None
        self.Q2 = None


        # self.Z = sp.csc_matrix(construct_Z(x_dim, self.N - 1))

        alpha = 1.
        self.alphas = np.zeros((50,))
        self.alphas[0] = 1.
        for i in range(1, 50):
            alpha = alpha * 0.6
            self.alphas[i] = alpha
        self._cur_cost = None
        self._cur_secondary_cost = None
        self._cur_decrease_cost = 1e5
        self._x_nom = None
        self._u_nom = None
        self.x_nom_log = []
        self.u_nom_log = []

        self._K = None
        self._k = None

        self.cost_log = []


    def rollout(self, K, k, x_nom, u_nom, verbose=False, real_system=False):
        x = x_nom[0]
        x_log = []
        u_log = []
        dx_vec = np.zeros(self.N*self.x_dim)
        for i in range(self.N):
            dx_vec[i * self.x_dim:(i + 1) * self.x_dim] = x - x_nom[i]
            u = K[i*self.u_dim:(i+1)*self.u_dim, :(i+1)*self.x_dim].dot(dx_vec[:(i+1)*self.x_dim])
            u += k[i * self.u_dim:(i + 1) * self.u_dim] + u_nom[i]
            # u = (K @ dx_vec + k)[i * self.u_dim:(i + 1) * self.u_dim] + u_nom[i]

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

    def backward_pass_CP(self, psi, PSI, cx=None, Cxx=None):
        if cx is None: # means quadratic cost
            xd = self.xd-self._x_nom.flatten()
            ud = -self._u_nom.flatten()
            DTQ = self.D.T @ self.Q
            l_side = DTQ @ self.D + self.R
            psi_l_side = psi.T @ l_side @ psi
            self.du = psi @ np.linalg.lstsq(psi_l_side, psi.T @ (DTQ @ xd + self.R @ ud), rcond=-1)[0]
            print("Feedforward computed")

            r_side = - DTQ @ self.C
            for i in range(self.N):
                AA = PSI[i].T @ l_side[i * self.u_dim:, i * self.u_dim:]
                BB = PSI[i].T @ r_side[i * self.u_dim:, i * self.x_dim:(i + 1) * self.x_dim]
                phi_u = PSI[i] @ np.linalg.lstsq(AA, BB, rcond=-1)[0]
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

    def batch_rollout_open_loop(self, x_nom, u_nom):
        """
        x_nom : [N, x_dim]
        u_nom : [nb_batch, N, u_dim]
        """
        nb_batch = u_nom.shape[0]
        x = np.tile(x_nom[0],(nb_batch, 1))
        x_log = np.zeros((nb_batch, self.N, self.x_dim))
        for i in range(self.N):
            x_log[:,i] = x
            x = self._forward_model(x, u_nom[:,i])
        return x_log, u_nom

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
        v = -2 * Q.dot(self.zs[self.seq[-1]]-self._x_nom[-1])
        for t in range(self.N - 2, -1, -1):
            Q = self.Qs[self.seq[t]]

            Ct[:self.x_dim, :self.x_dim] = 2*Q
            Ct[self.x_dim:, self.x_dim:] = 2*self.Rt

            ct[:self.x_dim] = -2*Q.dot(self.zs[self.seq[t]] - self._x_nom[t])
            ct[self.x_dim:] = -2*self.Rt.dot(-self._u_nom[t])

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
        x = np.tile(self._x_nom[0],(nb_batch, 1))
        x_log = np.zeros((nb_batch, self.N, self.x_dim))
        u_log = np.zeros((nb_batch, self.N, self.u_dim))
        for i in range(self.N):
            dx_vec = x - self._x_nom[i]
            u = dx_vec @ K[i].T
            u += k[:,i] + self._u_nom[i]

            u_log[:, i] = u
            x_log[:, i] = x
            x = self._forward_model(x, u)

        return x_log, u_log

    def iterate_once_DP_ilqr(self, max_iter=15):

        success = False
        # Backward Pass
        K,k = self.solve_DP()

        print("Backward pass finished.")
        # Line Search
        k_new = k[None] * self.alphas[:max_iter,None,None]
        x_noms, u_noms = self.batch_DP_rollout(K, k_new)

        costs = self.cost_function(x_noms, u_noms)
        ind = np.argmin(costs)
        # print(costs, self.cur_cost)
        if costs[ind] < self.cur_cost:
            print("We found a better solution with alpha ", self.alphas[ind], " with a new cost of ",
                  "{:.2e}".format(costs[ind]),
                  "which is smaller than the old cost of", "{:.2e}".format(self.cur_cost))
            self.nominal_values = x_noms[ind], u_noms[ind]
            success = True

        if not success:
            print("Forward pass failed.")
            return None
        else:
            return 1

    def iterate_once_batch_ilqr(self, max_iter=15):
        if type(self.Q) != sp.csc.csc_matrix:
            self.Q = sp.csc_matrix(self.Q)

        success = False
        # Backward Pass
        xd = self.xd - self._x_nom.flatten()
        ud = -self._u_nom.flatten()
        DTQ = self.D.T @ self.Q

        delta_u_opt = scipy.linalg.solve( DTQ @ self.D + self.R,
                                DTQ @ xd + self.R @ ud, assume_a='sym').reshape(self.N,-1)

        print("Backward pass finished.")

        # Line Search
        delta_u = delta_u_opt[None]*self.alphas[:max_iter,None,None]
        x_noms, u_noms = self.batch_rollout_open_loop(self._x_nom, self._u_nom+delta_u)
        costs = self.cost_function(x_noms, u_noms)

        ind = np.argmin(costs)
        # print(costs, self.cur_cost)
        if costs[ind] < self.cur_cost:
            print("We found a better solution with alpha ", self.alphas[ind], " with a new cost of ",
                  "{:.2e}".format(costs[ind]),
                  "which is smaller than the old cost of", "{:.2e}".format(self.cur_cost))
            self.nominal_values = x_noms[ind], self._u_nom + delta_u[ind]
            success = True

        if not success:
            print("Forward pass failed.")
            return None
        else:
            return 1


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


    ################################## ADMM #########################################
    def iterate_once_batch_ilqr_reg(self, rho_u, rho_x, x_reg, u_reg, max_iter=15):
        if type(self.Q) != sp.csc.csc_matrix:
            self.Q = sp.csc_matrix(self.Q)

        I = np.eye(self.R.shape[0])
        success = False
        # Backward Pass
        xd = self.xd - self._x_nom.flatten()
        ud = -self._u_nom.flatten()
        x_reg_ = x_reg - self._x_nom.flatten()
        u_reg_ = u_reg - self._u_nom.flatten()
        DTQ = self.D.T @ self.Q
        DTD = self.D.T @ rho_x @ self.D

        delta_u_opt = scipy.linalg.solve(DTQ @ self.D + self.R + I * rho_u + DTD,
                                         DTQ @ xd + self.R @ ud + rho_u * u_reg_ + self.D.T @ rho_x @ x_reg_,
                                         assume_a='sym').reshape(self.N, -1)

        # print("Backward pass finished.")

        # Line Search
        delta_u = delta_u_opt[None] * self.alphas[:max_iter, None, None]
        x_noms, u_noms = self.batch_rollout_open_loop(self._x_nom, self._u_nom + delta_u)
        costs = self.cost_function(x_noms, u_noms)

        ind = np.argmin(costs)
        # print(costs, self.cur_cost)
        if costs[ind] < self.cur_cost:
            # print("We found a better solution with alpha ", self.alphas[ind], " with a new cost of ",
            #       "{:.2e}".format(costs[ind]),
            #       "which is smaller than the old cost of", "{:.2e}".format(self.cur_cost))
            self.nominal_values = x_noms[ind], self._u_nom + delta_u[ind]
            success = True

        if not success:
            # print("Forward pass failed.")
            return None
        else:
            return 1

    def batch_ilqr_admm(self, get_A_B, list_of_proj_x=[], list_of_proj_u=[], z_init=None, max_admm_iter=20,
                        max_line_search=20, rho_x=0., rho_u=0., alpha=1, threshold=1e-3, verbose=False, log=False):

        if type(rho_x) == float:
            rho_x = np.eye(self.x_dim*self.N)*rho_x
        elif rho_x.shape[0] == self.x_dim:
            rho_x = np.kron(np.eye(self.N), rho_x)

        def f_argmin(x, u):
            As, Bs = get_A_B(self._x_nom, self._u_nom)

            self.AB = As, Bs
            success = self.iterate_once_batch_ilqr_reg(rho_u=rho_u, rho_x=rho_x, x_reg=x, u_reg=u, max_iter=max_line_search)
            return self._x_nom.flatten(), self._u_nom.flatten()

        _ = ADMMLQT(self.x_dim * self.N, self.u_dim * self.N, f_argmin, list_of_proj_x, list_of_proj_u,alpha=alpha,
                z_x_init=z_init, max_iter=max_admm_iter, threshold=threshold, verbose=verbose, log=log)
        if log:
            return _[-1]
        else:
            return 1

    # def iterate_once_batch_ilqr_reg2(self, rho_x, rho_u, sigma, Ax, Au, xr, ur, xk, uk,max_iter=15):
    #     if type(self.Q) != sp.csc.csc_matrix:
    #         self.Q = sp.csc_matrix(self.Q)
    #
    #     success = False
    #     # Backward Pass
    #     I = np.eye(self.R.shape[0])
    #     xd = self.xd - self._x_nom.flatten()
    #     ud = -self._u_nom.flatten()
    #     xr_ = xr - Ax @ self._x_nom.flatten()
    #     ur_ = ur - Au @ self._u_nom.flatten()
    #
    #     xk_ = xk - self._x_nom.flatten()
    #     uk_ = uk - self._u_nom.flatten()
    #
    #     DQ = self.D.T @ self.Q
    #     DTD = self.D.T @ self.D
    #     DTAxT = self.D.T @ Ax.T
    #     DTAxD = DTAxT @ Ax @ self.D
    #     AuTAu = Au.T@Au
    #     DQD = DQ @ self.D
    #     l_side =  DQD + self.R + rho_x*DTAxD + rho_u*AuTAu + sigma*I + sigma*DTD
    #     r_side = DQ @ xd + rho_x * DTAxT @ xr_ + rho_u * Au.T @ ur_ + self.R @ ud + sigma * (self.D.T @ xk_ + uk_)
    #
    #     delta_u_opt = scipy.linalg.solve(l_side, r_side,
    #                                      assume_a='sym').reshape(self.N, -1)
    #
    #     # Line Search
    #     delta_u = delta_u_opt[None] * self.alphas[:max_iter, None, None]
    #     x_noms, u_noms = self.batch_rollout_open_loop(self._x_nom, self._u_nom + delta_u)
    #     costs = self.cost_function(x_noms, u_noms)
    #
    #     ind = np.argmin(costs)
    #     # print(costs, self.cur_cost)
    #     if costs[ind] < self.cur_cost:
    #         # print("We found a better solution with alpha ", self.alphas[ind], " with a new cost of ",
    #         #       "{:.2e}".format(costs[ind]),
    #         #       "which is smaller than the old cost of", "{:.2e}".format(self.cur_cost))
    #         self.nominal_values = x_noms[ind], self._u_nom + delta_u[ind]
    #         success = True
    #
    #     if not success:
    #         # print("Forward pass failed.")
    #         return None
    #     else:
    #         return 1


    # def batch_ilqr_admm2(self, get_A_B,Ax = None, Au = None, list_of_proj_x=[], list_of_proj_u=[],
    #                      rho_x=0., rho_u=0., sigma=0.,
    #                      alpha=0.5, max_admm_iter=20, max_line_search=20, threshold=1e-3, verbose=False, log=False):
    #
    #     def f_argmin(xr, ur, xk, uk):
    #         As, Bs = get_A_B(self._x_nom, self._u_nom)
    #         self.AB = As, Bs
    #         success = self.iterate_once_batch_ilqr_reg2(rho_x=rho_x, rho_u=rho_u, sigma=sigma,xr=xr, ur=ur, Ax=Ax, Au=Au, xk=xk, uk=uk,
    #                                                     max_iter=max_line_search)
    #         return self._x_nom.flatten(), self._u_nom.flatten()
    #
    #     _ = ADMMLQT2(self.x_dim * self.N, self.u_dim * self.N, f_argmin, Ax=Ax, Au=Au,
    #                     list_of_proj_x=list_of_proj_x, list_of_proj_u=list_of_proj_u,alpha=alpha,
    #                    max_iter=max_admm_iter, threshold=threshold, verbose=verbose, log=log)
    #     if log:
    #         return _[-1]
    #     else:
    #         return 1
    ############################## Setter and Getters ################################
    @property
    def nominal_values(self):
        return self._x_nom, self._u_nom

    @nominal_values.setter
    def nominal_values(self, value):
        self._x_nom = value[0]
        self._u_nom = value[1]
        self.cur_cost = self.cost_function(self._x_nom.flatten(), self._u_nom.flatten())
        if self.Q2 is not None:
            self.cur_secondary_cost = self.secondary_cost_function(self._x_nom.flatten(), self._u_nom.flatten())
        self.x_nom_log.append(value[0])
        self.u_nom_log.append(value[1])
        # self.cost_log.append(deepcopy(self.cur_cost))

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value

    @property
    def cur_cost(self):
        return self._cur_cost

    @cur_cost.setter
    def cur_cost(self, value):
        self._cur_cost = value

    @property
    def cur_secondary_cost(self):
        return self._cur_secondary_cost

    @cur_secondary_cost.setter
    def cur_secondary_cost(self, value):
        self._cur_secondary_cost = value

    @property
    def forward_model(self):
        return self._forward_model

    @forward_model.setter
    def forward_model(self, function):
        self._forward_model = function

    @property
    def cost_function(self):
        if self._cost_function is None:
            self._cost_function = self.cost
        return self._cost_function

    @cost_function.setter
    def cost_function(self, function):
        self._cost_function = function

    @property
    def secondary_cost_function(self):
        if self._secondary_cost_function is None:
            self._secondary_cost_function = self.cost_secondary
        return self._secondary_cost_function

    @secondary_cost_function.setter
    def secondary_cost_function(self, function):
        self._secondary_cost_function = function

    @property
    def AB(self):
        """
        Returns: A list of A and B matrices
        """
        return [self.A, self.B]

    @AB.setter
    def AB(self, value):
        """
        Sets or updates the values of A, B, C and D matrices. We define $C = (I-ZA_d)^{-1}$ and $D = CZB_d$
        Args:
            value: a list of A and B [A,B]
        """
        self.A, self.B = value[0].astype(self.dtype), value[1].astype(self.dtype)
        for i in range(self.N - 1, 0, -1):
            self.D[i * self.x_dim:, (i - 1) *  self.u_dim:i *  self.u_dim] = \
                self.C[i *  self.x_dim:, i *  self.x_dim:(i + 1) *  self.x_dim].dot(self.B[i])

            self.C[i * self.x_dim:, (i - 1) * self.x_dim:i * self.x_dim] =\
                self.C[i * self.x_dim:, i * self.x_dim:(i + 1) * self.x_dim].dot(self.A[i])

    def reset(self):
        self.PHI_U = np.zeros((self.N * self.u_dim, self.N * self.x_dim))
        self.PHI_X = None
        self.du = None
        self.dx = None
        # self._forward_model = None
        self._cur_cost = None
        self._cur_secondary_cost = None
        self._x_nom = None
        self._u_nom = None
        self.x_nom_log = []
        self.u_nom_log = []

        self._K = None
        self._k = None

        self.cost_log = []

        self.psi_u = np.zeros(self.N * self.u_dim, )
        self.lmb_u = np.zeros(self.N * self.u_dim, )







