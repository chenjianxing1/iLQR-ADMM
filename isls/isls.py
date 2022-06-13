from .admm import ADMM
from .projections import *
from .isls_base import iSLSBase

class iSLS(iSLSBase):
    def __init__(self, x_dim, u_dim, N):
        """

        Parameters
        ----------
        x_dim
        u_dim
        N
        """
        super().__init__(x_dim, u_dim, N)

    def solve_ilqr(self, get_AB, get_Cs=None, max_ilqr_iter=500, max_line_search_iter=50, dp=True,
                   tol_fun=1e-5, tol_grad=1e-4, verbose=False):
        """
        Solves an iLQR problem either with dynamic programming (dp=True) or in batch form with least-squares (dp=False)

        Parameters
        ----------
        get_AB
        get_Cs
        max_ilqr_iter
        max_line_search_iter
        dp
        tol_fun
        tol_grad
        verbose

        Returns
        -------

        """

        cts = None
        Cts = None

        for i in range(max_ilqr_iter):
            if verbose: print("Iteration", i)

            As, Bs = get_AB(self.x_nom, self.u_nom)
            if np.isnan(np.sum(As) + np.sum(Bs)):
                print("Nan value in the derivatives of the dynamics.")
                break

            if get_Cs:
                dp = True
                cts, Cts = get_Cs(self.x_nom, self.u_nom)
                if np.isnan(np.sum(cts) + np.sum(Cts)):
                    print("Nan value in the derivatives of the cost.")
                    break

            if dp:
                self.A = As
                self.B = Bs
                fp_success, K, k = self.iterate_once_dp(max_line_search=max_line_search_iter, verbose=verbose, cts=cts,
                                                    Cts=Cts)
                # todo: check if feedforwqrd gains go to zero  np.max(k) with tol_grad

            else:
                self.AB = As, Bs
                # self.AB computes Sx and Su as well.
                fp_success = self.iterate_once_batch(max_line_search=max_line_search_iter, verbose=verbose)

            if np.isnan(np.sum(self.x_nom) + np.sum(self.u_nom)):
                print("Nan value in the nominal solution.")
                break
            if np.isnan(self.cost):
                print("Nan value in the cost.")
                break

            if np.abs(np.diff(self.cost_log[-2:])) < tol_fun:
                print("Cost change is too low, cannot improve anymore at iteration", i+1, ".")
                break
            if not fp_success:
                print("Forward pass failed, cannot improve anymore at iteration", i+1, ".")
                break
            if i == max_ilqr_iter - 1:
                print("Maximum iterations reached.")

    ######################################### Batch iLQR #################################################
    def rollout_batch(self, x_nom, u_nom):
        """

        Parameters
        ----------
        x_nom : [N, x_dim]
        u_nom : [nb_batch, N, u_dim]

        Returns
        -------

        """
        nb_batch = u_nom.shape[0]
        x0 = x_nom[0]
        x = np.tile(x0,(nb_batch, 1))
        x_log = np.zeros((nb_batch, self.N, self.x_dim))
        for i in range(self.N):
            x_log[:,i] = x
            x = self._forward_model(x, u_nom[:,i])
        return x_log, u_nom

    def backward_pass_batch(self, Cts=None, cts=None):
        """

        Parameters
        ----------
        Cts:
        cts

        Returns
        -------

        """

        if Cts is not None:
            Cxx = scipy.linalg.block_diag(*Cts[:, :self.x_dim, :self.x_dim])
            Cuu = scipy.linalg.block_diag(*Cts[:, self.x_dim:, self.x_dim:])
            cx = cts[:, :self.x_dim].flatten()
            cu = cts[:, self.x_dim:].flatten()

            DTQ = self.D.T @ (0.5 * Cxx)
            l_side = DTQ @ self.D + (0.5 * Cuu)
            r_side = self.D.T @ (-0.5 * cx) + (-0.5 * cu)

        else:
            x_nom_ = self.x_nom.flatten()
            u_nom_ = self.u_nom.flatten()
            xd = self.xd - x_nom_
            ud = - u_nom_

            DTQ = self.D.T @ self.Q
            l_side = DTQ @ self.D + self.R
            r_side = DTQ @ xd + self.R @ ud

        delta_u_opt = scipy.linalg.solve(l_side, r_side, assume_a='sym').reshape(self.N, -1)
        return delta_u_opt

    def iterate_once_batch(self, verbose=False, max_line_search=15):
        """

        Parameters
        ----------
        verbose
        max_line_search

        Returns
        -------

        """
        fp_success = False

        # Backward pass
        delta_u_opt = self.backward_pass_batch()
        if verbose:
            print("Backward pass finished.")

        # Line Search
        delta_u = delta_u_opt[None] * self.alphas[:max_line_search, None, None]
        x_noms, u_noms = self.rollout_batch(self.x_nom, self.u_nom + delta_u)
        costs = self.cost_function(x_noms, u_noms)
        ind = np.argmin(costs)

        if costs[ind] < self.cost:
            if verbose:
                print("We found a better solution with alpha ", self.alphas[ind], " with a new cost of ",
                      "{:.2e}".format(costs[ind]),
                      "which is smaller than the old cost of", "{:.2e}".format(self.cost))
            self.nominal_values = x_noms[ind], u_noms[ind]
            fp_success = True

        return fp_success

    ######################################### DP-iLQR #################################################

    def backward_pass_DP(self, Cts=None, cts=None):
        """
        Assuming that there is no cost multiplying x and u -> Cux = 0
        zt = [xt, ut]
        x{t+1} = Ft @ zt + ft
        cost_t = 0.5* zt.T @ Ct @ zt + zt.T @ ct

        Parameters
        ----------
        Cts
        cts

        Returns
        -------

        """
        K = np.zeros((self.N, self.u_dim, self.x_dim))
        k = np.zeros((self.N, self.u_dim))

        # Final timestep computations
        if Cts is None:
            Q = self.Qs[self.seq[-1]]
            V = 2 * Q
            v = -2 * Q.dot(self.zs[self.seq[-1]] - self.x_nom[-1])

        else:
            # Cost is not quadratic, need to compute derivatives
            assert cts is not None
            V = Cts[-1][:self.x_dim,:self.x_dim]
            v = cts[-1][:self.x_dim]

        # dV = np.zeros(2)
        for t in range(self.N - 2, -1, -1):

            if Cts is None:
                Q = self.Qs[self.seq[t]]

                Cxx = 2 * Q
                Cuu = 2 * self.Rt
                Cux = 0.

                cx = -2 * Q.dot(self.zs[self.seq[t]] - self.x_nom[t])
                cu = -2 * self.Rt.dot(-self.u_nom[t])

            else:
                Cxx = Cts[t][:self.x_dim, :self.x_dim]
                Cuu = Cts[t][self.x_dim:, self.x_dim:]
                Cux = Cts[t][self.x_dim:, :self.x_dim]

                cx = cts[t][:self.x_dim]
                cu = cts[t][self.x_dim:]


            A = self.A[t]
            B = self.B[t]

            qx = cx + A.T.dot(v)
            qu = cu + B.T.dot(v)

            Qxx = Cxx + A.T.dot(V).dot(A)
            Qux = Cux + B.T.dot(V).dot(A)
            Quu = Cuu + B.T.dot(V).dot(B)

            Quu_inv = np.linalg.inv(Quu)
            Kt = -Quu_inv.dot(Qux)
            kt = -Quu_inv.dot(qu)

            V = Qxx + Kt.T.dot(Quu).dot(Kt) + Qux.T.dot(Kt) + Kt.T.dot(Qux)
            # V = 0.5*(V.T + V)
            v = qx + Kt.T.dot(qu) + Kt.T.dot(Quu).dot(kt) + Qux.T.dot(kt)
            # dV = dV + np.array([kt.T.dot(qu), 0.5*kt.T.dot(Quu.dot(kt)) ])

            K[t] = Kt
            k[t] = kt

        return K, k

    def rollout_DP(self, K, k):
        """

        Parameters
        ----------
        K
        k

        Returns
        -------

        """

        nb_batch = k.shape[0]
        x = np.tile(self.x_nom[0],(nb_batch, 1))
        x_log = np.zeros((nb_batch, self.N, self.x_dim))
        u_log = np.zeros((nb_batch, self.N, self.u_dim))
        for i in range(self.N):
            dx_vec = x - self.x_nom[i]
            u = dx_vec @ K[i].T + k[:,i] + self.u_nom[i]
            u_log[:, i] = u.copy()
            x_log[:, i] = x.copy()
            x = self.forward_model(x, u)

        return x_log, u_log

    def iterate_once_dp(self, max_line_search=15, verbose=False, **kwargs):
        """

        Parameters
        ----------
        max_line_search
        verbose
        kwargs

        Returns
        -------

        """
        fp_success = False

        # Backward Pass
        K, k = self.backward_pass_DP(**kwargs)

        # Line Search
        k_new = k[None] * self.alphas[:max_line_search,None,None]
        x_noms, u_noms = self.rollout_DP(K, k_new)

        costs = self.cost_function(x_noms, u_noms)
        if np.isnan(np.sum(costs)):
            costs[np.isnan(costs)] = 1e5

        ind = np.argmin(costs)
        dcost = costs[ind] - self.cost

        if dcost < 0.:
            self.nominal_values = x_noms[ind], u_noms[ind]
            fp_success = True
        else:
            if verbose:
                print("Forward pass failed with a cost of", costs[ind])

        return fp_success, K, k


    ################################## ADMM #########################################

    def ilqr_admm(self, get_AB, get_Cs=None, list_of_proj_x=None, list_of_proj_u=None, max_admm_iter=20,
                  k_max=20, max_line_search=20, rho_x=None, rho_u=None, alpha=1, threshold=1e-3, verbose=False,
                  log=False):
        """

        Parameters
        ----------
        get_AB
        get_Cs
        list_of_proj_x
        list_of_proj_u
        max_admm_iter
        k_max
        max_line_search
        rho_x
        rho_u
        alpha
        threshold
        verbose
        log

        Returns
        -------

        """

        # TODO: add dp solution
        if list_of_proj_u is None:
            list_of_proj_u = []
        if list_of_proj_x is None:
            list_of_proj_x = []
        Qr, Rr = self.compute_Rr_Qr(rho_x=rho_x, rho_u=rho_u, dp=False)
        alphas = self.alphas[:max_line_search]
        lmb_x_init = None
        lmb_u_init = None
        z_x_init = None
        z_u_init = None


        for j in range(k_max):
            prev_cost = self.cost.copy()
            # z_x_init = None
            # z_u_init = self.u_nom.flatten()

            # Dynamics derivatives
            As, Bs = get_AB(self.x_nom, self.u_nom)
            self.AB = As, Bs

            # Cost function
            if get_Cs is not None:
                cts, Cts = get_Cs(self.x_nom, self.u_nom)
                Cxx = scipy.linalg.block_diag(*Cts[:,:self.x_dim, :self.x_dim])
                Cuu = scipy.linalg.block_diag(*Cts[:,self.x_dim:, self.x_dim:])
                cx = cts[:, :self.x_dim].flatten()
                cu = cts[:, self.x_dim:].flatten()

                DTQ = self.D.T @ (0.5*Cxx)
                l_side = DTQ @ self.D + (0.5*Cuu)
                r_side = self.D.T @ (-0.5*cx) + (-0.5*cu)

            else:
                x_nom_ = self.x_nom.flatten()
                u_nom_ = self.u_nom.flatten()
                xd = self.xd - x_nom_
                ud = - u_nom_

                DTQ = self.D.T @ self.Q
                l_side = DTQ @ self.D + self.R
                r_side = DTQ @ xd + self.R @ ud

            if Rr is not None:
                l_side += Rr
            if Qr is not None:
                l_side += self.D.T @ Qr @ self.D

            l_side_inv = np.array(np.linalg.inv(l_side))


            def f_argmin(x, u):
                u_ = u - self.u_nom.flatten()
                x_ = x - self.x_nom.flatten()
                add_r_side = 0.
                if Qr is not None:
                    add_r_side += self.D.T @ Qr @ x_
                if Rr is not None:
                    add_r_side += Rr @ u_

                delta_u_opt = (l_side_inv @ (r_side + add_r_side)).reshape(self.N, -1)

                # Line Search
                delta_u =  delta_u_opt[None] * alphas[:, None, None]
                x_noms, u_noms = self.rollout_batch(self.x_nom, self.u_nom + delta_u)
                costs = self.cost_function(x_noms, u_noms)
                ind = np.argmin(costs)
                return x_noms[ind].flatten(), u_noms[ind].flatten()

            admm  = ADMM(self.x_dim * self.N, self.u_dim * self.N, f_argmin, list_of_proj_x,
                                           list_of_proj_u,
                                           z_x_init=z_x_init, z_u_init=z_u_init, lmb_x_init=lmb_x_init,
                                           lmb_u_init=lmb_u_init,
                                           return_lmb=1,
                                           alpha=alpha, max_iter=max_admm_iter, threshold=threshold, verbose=verbose,
                                           log=log)

            self.nominal_values = admm[0].reshape(self.N, -1), admm[1].reshape(self.N, -1)
            # lmb_x_init = admm[-5]
            # lmb_u_init = admm[-4]
            z_x_init = admm[-3]
            z_u_init = admm[-2]

            print("Iteration number ", j, "iLQR cost: ", self.cost)
            if np.abs(self.cost - prev_cost) < 1e-5:
                print("Cost change is too low, cannot improve anymore at iteration", j + 1, ".")
                break

        return admm[-1]







