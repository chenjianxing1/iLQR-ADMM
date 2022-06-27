from .admm import ADMM
from .projections import *
from .isls_base import iSLSBase
from .utils import run_once



class iSLS(iSLSBase):
    def __init__(self, x_dim, u_dim, N):
        """
        Supports solution methods below:

            1. batch: least-square solution to LQT and SLS problems. Supports costs correlating different time-steps.
                Computes an optimal solution starting from an initial state x0. DOES NOT compute a feedback controller.
            2. dp   :  dynamic programming solution to LQT problems. DOES NOT support costs correlating different
                time-steps. Computes a feedback controller in the form u_t = K_t @ x_t + k_t. Feedback is only on the
                current state x_t.
            3. sls  : system level synthesis solution to LQT and SLS problems. Supports costs correlating different
                time-steps. Computes a feedback controller in the form u_t = K{0:t} @ x{0:t} + k_t. Feedback is on the
                current state x_t and all the past states x{0:t-1}.

        Parameters
        ----------
        x_dim : int
                Dimension of the states
        u_dim : int
                Dimension of the control
        N     : int
                Number of time-steps

        Examples
        --------
         isls = iSLS(x_dim, u_dim, N)
            if nonlinear dynamics:
                f(x_{t}, u_{t}): returns x_{t+1} forward dynamics function
                isls.forward_model = f
                get_AB(x,u): returns A,B
            else:
                isls.AB = A,B

            if nonquadratic cost:
                isls.cost_function = cost
                get_Cs()
            else:
                isls.set_cost_variables()

            isls.solve_ilqr()
        """



        super().__init__(x_dim, u_dim, N)

    def solve(self, get_AB, get_Cs=None, is_dynamics_linear=False, is_cost_quadratic=False, method='dp',
                   max_iter=100, max_line_search_iter=25, tol_fun=1e-5, tol_grad=1e-4, verbose=False):
        """
        Solves an iLQR problem either with dynamic programming (dp=True) or in batch form with least-squares (dp=False)

        Parameters
        ----------
        get_AB : function, (optional)
                get_AB(x, u): returns A,B.
                x : array([N, x_dim]) , u: array([N, u_dim]), A: array([N, x_dim, x_dim]) and  B: array([N, x_dim, u_dim]).
                A is the derivative of the dynamics function with respect to the state and  B is the derivative of the
                dynamics function with respect to the control.


        get_Cs : function, (optional)

        max_iter: int, default = 200
                  Maximum number of iterations to solve the problem.

        max_line_search_iter: int, default = 50
                              Maximum number of line search iterations.

        method : {'batch', 'dp' or 'sls'}, default='sls'
                 Method to solve LQT-SLS problems.

        tol_fun : float
                  Threshold for the iLQR cost to stop the iterations

        tol_grad : float
                  Threshold for the iLQR cost to stop the iterations

        verbose : bool
                  If True, then prints iteration costs.

        Returns
        -------
        None
        """

        def set_dynamics_():
            if method == 'dp':
                self.A, self.B = get_AB(self.x_nom, self.u_nom)
            else:
                self.AB = get_AB(self.x_nom, self.u_nom) # self.AB computes Sx and Su as well.
            return 0
        set_dynamics = run_once(set_dynamics_) if is_dynamics_linear else set_dynamics_

        def set_costs_():
            cts, Cts = get_Cs(self.x_nom, self.u_nom)
            return cts, Cts
        set_costs = run_once(set_costs_) if is_cost_quadratic else set_costs_

        # Start iterations
        for i in range(max_iter):
            if verbose: print("Iteration", i)

            # Set current iteration parameters
            set_dynamics() # runs only once if is_dynamics_linear
            cts, Cts = set_costs() # runs only once if is_cost_quadratic

            # Solve
            if method == 'dp':
                fp_success, K, k = self.iterate_once_dp(max_line_search=max_line_search_iter, verbose=verbose, cts=cts,
                                                    Cts=Cts)
            elif method == 'batch':
                fp_success = self.iterate_once_batch(max_line_search=max_line_search_iter,
                                                     cts=cts, Cts=Cts, verbose=verbose)
            elif method == 'sls':
                raise NotImplementedError

            # Check convergence
            if np.abs(np.diff(self.cost_log[-2:])) < tol_fun:
                print("Cost change is too low, cannot improve anymore at iteration", i+1, ".")
                break
            if not fp_success:
                print("Forward pass failed, cannot improve anymore at iteration", i+1, ".")
                break
            if i == max_iter - 1:
                print("Maximum iterations reached.")

    ######################################### Batch iLQR #################################################
    def rollout_batch(self, x_nom, u_nom):
        """
        Dynamics rollout function for the line search step to compute costs in batch form.
        Parameters
        ----------
        x_nom : ndarray [N, x_dim]
        u_nom : ndarray [nb_batch, N, u_dim]

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

    def iterate_once_batch(self, verbose=False, max_line_search=15, **kwargs):
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
        delta_u_opt = self.backward_pass_batch(**kwargs)
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

            # Quu_inv = np.linalg.inv(Quu)
            # Kt = -Quu_inv.dot(Qux)
            # kt = -Quu_inv.dot(qu)

            sol = -scipy.linalg.solve(Quu, np.concatenate([Qux, qu[:,None]],axis=-1),check_finite=False, assume_a ="pos")
            Kt = sol[:,:-1]
            kt = sol[:, -1]

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
        # todo: check if feedforward gains go to zero  np.max(k) with tol_grad

        fp_success = False

        # Backward Pass
        K, k = self.backward_pass_DP(**kwargs)

        # Line Search
        k_new = k[None] * self.alphas[:max_line_search,None,None]
        x_noms, u_noms = self.rollout_DP(K, k_new)

        costs = self.cost_function(x_noms, u_noms)

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

    def ilqr_admm(self, get_AB, get_Cs=None, project_x=False, project_u=False,
                  max_iter=20, max_line_search_iter=20, max_admm_iter=20, rho_x=None, rho_u=None, alpha=1, tol=1e-3, verbose=False,
                  log=False):
        """
        Currently supports only the batch solution. For isls version, call isls_admm().
        Parameters
        ----------
        get_AB : function, (optional)
                get_AB(x, u): returns A,B.
                x : array([N, x_dim]) , u: array([N, u_dim]), A: array([N, x_dim, x_dim]) and  B: array([N, x_dim, u_dim]).
                A is the derivative of the dynamics function with respect to the state and  B is the derivative of the
                dynamics function with respect to the control.
        get_Cs : function, (optional)
        project_x : function, (optional) default = False
        project_u : function, (optional) default = False
        max_iter : int
        max_line_search_iter : int
        max_admm_iter : int
        rho_x : float or ndarray [x_dim, x_dim] or ndarray [N, x_dim, x_dim]
        rho_u : float or ndarray [u_dim, u_dim] or ndarray [N, u_dim, u_dim]
        alpha : 1.
                Relaxation parameter for ADMM
        tol : float
              Threshold for the iLQR cost to stop the iterations
        verbose : bool
        log : bool

        Returns
        -------

        """

        # TODO: add dp solution
        Qr, Rr = self.compute_Rr_Qr(rho_x=rho_x, rho_u=rho_u, dp=False)
        alphas = self.alphas[:max_line_search_iter]
        lmb_x_init = None
        lmb_u_init = None
        z_x_init = None
        z_u_init = None


        for j in range(max_iter):
            prev_cost = self.cost.copy()

            # Dynamics derivatives
            As, Bs = get_AB(self.x_nom, self.u_nom)
            self.AB = As, Bs
            Su = self.D

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
                xd = self.xd - self.x_nom.flatten()
                ud = - self.u_nom.flatten()

                SuTQ = Su.T @ self.Q
                l_side = SuTQ @ Su + self.R
                r_side = SuTQ @ xd + self.R @ ud

            if project_x:
                SuTQr = Su.T @ Qr
                l_side += SuTQr @ Su
            if project_u:
                l_side += Rr

            l_side_inv = np.array(np.linalg.inv(l_side))

            def f_argmin(x, u):
                add_r_side = 0.
                if project_x:
                    x_ = x - self.x_nom.flatten()
                    add_r_side += self.D.T @ Qr @ x_
                if project_u:
                    u_ = u - self.u_nom.flatten()
                    add_r_side += Rr @ u_

                delta_u_opt = (l_side_inv @ (r_side + add_r_side)).reshape(self.N, -1)

                # Line Search
                delta_u =  delta_u_opt[None] * alphas[:, None, None]
                x_noms, u_noms = self.rollout_batch(self.x_nom, self.u_nom + delta_u)
                costs = self.cost_function(x_noms, u_noms)
                if project_x:
                    dx = x_noms.reshape(-1, self.N*self.x_dim) - x
                    costs += np.sum(dx * dx @ Qr, axis=-1)
                if project_u:
                    du = u_noms.reshape(-1, self.N*self.u_dim) - u
                    costs += np.sum(du * du @ Rr, axis=-1)
                ind = np.argmin(costs)
                return x_noms[ind].flatten(), u_noms[ind].flatten()

            admm  = ADMM(self.x_dim * self.N, self.u_dim * self.N, f_argmin, project_x = project_x,
                                           project_u = project_u,
                                           z_x_init=z_x_init, z_u_init=z_u_init, lmb_x_init=lmb_x_init,
                                           lmb_u_init=lmb_u_init,Qr=Qr, Rr=Rr,
                                           return_lmb=1,
                                           alpha=alpha, max_iter=max_admm_iter, threshold=tol, verbose=verbose,
                                           log=log)

            self.nominal_values = admm[0].reshape(self.N, -1), admm[1].reshape(self.N, -1)
            z_x_init = admm[-3]
            z_u_init = admm[-2]

            print("Iteration number ", j, "iLQR cost: ", self.cost)
            if np.abs(self.cost - prev_cost) < 1e-3:
                print("Cost change is too low, cannot improve anymore at iteration", j, ".")
                break
            # Detect oscillations
            if np.abs(np.mean(self.cost_log[-4:]) - np.mean(self.cost_log[-8:-4])) < 1e-3:
                print("Cost is oscillating at iteration", j)
                break

        return admm[-1]

    def isls_admm(self, dim, get_AB, get_Cs=None, project_x=False, project_u=False, max_admm_iter=20,
                  k_max=20, max_line_search=20, rho_x=None, rho_u=None, alpha=1, threshold=1e-3, verbose=False,
                  log=False):
        """

        Parameters
        ----------
        get_AB
        get_Cs
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

        Qr, Rr = self.compute_Rr_Qr(rho_x=rho_x, rho_u=rho_u, dp=False)
        alphas = self.alphas[:max_line_search]

        reg_x = None
        reg_u = None
        dim_x = (self.x_dim * self.N,
                 dim + 1)  # shape[1] is self.x_dim+1 because we have robustness only wrt x0 position
        dim_u = (self.u_dim * self.N,
                 dim + 1)  # shape[1] is self.x_dim+1 because we have robustness only wrt x0 position

        z_x_init = np.zeros(dim_x)
        z_u_init = np.zeros(dim_u)

        for k in range(k_max):
            prev_cost = self.cost.copy()

            # Dynamics derivatives
            As, Bs = get_AB(self.x_nom, self.u_nom)
            self.AB = As, Bs
            Sx = self.C[:, :dim]
            Su = self.D

            # Cost function
            if get_Cs is not None:
                cts, Cts = get_Cs(self.x_nom, self.u_nom)
                Cxx = scipy.linalg.block_diag(*Cts[:,:self.x_dim, :self.x_dim])
                Cuu = scipy.linalg.block_diag(*Cts[:,self.x_dim:, self.x_dim:])
                cx = cts[:, :self.x_dim].flatten()
                cu = cts[:, self.x_dim:].flatten()

                SuTQ = Su.T @ (0.5*Cxx)
                l_side = SuTQ @ Su + (0.5*Cuu)
                r_side_fb = - SuTQ @ Sx
                r_side_ff = Su.T @ (-0.5*cx) + (-0.5*cu)

            else:
                xd = self.xd - self.x_nom.flatten()
                ud = - self.u_nom.flatten()

                SuTQ = Su.T @ self.Q
                l_side = SuTQ @ Su + self.R
                r_side_ff = SuTQ @ xd + self.R @ ud
                r_side_fb = - SuTQ @ Sx

            if project_x:
                SuTQr = Su.T @ Qr
                l_side += SuTQr @ Su
                r_side_fb += - SuTQr @ Sx
            if project_u:
                l_side += Rr

            l_side_inv = np.array(np.linalg.inv(l_side))
            r_side = np.concatenate([r_side_ff[:, None], r_side_fb], axis=-1)

            # Function for the first step of ADMM (unconstrained regularized iSLS)
            def f_argmin(x, u):
                # here shapes: x = [d_x, phi_x] and u = [d_u, phi_u]
                add_r_side = 0.
                if project_x:
                    add_r_side +=  SuTQr @ x
                if project_u:
                    add_r_side += Rr @ u

                du_ = l_side_inv @ (r_side + add_r_side) # [d_u, phi_u]
                dx_ = Su @ du_
                dx_[:, 1:] += Sx

                # Line search on d_u [ delta_u = phi_u @ delta_w + d_u, delta_w = 0]
                delta_u_opt = du_[:,0]
                delta_u = delta_u_opt.reshape(self.N, -1)[None] * alphas[:, None, None]
                x_noms, u_noms = self.rollout_batch(self.x_nom, self.u_nom + delta_u)
                costs = self.cost_function(x_noms, u_noms)
                # if project_x: todo
                #     dx = x_noms.reshape(-1, self.N*self.x_dim) - x
                #     costs += np.sum(dx * dx @ Qr, axis=-1)
                # if project_u:
                #     diff_du = delta_u.reshape(-1, self.N*self.u_dim) - u[:,0][None]
                #     costs += np.sum((diff_du**2)@Rr, axis=-1)

                    # diff_phiu = np.sum((du_[:,1:] - u[:,1:])**2, axis=-1)
                    # costs += np.sum(diff_phiu@Rr)

                ind = np.argmin(costs)

                # Optimal values for du = [d_u, phi_u] and  dx = [d_x, phi_x]
                du_opt = du_.copy()
                du_opt[:, 0] = delta_u[ind].flatten()

                dx_opt = dx_.copy()
                dx_opt[:, 0] = (x_noms[ind] - self.x_nom).flatten()

                return dx_opt, du_opt


            z_x = z_x_init
            z_u = z_u_init
            lmb_x = 0
            lmb_u = 0
            prim_res_norm = 1e6
            dual_res_norm = 1e6
            # Run ADMM for max_admm_iter
            for j in range(max_admm_iter):

                #### STEP 1 ####
                if project_x: reg_x = z_x - lmb_x
                if project_u: reg_u = z_u - lmb_u
                x_x, x_u = f_argmin(reg_x, reg_u)

                #### STEP 2 ####
                prev_prim_res_norm = np.copy(prim_res_norm)
                prev_dual_res_norm = np.copy(dual_res_norm)
                if project_x:
                    z_prev_x = z_x.copy()
                    z_x_ = alpha * x_x + (1 - alpha) * z_x
                    z_x = project_x(z_x_ + lmb_x, self.x_nom)

                    prim_res_x = x_x - z_x
                    lmb_x += prim_res_x

                if project_u:
                    z_prev_u = z_u.copy()
                    z_u_ = alpha * x_u + (1 - alpha) * z_u
                    z_u = project_u(z_u_ + lmb_u, self.u_nom)
                    prim_res_u = x_u - z_u
                    lmb_u += prim_res_u

                prim_res_norm = 0.
                dual_res_norm = 0.
                if project_x:
                    dual_res_norm += np.linalg.norm(Qr @ (z_x - z_prev_x))
                    prim_res_norm += np.linalg.norm(Qr @ prim_res_x)
                if project_u:
                    dual_res_norm += np.linalg.norm(Rr @ (z_u - z_prev_u))
                    prim_res_norm += + np.linalg.norm(Rr @ prim_res_u)

                if prim_res_norm < threshold and dual_res_norm < threshold:  # or np.abs(prim_res_norm-prim_res_norm_prev) < 1e-5:
                    if verbose:
                        print("ADMM converged at iteration ", j, "!")
                        print("ADMM residual is ", "{:.2e}".format(prim_res_norm), "{:.2e}".format(dual_res_norm))
                    break
                else:
                    prim_change = np.abs(prev_prim_res_norm - prim_res_norm) / (prev_prim_res_norm + 1e-30)
                    dual_change = np.abs(prev_dual_res_norm - dual_res_norm) / (prev_dual_res_norm + 1e-30)
                    if prim_change < 1e-3 and dual_change < 1e-3:
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

                if j == max_admm_iter - 1:
                    if verbose:
                        print("ADMM residuals-> primal:", "{:.2e}".format(prim_res_norm), "dual:",
                              "{:.2e}".format(dual_res_norm))
                        print("ADMM: Max iteration reached.")

            # Set new nominal values
            u_nom = self.u_nom + x_u[:, 0].reshape(self.N, -1)
            x_nom = self.x_nom + x_x[:, 0].reshape(self.N, -1)
            self.nominal_values = x_nom, u_nom


            z_x_init = z_x.copy()
            z_u_init = z_u.copy()

            # Check convergence
            print("Iteration number ", k, "iSLS cost: ", self.cost)
            if np.abs(self.cost - prev_cost) < 1e-4:
                print("Cost change is too low, cannot improve anymore at iteration", k, ".")
                break
            # Detect oscillations
            if np.abs(np.mean(self.cost_log[-4:]) - np.mean(self.cost_log[-8:-4])) < 1e-3:
                print("Cost is oscillating at iteration", k)
                break


        du = x_u[:,0]
        phi_u = x_u[:, 1:]
        return du, phi_u












