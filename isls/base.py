import numpy as np
from .utils import find_precs, find_mus
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy

class Base(object):
    """
    Common base class for SLS and iSLS classes.
    """
    def __init__(self, x_dim, u_dim, N):
        self.N = N
        self.x_dim = x_dim
        self.u_dim = u_dim

        self.A = None
        self.B = None
        self.Sw = np.kron(np.triu(np.ones((N, N))).T, np.eye(x_dim))
        self.Su = np.zeros((x_dim*N,N*u_dim ))

        # if the cost is non-quadratic, these will stay None
        self.Q = None
        self.xd = None
        self.R = None
        self.ud = None


        bad_rows = np.arange(0, u_dim).tolist()
        bad_cols = np.arange(0, u_dim).tolist()

        # Below is the code to take the inverses in a fast way by 1-rank reduction techniques.
        def invert_k_rank_down(A, A_inv):
            good_rows = np.setdiff1d(np.arange(A.shape[0]), bad_rows)
            good_cols = np.setdiff1d(np.arange(A.shape[1]), bad_cols)
            U = np.zeros((A.shape[0], 2 * u_dim))
            U[bad_rows, :u_dim] = np.eye(u_dim)
            U[good_rows, u_dim:] = A[np.ix_(good_rows, bad_cols)]

            V = np.zeros((2 * u_dim, A.shape[1]))
            V[:u_dim, good_cols] = A[np.ix_(bad_rows, good_cols)]
            V[u_dim:, bad_cols] = np.eye(u_dim)
            return (A_inv + A_inv @ U @ np.linalg.inv(np.eye(V.shape[0]) - V @ A_inv @ U) @ V @ A_inv)[u_dim:, u_dim:]

        def compute_inverses(A):
            A_ = A.copy()
            A_inv_ = np.linalg.inv(A_)
            A_invs2 = [A_inv_]
            for i in range(self.N):
                A_invs2 += [invert_k_rank_down(A[i * u_dim:, i * u_dim:], A_invs2[i])]
            return A_invs2

        self.invert_k_rank_down = invert_k_rank_down
        self.compute_inverses = compute_inverses

    def compute_Rr_Qr(self, rho_x, rho_u, dp=True):
        if rho_x is None:
            Qr = None
        else:
            if type(rho_x) == float or type(rho_x) == int:
                Qr = np.tile(rho_x * np.eye(self.x_dim)[None], (self.N, 1, 1))
            elif rho_x.shape[0] == self.x_dim:
                Qr = np.zeros((self.N, self.x_dim, self.x_dim))
                Qr[:] = rho_x
            else:
                Qr = rho_x
            if not dp:
                Qr = scipy.linalg.block_diag(*Qr)
        if rho_u is None:
            Rr = None
        else:
            if type(rho_u) == float or type(rho_x) == int:
                Rr = [rho_u * np.eye(self.u_dim)]*self.N
            elif rho_u.shape[0] == self.u_dim:
                Rr = [rho_u] * self.N
            else:
                Rr = rho_u
            if not dp:
                Rr = scipy.linalg.block_diag(*Rr)
        return Qr, Rr

    def set_quadratic_cost(self, zs, Qs, seq, u_std):
        self.l_side_invs = None # resetting inverses
        self.zs = zs
        self.Qs = Qs
        self.seq = seq
        self.Rt = np.eye(self.u_dim) * u_std
        self.Q = find_precs(Qs, seq, sqrt=False)
        self.xd = find_mus(zs, seq)
        self.R = sp.csc_matrix(sp.block_diag([sp.eye(self.u_dim) * u_std] * self.N))

    @property
    def AB(self):
        """
        Returns: A list of A and B matrices
        """
        return [self.A, self.B]

    @AB.setter
    def AB(self, value):
        """
        Sets or updates the values of A, B, Sw and Su matrices. We define $Sw = (I-ZA_d)^{-1}$ and $Su = CZB_d$
        Parameters
        ----------
        value : a list of A and B [A,B]
                A is of shape [N, x_dim, x_dim] or [x_dim, x_dim]
                B is of shape [N, x_dim, u_dim] or [u_dim, u_dim]
        Returns
        -------
        None
        """
        self.A, self.B = value[0], value[1]
        for i in range(self.N - 1, 0, -1):
            A = self.A if self.A.ndim == 2 else self.A[i-1]
            B = self.B if self.B.ndim == 2 else self.B[i-1]
            self.Su[i * self.x_dim:, (i - 1) * self.u_dim:i * self.u_dim] = \
                self.Sw[i * self.x_dim:, i * self.x_dim:(i + 1) * self.x_dim] @ B

            self.Sw[i * self.x_dim:, (i - 1) * self.x_dim:i * self.x_dim] = \
                self.Sw[i * self.x_dim:, i * self.x_dim:(i + 1) * self.x_dim] @ A
