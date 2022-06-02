import numpy as np
import scipy
from scipy.linalg import block_diag, null_space
import scipy.sparse as sp
from scipy.special import factorial

def nullspace_matrix(J):
    if type(J) != np.ndarray:
        J = J.toarray()
    return np.eye(J.shape[-1]) - scipy.linalg.lstsq(J, J)[0]
    # return np.eye(J.shape[-1]) - np.linalg.pinv(J)@J

def nullspace_matrix2(J):
    if type(J) != np.ndarray:
        J = J.toarray()
    N = null_space(J)
    return N @ N.T

def selection_matrix(m,n, horizon):
    for i in range(horizon+1):
        if i == 0:
            block_col = np.ones(((horizon+1)*m,n))
            G = block_col
        else:
            block_col = np.vstack([np.zeros((i*m,n)), np.ones(((horizon+1-i)*m,n))])
            G = np.hstack([G, block_col])
    return G

def construct_Z(d,N):
    Z = np.eye((d*(N+1)))*0
    i,j = np.indices(Z.shape)
    Z[i==d+j] = 1.# construction of Z is important!
    return Z

def find_mus(zs, seq):
    mus = []
    for i in range(len(seq)):
        mus += [zs[seq[i]]]
    return np.stack(mus).flatten()

def find_precs(Qs, seq, sqrt=False):
    precs = []
    precs_sqrt = []
    for i in range(len(seq)):
        Qt = Qs[seq[i]]
        precs += [Qt]
        if sqrt:
            if not np.sum(Qt != 0) > 0:
                precs_sqrt += [np.zeros_like(Qt)]
            else:
                precs_sqrt += [Qt ** 0.5]
    if sqrt:
        return block_diag(*precs), block_diag(*precs_sqrt)
    else:
        return sp.block_diag(precs).tocsr()

#### for augmented version ####
def augment_Qt(Q):
    n = Q.shape[0]
    Q_aug = np.eye(n + 1)
    Q_aug[:n, :n] = Q
    return Q_aug

def augment_mut(mu):
    n = mu.shape[0]
    M = np.eye(n + 1)
    M[n:, :-1] = -mu
    return M

def find_augmented_precs(zs, Qs, seq, sqrt=False):
    precs = []
    for i in range(seq.shape[0]):
        Qt = augment_Qt(Qs[seq[i]])
        M = augment_mut(zs[seq[i]])
        precs += [M @ Qt @ M.T]
    return block_diag(*precs)

##### For tasks in end-effector space ####
def batch_cost_vars(zs, Qs, seq):
    Q = find_precs(Qs, seq, sqrt=False)
    mu = find_mus(zs, seq)
    return mu, Q



# Julius' optimal basis functions
class TrajOpt():
    def __init__(self, ndof):
        self.ndof = ndof

    def setup_task(self, h):
        self.h = h
        self.N = len(h)
        self.nw_scalar = self.N + 1 + 2

        self.__M = np.zeros((self.N, 2, 2))
        for n in range(self.N):
            h_ = self.h[n]
            M_inv = np.array([[h_ ** 3 / 3, -h_ ** 2 / 2], [-h_ ** 2 / 2, h_]])
            self.__M[n] = np.linalg.inv(M_inv)

        self.__P = self.__get_P()

    def get_y(self, t, y_nodes, dy_0, dy_T):
        w = np.concatenate((y_nodes.flatten(), dy_0, dy_T))
        Phi = self.get_Phi(t)
        y = Phi @ w
        if np.size(t) == 1:
            return y.reshape(self.ndof)
        return y.reshape(np.size(t), self.ndof)

    def get_dy(self, t, y_nodes, dy_0, dy_T):
        w = np.concatenate((y_nodes.flatten(), dy_0, dy_T))
        dPhi = self.get_dPhi(t)
        dy = dPhi @ w
        if np.size(t) == 1:
            return dy.reshape(self.ndof)
        return dy.reshape(np.size(t), self.ndof)

    def get_ddy(self, t, y_nodes, dy_0, dy_T):
        w = np.concatenate((y_nodes.flatten(), dy_0, dy_T))
        ddPhi = self.get_ddPhi(t)
        ddy = ddPhi @ w
        if np.size(t) == 1:
            return ddy.reshape(self.ndof)
        return ddy.reshape(np.size(t), self.ndof)

    def get_Phi(self, t):
        return self.__get_base(t, 0)

    def get_dPhi(self, t):
        return self.__get_base(t, 1)

    def get_ddPhi(self, t):
        return self.__get_base(t, 2)

    def get_Omega(self, n):
        return self.__M[n] @ (self.__get_L_w(n) + self.__get_L_dq(n) @ self.__P)

    def __get_base(self, t, der):
        base = np.zeros((self.ndof * np.size(t), self.ndof * self.nw_scalar))
        if np.size(t) == 1:
            t_array = np.array([t])
        else:
            t_array = t

        for i in range(np.size(t)):
            t_ = t_array[i]
            t_start = 0.
            for n in range(self.N):
                if t_ <= t_start + self.h[n]:
                    t__ = t_ - t_start
                    c_q = np.zeros((1, self.nw_scalar))
                    c_q[0, n] = 1.
                    c_dq = np.zeros((1, self.N + 1))
                    c_dq[0, n] = 1.
                    Omega = self.get_Omega(n)
                    if der == 0:
                        base_ = c_q + t__ * c_dq @ self.__P + np.array([[-t__ ** 3 / 6, t__ ** 2 / 2]]) @ Omega
                    elif der == 1:
                        base_ = c_dq @ self.__P + np.array([[-t__ ** 2 / 2, t__]]) @ Omega
                    elif der == 2:
                        base_ = np.array([[-t__, 1.]]) @ Omega
                    else:
                        print('Invalid argument.')
                    base[i * self.ndof:(i + 1) * self.ndof] = np.kron(base_, np.eye(self.ndof))
                    break
                t_start += self.h[n]

        return base

    def __get_P(self):
        P_dq = np.zeros((self.N + 1, self.N + 1))
        P_w = np.zeros((self.N + 1, self.nw_scalar))
        for n in range(self.N - 1):
            a_n = np.array([[0., 1.]]) @ self.__M[n + 1]
            b_n = np.array([[-self.h[n], 1.]]) @ self.__M[n]
            P_dq[n] = b_n @ self.__get_L_dq(n) - a_n @ self.__get_L_dq(n + 1)
            P_w[n] = a_n @ self.__get_L_w(n + 1) - b_n @ self.__get_L_w(n)
        P_dq[self.N - 1, 0] = 1
        P_w[self.N - 1, self.N + 1] = 1
        P_dq[self.N, self.N] = 1
        P_w[self.N, self.N + 2] = 1

        return np.linalg.inv(P_dq) @ P_w

    def __get_L_w(self, n):
        if n < 0 or n >= self.N:
            print('Invalid argument.')
            return []
        L_w_n = np.zeros((2, self.nw_scalar))
        L_w_n[0, n] = -1
        L_w_n[0, n + 1] = 1
        return L_w_n

    def __get_L_dq(self, n):
        if n < 0 or n >= self.N:
            print('Invalid argument.')
            return []
        L_dq_n = np.zeros((2, self.N + 1))
        L_dq_n[0, n + 1] = -self.h[n]
        L_dq_n[1, n] = -1
        L_dq_n[1, n + 1] = 1
        return L_dq_n

def get_double_integrator_AB(nb_dim, nb_deriv=2, dt=0.01):
    A1d = np.zeros((nb_deriv, nb_deriv))

    for i in range(nb_deriv):
        A1d += np.diag(np.ones(nb_deriv - i), i) * np.power(dt, i) / factorial(i)

    B1d = np.zeros((nb_deriv, 1))
    for i in range(1, nb_deriv + 1):
        B1d[nb_deriv - i] = np.power(dt, i) / factorial(i)

    return np.kron(A1d, np.eye(nb_dim)), np.kron(B1d, np.eye(nb_dim))