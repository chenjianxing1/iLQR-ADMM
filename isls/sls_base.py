import numpy as np
from .utils import find_precs, find_mus
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy

from .base import Base

class SLSBase(Base):
    def __init__(self, x_dim, u_dim, N):
        """
        Base class for SLS module.
        Sets the dynamics model and the cost function.
        """
        super().__init__(x_dim, u_dim, N)


        self.PHI_U = np.zeros((self.N * self.u_dim, self.N * self.x_dim))

        self.du = None
        self.dx = None
        self.PHI_X = None
        self.l_side_invs = None

    def compute_cost(self, x, u=None, cost_function=None):
        if cost_function is None:
            if len(x.shape) == 3: # batch solutions
                x = x.reshape(x.shape[0], -1)
            else:
                x = x.reshape(1, -1)
            dx = x - self.xd
            cost_ = np.sum(dx * (self.Q @ dx.T).T, axis=-1)
            if u is not None:
                if len(u.shape) == 3:
                    u = u.reshape(u.shape[0], -1)
                else:
                    u = u.reshape(1, -1)
                cost_ += np.sum(u * (self.R @ u.T).T, axis=-1)
            if cost_.shape[0] == 1:
                return cost_[0]
            else:
                return cost_
        else:
            return cost_function(x=x, u=u)




    def forward_model(self, x, u):
        if x.ndim >= 2:
            return x.dot(self.A.T) + u.dot(self.B.T)
        else:
            return self.A.dot(x) + self.B.dot(u)

    #################################### AT OPTIMALITY, RUN THESE ################################
    def u_optimal(self, x0, PHI_U, du):
        return (PHI_U[:,:self.x_dim] @ x0 + du).reshape(self.N, -1)[:-1]

    def x_optimal(self, x0, PHI_X, dx):
        return (PHI_X[:, :self.x_dim] @ x0 + dx).reshape(self.N, -1)

    def get_trajectory_batch(self, x0, us, noise_scale=0):
        batch_size = x0.shape[0] if x0.ndim == 2 else 0
        u_log = np.zeros((batch_size, self.N, self.u_dim))
        x_log = np.zeros((batch_size, self.N+1, self.x_dim))
        x_log[:, 0] = x0
        for i in range(self.N):
            u_log[:, i] = us[i]
            w = np.random.normal(loc=0, scale=noise_scale, size=x0.shape)
            x_log[:, i + 1] = self.forward_model(x_log[:, i], u_log[:, i]) + w
        if batch_size == 0:
            return x_log[0, :-1], u_log[0]
        else:
            return x_log[:, :-1], u_log

    def get_trajectory_dp(self, x0, K, k, noise_scale=0):
        batch_size = x0.shape[0] if x0.ndim == 2 else 1
        u_log = np.zeros((batch_size, self.N, self.u_dim))
        x_log = np.zeros((batch_size, self.N+1, self.x_dim))
        x_log[:, 0] = x0
        for i in range(self.N):
            u_log[:, i] = x_log[:, i].dot(K[i].T) + k[i]
            w = np.random.normal(loc=0, scale=noise_scale, size=x0.shape)
            x_log[:, i+1] = self.forward_model(x_log[:, i], u_log[:, i]) + w
        if batch_size == 0:
            print(x_log.shape, u_log.shape)
            return x_log[0, :-1], u_log[0]
        else:
            return x_log[:, :-1], u_log

    def get_trajectory_sls(self, x0, K, k, noise_scale=0):
        batch_size = x0.shape[0] if x0.ndim == 2 else 0
        u_log = np.zeros((batch_size, self.N, self.u_dim))
        x_log = np.zeros((batch_size, self.N+1, self.x_dim))
        x_log[:, 0] = x0
        x_vec = np.zeros((batch_size, self.N * self.x_dim))
        for i in range(self.N):
            x_vec[:, i * self.x_dim:(i + 1) * self.x_dim] = x_log[:, i]
            u_log[:, i] = (x_vec @ K.T + k)[:, i * self.u_dim:(i + 1) * self.u_dim]
            w = np.random.normal(loc=0, scale=noise_scale, size=x0.shape)
            x_log[:, i+1] = self.forward_model(x_log[:, i], u_log[:, i]) + w
        if batch_size == 0:
            return x_log[0, :-1], u_log[0]
        else:
            return x_log[:, :-1], u_log


    ##################################### Setters and Getters #####################################

    # @property
    # def Q(self):
    #     return self._Q
    #
    # @Q.setter
    # def Q(self, value):
    #     self._Q = value
    #
    # @property
    # def R(self):
    #     return self._R
    #
    # @R.setter
    # def R(self, value):
    #     self._R = value
    #
    # @property
    # def xd(self):
    #     return self._xd
    #
    # @xd.setter
    # def xd(self, value):
    #     self._xd = value
    #
    # @property
    # def ud(self):
    #     return self._ud
    #
    # @ud.setter
    # def ud(self, value):
    #     self._ud = value

    ################################ RESETTING THE SOLUTION #######################################################
    def reset(self):
        self.PHI_U = np.zeros((self.N * self.u_dim, self.N * self.x_dim))
        self.PHI_X = None
        self.du = None
        self.dx = None

    ############################################### PLOTTING ################################################
