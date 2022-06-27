import numpy as np
from .base import Base

class iSLSBase(Base):
    def __init__(self, x_dim, u_dim, N):
        super().__init__(x_dim, u_dim ,N)
        self._forward_model = None
        self._cost_function = None

        nb_max_iterations_line_search = 50
        self.alphas = 10.**np.linspace(0., -5., nb_max_iterations_line_search)

        self._cost = None
        self._cur_decrease_cost = 1e5
        self.x_nom = None
        self.u_nom = None

        self.x_nom_log = []
        self.u_nom_log = []

        self._K = None
        self._k = None

        self.cost_log = []

    #################################### AT OPTIMALITY, RUN THESE ################################

    def get_trajectory_sls(self, x0, K, k, noise_scale=0):
        batch_size = x0.shape[0] if x0.ndim == 2 else 1
        u_log = np.zeros((batch_size, self.N, self.u_dim))
        x_log = np.zeros((batch_size, self.N+1, self.x_dim))
        x_log[:, 0] = x0
        x_vec = np.zeros((batch_size, self.N * self.x_dim))
        for i in range(self.N):
            x_vec[:, i * self.x_dim:(i + 1) * self.x_dim] = x_log[:, i] - self.x_nom[i]
            u_log[:, i] = (x_vec @ K.T + k)[:, i * self.u_dim:(i + 1) * self.u_dim] + self.u_nom[i]
            w = np.random.normal(loc=0, scale=noise_scale, size=x0.shape)
            x_log[:, i+1] = self.forward_model(x_log[:, i], u_log[:, i]) + w
        if x0.ndim == 2:
            return x_log[0, :-1], u_log[0]
        else:
            return x_log[:, :-1], u_log

    def get_trajectory_batch(self, x0, us, noise_scale=0):
        batch_size = x0.shape[0] if x0.ndim == 2 else 1
        u_log = np.zeros((batch_size, self.N, self.u_dim))
        x_log = np.zeros((batch_size, self.N+1, self.x_dim))
        x_log[:, 0] = x0
        for i in range(self.N):
            u_log[:, i] = us[i]
            w = np.random.normal(loc=0, scale=noise_scale, size=x0.shape)
            x_log[:, i + 1] = self.forward_model(x_log[:, i], u_log[:, i]) + w

        if x0.ndim == 1:
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
        if x0.ndim == 2:
            return x_log[0, :-1], u_log[0]
        else:
            return x_log[:, :-1], u_log



    ############################## Setter and Getters ################################
    @property
    def nominal_values(self):
        return self._x_nom, self._u_nom

    @nominal_values.setter
    def nominal_values(self, value):
        self.x_nom = value[0]
        self.u_nom = value[1]
        self.cost = self.cost_function(self.x_nom, self.u_nom)
        self.cost_log += [self.cost]
        # self.x_nom_log.append(value[0])
        # self.u_nom_log.append(value[1])

    @property
    def K(self):
        return self._K

    @property
    def k(self):
        return self._k

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self._cost = value

    @property
    def forward_model(self):
        return self._forward_model

    @forward_model.setter
    def forward_model(self, function):
        self._forward_model = function

    @property
    def cost_function(self):
        if self._cost_function is None:
            self._cost_function = self.compute_cost
        return self._cost_function

    @cost_function.setter
    def cost_function(self, function):
        """

        Parameters
        ----------
        function :

        Returns
        -------

        """
        self._cost_function = function


    @property
    def AB(self):
        return [self.A, self.B]

    @AB.setter
    def AB(self, value):
        """
        Sets or updates the values of A, B, C and D matrices. We define $C = (I-ZA_d)^{-1}$ and $D = CZB_d$

        Parameters
        ----------
        value : a list of A and B [A,B]
                A is a 3D tensor of [N, x_dim, x_dim]
                B is a 3D tensor of [N, x_dim, u_dim]
        Returns
        -------
        None
        """
        self.A, self.B = value[0], value[1]
        for i in range(self.N - 1, 0, -1):
            self.D[i * self.x_dim:, (i - 1) * self.u_dim:i * self.u_dim] = \
                self.C[i * self.x_dim:, i * self.x_dim:(i + 1) * self.x_dim] @ self.B[i-1]

            self.C[i * self.x_dim:, (i - 1) * self.x_dim:i * self.x_dim] = \
                self.C[i * self.x_dim:, i * self.x_dim:(i + 1) * self.x_dim] @ self.A[i-1]

    def reset(self):
        self.PHI_U = np.zeros((self.N * self.u_dim, self.N * self.x_dim))
        self.PHI_X = None
        self.du = None
        self.dx = None

        self._cur_cost = None
        self._x_nom = None
        self._u_nom = None
        self.x_nom_log = []
        self.u_nom_log = []

        self._K = None
        self._k = None

        self.cost_log = []

        # self.cost_function





