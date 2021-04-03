import numpy as np


from config import Config


class Regularizer:

    def __init__(self):
        self.m = Config.inverse_grid_number
        self.number_of_grids = self.m ** 2

    def lasso(self):
        pass

    def ridge(self, model, data, alpha):
        chi = np.linalg.inv((np.transpose(model) @ model) + alpha * np.eye(self.number_of_grids)) @ np.transpose(model) @ data
        chi = np.reshape(chi, (self.m, self.m))
        chi = 1 + np.real(chi)
        return chi

    def tv_1(self):
        pass

    def tv_2(self):
        pass

    def group_lasso(self):
        pass

    def fused_lasso(self):
        pass

    def elastic_net(self):
        pass
