import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from config import Config


class Regularizer:

    def __init__(self):
        self.m = Config.inverse_grid_number
        self.number_of_grids = self.m ** 2

    def least_squares(self, model, data):
        chi = np.linalg.inv(np.transpose(model) @ model) @ np.transpose(model) @ data
        chi = np.reshape(chi, (self.m, self.m), order='F')
        chi = 1 + np.real(chi)
        return chi

    def identity_shrinkage(self, model, data, shrinkage_intensity):
        target = np.eye(model.shape[1])
        chi = np.linalg.inv(shrinkage_intensity*np.transpose(model) @ model + (1 - shrinkage_intensity)*target) @ np.transpose(model) @ data
        chi = np.reshape(chi, (self.m, self.m), order='F')
        chi = 1 + np.real(chi)
        return chi

    def sv_shrinkage(self, model, data, shrinkage_intensity):
        target = np.diag(np.diag(np.transpose(model) @ model))
        chi = np.linalg.inv(shrinkage_intensity*np.transpose(model) @ model + (1 - shrinkage_intensity)*target) @ np.transpose(model) @ data
        chi = np.reshape(chi, (self.m, self.m), order='F')
        chi = 1 + np.real(chi)
        return chi

    def svmc_shrinkage(self, model, data, shrinkage_intensity):
        num_features = model.shape[1]
        scm = np.transpose(model) @ model
        mean_covariance = np.sum(scm * ~ np.eye(num_features, dtype=bool)) / (num_features ** 2 - num_features)
        target = np.diag(np.diag(scm)) + ~np.eye(num_features, dtype=bool) * mean_covariance
        chi = np.linalg.inv(shrinkage_intensity*scm + (1 - shrinkage_intensity)*target) @ np.transpose(model) @ data
        chi = np.reshape(chi, (self.m, self.m), order='F')
        chi = 1 + np.real(chi)
        return chi

    def lasso(self, model, data, alpha):
        # alpha=1e-6, positive=True
        ls = Lasso(alpha, positive=True)
        model = np.real(model)
        data = np.real(data)
        ls.fit(model, data)
        chi = ls.coef_
        chi = np.reshape(chi, (self.m, self.m), order='F')
        chi = 1 + np.real(chi)
        return chi

    def ridge(self, model, data, alpha):
        chi = np.linalg.inv((np.transpose(model) @ model) + alpha * np.eye(self.number_of_grids)) @ np.transpose(model) @ data
        chi = np.reshape(chi, (self.m, self.m), order='F')
        chi = 1 + np.real(chi)
        return chi

    def elastic_net(self, model, data):
        # alpha=1e-5, l1_ratio=0.7, positive=True
        ls = ElasticNet(alpha=1e-5, l1_ratio=0.7, positive=True)
        model = np.real(model)
        data = np.real(data)
        ls.fit(model, data)
        chi = ls.coef_
        chi = np.reshape(chi, (self.m, self.m), order='F')
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
