import os
import numpy as np
from scipy.special import jv as bessel1
from scipy.special import hankel1

import matplotlib.pyplot as plt

from config import Config
from inverse_models.approximations.models import Born, Rytov, PRytov
from inverse_models.approximations.regularize import Regularizer


class LinearApproximation:

    def __init__(self):

        # Physical parameters
        self.frequency = Config.frequency
        self.wavelength = 3e8 / self.frequency
        self.wave_number = 2 * np.pi / self.wavelength
        self.impedance = 120 * np.pi

        # Room parameters
        self.doi_size = Config.doi_size
        self.room_length = self.doi_size
        self.room_width = self.doi_size
        self.m = Config.inverse_grid_number
        self.number_of_grids = self.m ** 2

        # Sensor parameters
        self.sensor_position = Config.sensor_positions
        self.txrx_pairs = Config.txrx_pairs
        self.number_of_tx = Config.number_of_transmitters
        self.number_of_rx = Config.number_of_receivers

        self.nan_remove = Config.nan_remove

    def get_grid_positions(self):
        """
        Returns x and y coordinates for centroids of all grids
        Two m x m arrays, one for x coordinates of the grids, one for y coordinates
        """
        self.grid_length = self.doi_size / self.m
        self.grid_radius = np.sqrt(self.grid_length**2/np.pi)
        self.centroids_x = np.arange(start=self.grid_length/2, stop=self.doi_size, step=self.grid_length)
        self.centroids_y = np.arange(start=self.doi_size - self.grid_length/2, stop=0, step=-self.grid_length)
        return np.meshgrid(self.centroids_x, self.centroids_y)

    def get_direct_field(self, transmitter_positions, receiver_positions):
        """
        Field from transmitter to receiver
        Output dimension - number of transmitters x number of receivers
        """
        receiver_x = [pos[0] for pos in receiver_positions]
        receiver_y = [pos[1] for pos in receiver_positions]
        transmitter_x = [pos[0] for pos in transmitter_positions]
        transmitter_y = [pos[1] for pos in transmitter_positions]

        [xtd, xrd] = np.meshgrid(transmitter_x, receiver_x)
        [ytd, yrd] = np.meshgrid(transmitter_y, receiver_y)
        dist = np.sqrt((xtd - xrd) ** 2 + (ytd - yrd) ** 2)
        direct_field = (1j / 4) * hankel1(0, self.wave_number * dist)
        return direct_field

    def get_incident_field(self, transmitter_positions, grid_positions):
        """
        Field from transmitter on every incident grid
        Output dimension - number of transmitters x number of grids
        """
        transmitter_x = [pos[0] for pos in transmitter_positions]
        transmitter_y = [pos[1] for pos in transmitter_positions]

        grid_x = grid_positions[0] # 100 x 100
        grid_x = grid_x.reshape(grid_x.size, order='F') # 10000

        grid_y = grid_positions[1]
        grid_y = grid_y.reshape(grid_y.size, order='F')

        [xti, xsi] = np.meshgrid(transmitter_x, grid_x)
        [yti, ysi] = np.meshgrid(transmitter_y, grid_y)

        dist = np.sqrt((xti - xsi)**2 + (yti - ysi)**2)
        incident_field = (1j/4) * hankel1(0, self.wave_number * dist)
        return incident_field

    def get_greens_integral(self, transmitter_positions, grid_positions):
        transmitter_x = [pos[0] for pos in transmitter_positions]
        transmitter_y = [pos[1] for pos in transmitter_positions]

        grid_x = grid_positions[0] # 100 x 100
        grid_x = grid_x.reshape(grid_x.size, order='F') # 10000

        grid_y = grid_positions[1]
        grid_y = grid_y.reshape(grid_y.size, order='F')

        [xtg, xsg] = np.meshgrid(transmitter_x, grid_x)
        [ytg, ysg] = np.meshgrid(transmitter_y, grid_y)

        dist = np.sqrt((xtg - xsg)**2 + (ytg - ysg)**2)
        integral = (1j * np.pi * self.grid_radius / (2 * self.wave_number)) * \
            bessel1(1, self.wave_number * self.grid_radius) * hankel1(0, self.wave_number * np.transpose(dist))
        return integral

    def get_model_and_data(self, total_forward_field, total_forward_power, incident_forward_power, method="rytov"):
        transmitter_positions = self.sensor_position
        receiver_positions = self.sensor_position
        grid_positions = self.get_grid_positions()
        direct_field = self.get_direct_field(transmitter_positions, receiver_positions)
        incident_field = self.get_incident_field(transmitter_positions, grid_positions)
        integral_values = self.get_greens_integral(transmitter_positions, grid_positions)
        if method == "born":
            born = Born()
            model = born.get_model(incident_field, integral_values)
            data = born.get_data(direct_field, total_forward_field)
        if method == "rytov":
            rytov = Rytov()
            model = rytov.get_model(direct_field, incident_field, integral_values)
            data = rytov.get_data(direct_field, total_forward_field)
        if method == "prytov":
            prytov = PRytov()
            model = prytov.get_model(direct_field, incident_field, integral_values)
            data = prytov.get_data(total_forward_power, incident_forward_power)
        return model, data

    def get_reconstruction(self, method, total_forward_field, total_forward_power, incident_forward_power):
        model, data = self.get_model_and_data(total_forward_field, total_forward_power, incident_forward_power, method=method)
        regularizer = Regularizer()
        alpha = 1e5
        chi = regularizer.ridge(model, data, alpha)
        return chi


if __name__ == '__main__':

    model = LinearApproximation()

    # load data
    # filepath = r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\PROJECTS\ISP\data\scatterer_data\scatterers_mnist_5_2.npz"
    # scatterer_data = np.load(filepath)
    # scatterer_data = scatterer_data["scatterers"]
    filepath = r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\PROJECTS\ISP\data\field_data\forward_data.npz"
    field_data = np.load(filepath)
    total_forward_field = field_data["total_field"]
    total_forward_power = field_data["total_power"]
    incident_forward_power = field_data["incident_power"]

    chi = model.get_reconstruction("prytov", total_forward_field, total_forward_power, incident_forward_power)
    plt.imshow(chi, cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()
