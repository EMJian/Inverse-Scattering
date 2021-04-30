import os
import numpy as np
from scipy.special import jv as bessel1
from scipy.special import hankel1

import matplotlib.pyplot as plt

from config import Config


# noinspection PyAttributeOutsideInit
class MethodOfMomentModel:

    def __init__(self):

        # Physical parameters
        self.frequency = Config.frequency
        self.wavelength = 3e8 / self.frequency
        self.wave_number = 2*np.pi / self.wavelength
        self.impedance = 120*np.pi

        # Room parameters
        self.geometry = Config.geometry
        self.room_length = Config.room_length
        self.doi_size = Config.doi_size
        self.m = Config.forward_grid_number
        self.number_of_grids = self.m ** 2

        # Sensor parameters
        self.number_of_rx = Config.number_of_receivers
        self.number_of_tx = Config.number_of_transmitters
        self.sensor_positions = Config.sensor_positions
        self.transceiver = Config.transceiver

        # Scatterer parameters
        self.object_permittivity = Config.object_permittivity

        self.nan_remove = True
        self.noise_level = 0

        assert self.m * self.wavelength / (np.sqrt(self.object_permittivity) * self.doi_size) > 10

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

    def find_grids_with_object(self, grid_positions, grid_permittivities):
        """
        Returns centroids of grids containing scatterers
        List containing x,y indices of grids containing point scatterers
        """
        self.object_grid_centroids = []
        self.object_grid_locations = []
        for i in range(grid_permittivities.shape[0]):
            for j in range(grid_permittivities.shape[1]):
                if grid_permittivities[i, j] != 1:
                    x_coord = round(grid_positions[0][i, j], 4)
                    y_coord = round(grid_positions[1][i, j], 4)
                    self.object_grid_centroids.append((x_coord, y_coord))
                    self.object_grid_locations.append((i, j))
        unrolled_permittivities = grid_permittivities.reshape(grid_permittivities.size, order='F')
        self.object_grid_indices = np.nonzero(unrolled_permittivities != 1)

    def get_field_from_scattering(self, grid_permittivities):
        """
        Object field is a 2D array that captures the field on every point scatterer due to every other point scatterer
        """
        object_field = np.zeros((len(self.object_grid_centroids), len(self.object_grid_centroids)), dtype=complex)
        all_x = [grid[0] for grid in self.object_grid_centroids]
        all_y = [grid[1] for grid in self.object_grid_centroids]
        for i in range(len(self.object_grid_centroids)):
            # Centroid of grid containing point scatterer we are measuring the field on
            x_incident = self.object_grid_centroids[i][0]
            y_incident = self.object_grid_centroids[i][1]

            x_gridnum = self.object_grid_locations[i][0]
            y_gridnum = self.object_grid_locations[i][1]

            dipole_distances = np.sqrt((x_incident - all_x)**2 + (y_incident - all_y)**2)
            assert len(dipole_distances) == len(self.object_grid_centroids)

            z1 = -self.impedance * np.pi * (self.grid_radius/2) * bessel1(1, self.wave_number * self.grid_radius) * hankel1(0, self.wave_number*dipole_distances)
            z1[i] = -self.impedance*np.pi*(self.grid_radius/2)*hankel1(1,self.wave_number*self.grid_radius)-1j*self.impedance*grid_permittivities[x_gridnum,y_gridnum]/(self.wave_number*(grid_permittivities[x_gridnum,y_gridnum]-1))
            assert len(z1) == len(dipole_distances)
            object_field[i, :] = z1
        return object_field

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
        dist = np.sqrt((xtd - xrd)**2 + (ytd - yrd)**2)
        direct_field = (1j/4) * hankel1(0, self.wave_number * dist)
        return direct_field

    def get_incident_field(self, transmitter_positions, grid_positions):
        """
        Field from transmitter on every incident grid
        Output dimension - number of transmitters x number of grids
        """
        transmitter_x = [pos[0] for pos in transmitter_positions]
        transmitter_y = [pos[1] for pos in transmitter_positions]

        grid_x = grid_positions[0]
        grid_x = grid_x.reshape(grid_x.size, order='F')

        grid_y = grid_positions[1]
        grid_y = grid_y.reshape(grid_y.size, order='F')

        [xti, xsi] = np.meshgrid(transmitter_x, grid_x)
        [yti, ysi] = np.meshgrid(transmitter_y, grid_y)

        dist = np.sqrt((xti - xsi)**2 + (yti - ysi)**2)
        incident_field = (1j/4) * hankel1(0, self.wave_number * dist)
        return incident_field

    def get_induced_current(self, object_field, incident_field):

        incident_field_on_object = - incident_field[self.object_grid_indices]
        J1 = np.linalg.inv(object_field) @ incident_field_on_object

        current = np.zeros((self.m**2, self.number_of_tx), dtype=complex)
        for i in range(len(self.object_grid_indices[0])):
            current[self.object_grid_indices[0][i],:] = J1[i,:]

        return current

    def get_scattered_field(self, current, grid_positions, transmitter_positions):

        transmitter_x = [pos[0] for pos in transmitter_positions]
        transmitter_y = [pos[1] for pos in transmitter_positions]

        grid_x = grid_positions[0]
        grid_x = grid_x.reshape(grid_x.size, order='F')

        grid_y = grid_positions[1]
        grid_y = grid_y.reshape(grid_y.size, order='F')

        [xts, xss] = np.meshgrid(transmitter_x, grid_x)
        [yts, yss] = np.meshgrid(transmitter_y, grid_y)

        dist = np.sqrt((xts - xss)**2 + (yts - yss)**2)
        ZZ = - self.impedance * np.pi * (self.grid_radius/2) * bessel1(1, self.wave_number * self.grid_radius) \
             * hankel1(0, self.wave_number * np.transpose(dist))
        scattered_field = ZZ @ current

        return scattered_field

    def remove_nan_values(self, field):
        if self.nan_remove:
            np.fill_diagonal(field, np.nan)
            k = field.reshape(field.size, order='F')
            l = [x for x in k if not np.isnan(x)]
            m = np.reshape(l, (self.number_of_tx, self.number_of_rx-1))
            m = np.transpose(m)
            return m
        if not self.nan_remove:
            field[np.isnan(field)] = 0
            return field

    def transceiver_manipulation(self, direct_field, scattered_field, total_field):
        if self.transceiver:
            direct_field = self.remove_nan_values(direct_field)
            scattered_field = self.remove_nan_values(scattered_field)
            total_field = self.remove_nan_values(total_field)
        return direct_field, scattered_field, total_field

    def get_power_from_field(self, field):
        power = (np.abs(field)**2) * (self.wavelength**2) / (4*np.pi*self.impedance)
        power = 10 * np.log10(power / 1e-3)
        return power

    def get_field_plots(self, total_field, direct_field, scattered_field):
        plt.plot(range(self.number_of_rx - 1), np.abs(total_field[:, 19]), label="Total Field")
        plt.plot(range(self.number_of_rx - 1), np.abs(direct_field[:, 19]), label="Incident Field")
        plt.plot(range(self.number_of_rx - 1), np.abs(scattered_field[:, 19]), label="Scattered Field")
        plt.axis([0, 40, 0, 0.06])
        plt.legend()
        plt.show()

    def save_data(self, filename, incident_power, total_power, direct_field, total_field):
        script_dir = os.path.dirname(os.path.dirname(__file__))
        np.savez(os.path.join(script_dir, filename),
                 grid_centroids_x=self.centroids_x,
                 grid_centroids_y=self.centroids_y,
                 incident_power=incident_power,
                 total_power=total_power,
                 direct_field=direct_field,
                 total_field=total_field
                 )

    def load_data(self, filepath):
        data = np.load(filepath)
        return data.files

    def generate_forward_data(self, grid_permittivities, save=False, filename="forward_data", plot=False):
        receiver_positions = self.sensor_positions
        transmitter_positions = self.sensor_positions
        grid_positions = self.get_grid_positions()
        self.find_grids_with_object(grid_positions, grid_permittivities)
        object_field = self.get_field_from_scattering(grid_permittivities)
        direct_field = self.get_direct_field(transmitter_positions, receiver_positions)
        incident_field = self.get_incident_field(transmitter_positions, grid_positions)
        current = self.get_induced_current(object_field, incident_field)
        scattered_field = self.get_scattered_field(current, grid_positions, transmitter_positions)
        total_field = direct_field + scattered_field
        direct_field, scattered_field, total_field =\
            self.transceiver_manipulation(direct_field, scattered_field, total_field)
        incident_power = self.get_power_from_field(direct_field)
        total_power = self.get_power_from_field(total_field)

        if plot:
            self.get_field_plots(total_field, direct_field, scattered_field)

        if save:
            self.save_data(filename, incident_power, total_power, direct_field, total_field)
        return incident_power, total_power, direct_field, total_field


if __name__ == '__main__':

    def get_grid_permittivity(grid_positions):
        """
        Returns an MxM 2D array containing permittivity of each grid
        Need to be able to read this from images
        """
        m = Config.forward_grid_number
        h_side_y = 0.2
        epsilon_r = np.ones((m, m), dtype=float)
        epsilon_r[(grid_positions[0]-0.25)**2 + (grid_positions[1]-0.25)**2 <= h_side_y**2] = Config.object_permittivity
        return epsilon_r

    model = MethodOfMomentModel()
    grid_positions = model.get_grid_positions()
    grid_permittivity = get_grid_permittivity(grid_positions)
    incident_power, total_power, direct_field, total_field = model.generate_forward_data(grid_permittivity, save=False, plot=True)
    directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "field_data")
    filename = os.path.join(directory, "forward_data.npz")
    np.savez(filename,
             incident_power=incident_power,
             total_power=total_power,
             direct_field=direct_field,
             total_field=total_field,
             )
