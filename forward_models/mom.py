import os
import numpy as np
from scipy.special import jv as bessel1
from scipy.special import hankel1

import matplotlib.pyplot as plt

from config import Config


class MethodOfMomentModel:

    def __init__(self):

        # System parameters
        self.frequency = Config.frequency
        self.wavelength = 3e8 / self.frequency
        self.wave_number = 2*np.pi / self.wavelength
        self.impedance = 120*np.pi
        self.geometry = Config.geometry
        self.room_length = Config.room_length
        self.transreceiver = Config.transceiver
        self.nan_remove = True
        self.noise_level = 0
        self.doi_size = Config.doi_size
        self.object_permittivity = Config.object_permittivity
        self.number_of_rx = Config.number_of_receivers
        self.number_of_tx = Config.number_of_transmitters
        self.m = Config.grid_number
        self.number_of_grids = self.m ** 2
        assert self.m * self.wavelength / (np.sqrt(self.object_permittivity) * self.doi_size) > 10

    def get_sensor_positions(self):
        """
        Function returns the co-ordinates where sensors are placed, values come from config file
        1 x num_of_tx list containing tuples that specify sensor coordinates in the room
        """
        x = Config.sensor_x
        y = Config.sensor_y
        sensor_positions = np.transpose(np.array([x, y]))
        return sensor_positions

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
        Function returns centroids of grids containing scatterers
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
        # Object field is a 2D array that captures the field on every point scatterer due to every other point scatterer
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

        grid_x = grid_positions[0] # 100 x 100
        grid_x = grid_x.reshape(grid_x.size, order='F') # 10000

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
        # The inverse is correct

        current = np.zeros((self.m**2, self.number_of_tx), dtype=complex)
        for i in range(len(self.object_grid_indices[0])):
            current[self.object_grid_indices[0][i],:] = J1[i,:]

        return current

    def get_scattered_field(self, current, grid_positions, transmitter_positions):

        transmitter_x = [pos[0] for pos in transmitter_positions]
        transmitter_y = [pos[1] for pos in transmitter_positions]

        grid_x = grid_positions[0]  # 100 x 100
        grid_x = grid_x.reshape(grid_x.size, order='F')  # 10000

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

    def transreceiver_manipulation(self, direct_field, scattered_field, total_field):

        txrx_pairs = []

        if self.transreceiver:

            direct_field = self.remove_nan_values(direct_field)
            scattered_field = self.remove_nan_values(scattered_field)
            total_field = self.remove_nan_values(total_field)

            for i in range(self.number_of_tx):
                for j in range(self.number_of_rx):
                    if i != j:
                        txrx_pairs.append((i, j))

        else:

            for i in range(self.number_of_tx):
                for j in range(self.number_of_rx):
                    txrx_pairs.append((i, j))

        return direct_field, scattered_field, total_field, txrx_pairs

    def get_power_from_field(self, field):
        power = (np.abs(field)**2) * (self.wavelength**2) / (4*np.pi*self.impedance)
        power = 10 * np.log10(power / 1e-3)
        return power

    @staticmethod
    def get_field_plots(total_field, direct_field, scattered_field):
        plt.plot(range(model.number_of_rx - 1), np.abs(total_field[:, 19]), label="Total Field")
        plt.plot(range(model.number_of_rx - 1), np.abs(direct_field[:, 19]), label="Incident Field")
        plt.plot(range(model.number_of_rx - 1), np.abs(scattered_field[:, 19]), label="Scattered Field")
        plt.axis([0, 40, 0, 0.06])
        plt.legend()
        plt.show()

    def save_data(self, filename, txrx_pairs, incident_power, total_power, sensor_positions, direct_field, scattered_field, total_field):
        script_dir = os.path.dirname(os.path.dirname(__file__))
        np.savez(os.path.join(script_dir, filename),
                 txrx_pairs=txrx_pairs,
                 DOI_size=self.doi_size,
                 incident_power=incident_power,
                 total_power=total_power,
                 sensor_positions=sensor_positions,
                 grid_centroids_x=self.centroids_x,
                 grid_centroids_y=self.centroids_y,
                 direct_field=direct_field,
                 scattered_field=scattered_field,
                 total_field=total_field
                 )

    def load_data(self, filepath):
        data = np.load(filepath)
        print("Data: ", data.files)
        return data.files

    def generate_forward_data(self, grid_permittivities, save=True, filename="forward_data", plot=False):
        sensor_positions = self.get_sensor_positions()
        receiver_positions = sensor_positions
        transmitter_positions = sensor_positions
        grid_positions = self.get_grid_positions()
        self.find_grids_with_object(grid_positions, grid_permittivities)
        object_field = self.get_field_from_scattering(grid_permittivities)
        direct_field = self.get_direct_field(transmitter_positions, receiver_positions)
        incident_field = self.get_incident_field(transmitter_positions, grid_positions)
        current = model.get_induced_current(object_field, incident_field)
        scattered_field = model.get_scattered_field(current, grid_positions, transmitter_positions)
        total_field = direct_field + scattered_field
        direct_field, scattered_field, total_field, txrx_pairs = model.transreceiver_manipulation(direct_field,
                                                                                                  scattered_field,
                                                                                                  total_field)
        incident_power = self.get_power_from_field(direct_field)
        total_power = self.get_power_from_field(total_field)

        if plot:
            MethodOfMomentModel.get_field_plots(total_field, direct_field, scattered_field)

        if save:
            filename = "forward_data"
            model.save_data(filename, txrx_pairs, incident_power, total_power, sensor_positions, direct_field,
                            scattered_field, total_field)
        else:
            return txrx_pairs, incident_power, total_power, sensor_positions, direct_field, scattered_field, total_field


if __name__ == '__main__':

    def get_grid_permittivity(grid_positions):
        """
        Returns an MxM 2D array containing permittivity of each grid
        Need to be able to read this from images
        """
        m = Config.grid_number
        h_side_x = 0.15
        h_side_y = 0.15
        epsilon_r = np.ones((m, m), dtype=float)
        epsilon_r[(grid_positions[0]-0.25)**2 + (grid_positions[1]-0.25)**2 <= h_side_y**2] = Config.object_permittivity
        return epsilon_r

    model = MethodOfMomentModel()
    grid_positions = model.get_grid_positions()
    grid_permittivity = get_grid_permittivity(grid_positions)
    model.generate_forward_data(grid_permittivity, save=False, plot=True)
