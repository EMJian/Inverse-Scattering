import numpy as np
from scipy.special import jv as bessel1
from scipy.special import hankel1
import copy


class MethodOfMomentModel:

    def __init__(self):

        # System parameters
        self.frequency = 2.4e9
        self.wavelength = 3e8 / self.frequency
        self.wave_number = 2*np.pi / self.wavelength
        self.impedance = 120*np.pi
        self.geometry = "square"
        self.room_length = 3
        self.transreceiver = True
        self.nan_remove = True
        self.noise_level = 0
        self.doi_size = 0.5
        self.object_permittivity = 3
        self.number_of_rx = 40
        self.number_of_tx = self.number_of_rx
        self.m = 100
        self.number_of_grids = self.m ** 2
        assert self.m * self.wavelength / (np.sqrt(self.object_permittivity) * self.doi_size) > 10

    def get_sensor_positions(self):
        """
        Function returns the co-ordinates where sensors are placed
        """
        X = [-1.5, -1.2, -0.9, -0.6, -0.3, 0,
         0.3, 0.6, 0.9, 1.2, 1.5, 1.5,
         1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
         1.5, 1.5, 1.5, 1.2, 0.9, 0.6,
         0.3, 0, -0.3, -0.6, -0.9, -1.2,
         -1.5, -1.5, -1.5, -1.5, -1.5,
         -1.5, -1.5, -1.5, -1.5, -1.5]

        Y = [-1.5, -1.5, -1.5, -1.5, -1.5,
         -1.5, -1.5, -1.5, -1.5, -1.5,
         -1.5, -1.2, -0.9, -0.6, -0.3, 0,
         0.3, 0.6, 0.9, 1.2, 1.5, 1.5,
         1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
         1.5, 1.5, 1.5, 1.2, 0.9, 0.6,
         0.3, 0, -0.3, -0.6, -0.9, -1.2]

        sensor_positions = np.transpose(np.array([X, Y]))
        return sensor_positions

    def get_grid_positions(self):
        """
        Returns x and y coordinates for centroids of all grids
        """
        self.grid_length = self.doi_size / self.m
        self.grid_radius = np.sqrt(self.grid_length**2/np.pi)
        self.centroids_x = np.arange(start=self.grid_length/2, stop=self.doi_size, step=self.grid_length)
        self.centroids_y = np.arange(start=self.doi_size - self.grid_length/2, stop=0, step=-self.grid_length)
        return np.meshgrid(self.centroids_x, self.centroids_y)

    def get_grid_permittivities(self, grid_positions):
        """
        Returns an MxM 2D array containing permittivity of each grid
        Need to be able to read this from images
        """
        h_side_x = 0.15
        h_side_y = 0.15
        epsilon_r = np.ones((self.m, self.m), dtype=float)
        epsilon_r[(grid_positions[0] - 0.25)**2 + (grid_positions[1] - 0.25)**2 <= h_side_y**2] = self.object_permittivity
        return epsilon_r

    def find_grids_with_object(self, grid_positions, grid_permittivities):
        """
        Function returns centroids of grids containing scatterers
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
        Field from transmitter on incident every grid
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
            print(i, self.object_grid_indices[0][i], J1[i,:][0])
            current[self.object_grid_indices[0][i],:] = J1[i,:]

        return current

    def get_scattered_field(self, current):

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
            k = field.reshape(field.size)
            l = [x for x in k if not np.isnan(x)]
            m = np.reshape(l, (self.number_of_rx - 1, self.number_of_tx))
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

    def save_data(self, txrx_pairs, incident_power, total_power, sensor_positions, direct_field, scattered_field, total_field):
        np.savez("Forward data",
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


if __name__ == '__main__':

    model = MethodOfMomentModel()

    # 1 x 40 list containing tuples that specify sensor coordinates in the room
    sensor_positions = model.get_sensor_positions()
    receiver_positions = sensor_positions
    transmitter_positions = sensor_positions

    # Two 100 x 100 arrays, one for x coordinates of the grids, one for y coordinates
    grid_positions = model.get_grid_positions()

    # One 100 x 100 array, value specifying permittivity for each grid
    grid_permittivities = model.get_grid_permittivities(grid_positions)

    # List containing x,y indices of grids containing point scatterers - dim (1 x 2820)
    model.find_grids_with_object(grid_positions, grid_permittivities)

    # Scattered field on every scatterer due to every other - dim (2820 x 2820)
    object_field = model.get_field_from_scattering(grid_permittivities)

    # Field from transmitter to receiver - dim (40 x 40)
    direct_field = model.get_direct_field(transmitter_positions, receiver_positions)

    # Field from transmitter to every other grid - dim (40 * 10000)
    incident_field = model.get_incident_field(transmitter_positions, grid_positions)

    current = model.get_induced_current(object_field, incident_field)

    scattered_field = model.get_scattered_field(current)

    total_field = direct_field + scattered_field

    total_field_copy = copy.deepcopy(total_field)

    direct_field, scattered_field, total_field, txrx_pairs = model.transreceiver_manipulation(direct_field, scattered_field, total_field)

    incident_power = model.get_power_from_field(direct_field)

    total_power = model.get_power_from_field(total_field)

    # ----------------------------------------------------------------------
    # All field values are now calculated
    # ----------------------------------------------------------------------

    model.save_data(txrx_pairs, incident_power, total_power, sensor_positions, direct_field, scattered_field, total_field)
