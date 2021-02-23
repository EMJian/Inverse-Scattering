import numpy as np
from scipy.special import jv as bessel1
from scipy.special import hankel1


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

        # Method of Moment parameters
        self.doi_size = 0.5
        self.object_permittivity = 3
        self.number_of_rx = 40
        self.number_of_tx = self.number_of_rx
        self.m = 100
        self.number_of_grids = self.m ** 2
        assert self.m * self.wavelength / (np.sqrt(self.object_permittivity) * self.doi_size) > 10

    def get_sensor_positions(self):
        x_nodes = np.linspace(start=-self.room_length/2, stop=self.room_length/2, num=int(self.number_of_rx/4)+1)
        y_nodes = np.linspace(start=-self.room_length/2, stop=self.room_length/2, num=int(self.number_of_rx/4)+1)
        sensor_positions = []
        if self.geometry == "square" and self.transreceiver:
            """ 
            If the geometry is a square, sensors are on the sides of the square room 
            If they are transreceivers, both transmitters and receivers have the same coordinates
            """
            for i in x_nodes:
                for j in y_nodes:
                    if j == y_nodes[0] or j == y_nodes[-1]:
                        i = round(i, 1)
                        j = round(j, 1)
                        sensor_positions.append((i, j))
                    if i == x_nodes[0] or i == x_nodes[-1]:
                        i = round(i, 1)
                        j = round(j, 1)
                        sensor_positions.append((i, j))
        sensor_positions = list(set(sensor_positions))
        return sensor_positions

    def get_grid_positions(self):
        """
        Returns x and y coordinates for centroids of all grids
        """
        self.grid_length = self.doi_size / self.m
        self.grid_radius = np.sqrt(self.grid_length**2/np.pi)
        centroids_x = np.arange(start=self.grid_length/2, stop=self.doi_size, step=self.grid_length)
        centroids_y = np.arange(start=self.doi_size, stop=self.grid_length/2, step=-self.grid_length)
        return np.meshgrid(centroids_x, centroids_y)

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

    def field_due_to_scatterers(self, grid_permittivities):
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


if __name__ == '__main__':

    model = MethodOfMomentModel()
    sensor_positions = model.get_sensor_positions()
    if model.transreceiver:
        receiver_positions = sensor_positions
        transmitter_positions = sensor_positions
    grid_positions = model.get_grid_positions()
    grid_permittivities = model.get_grid_permittivities(grid_positions)
    model.find_grids_with_object(grid_positions, grid_permittivities)
    field = model.field_due_to_scatterers(grid_permittivities)

