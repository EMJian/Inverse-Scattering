import numpy as np


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
        if self.geometry == "square":
            """ If the geometry is a square, sensors are on the sides of the square room """
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
        grid_length = self.doi_size / self.m
        centroids_x = np.arange(start=grid_length/2, stop=self.doi_size, step=grid_length)
        centroids_y = np.arange(start=self.doi_size, stop=grid_length/2, step=-grid_length)
        return np.meshgrid(centroids_x, centroids_y)

    def get_grid_permittivities(self, grid_positions):
        h_side_x = 0.15
        h_side_y = 0.15
        epsilon_r = np.ones((self.m, self.m), dtype=float)
        epsilon_r[(grid_positions[0] - 0.25)**2 + (grid_positions[1] - 0.25)**2 <= h_side_y**2] = self.object_permittivity
        return epsilon_r


if __name__ == '__main__':

    model = MethodOfMomentModel()
    sensor_positions = model.get_sensor_positions()
    if model.transreceiver:
        receiver_positions = sensor_positions
        transmitter_positions = sensor_positions
    grid_positions = model.get_grid_positions()
    grid_permittivities = model.get_grid_permittivities(grid_positions)
