import numpy as np


def get_txrx_pairs():
    txrx_pairs = []
    if Config.transceiver:
        for i in range(Config.number_of_transmitters):
            for j in range(Config.number_of_receivers):
                if i != j:
                    txrx_pairs.append((i, j))
    else:
        for i in range(Config.number_of_transmitters):
            for j in range(Config.number_of_receivers):
                txrx_pairs.append((i, j))
    return txrx_pairs


class Config:

    frequency = 2.4e9

    # Room details
    geometry = "square"
    room_length = 3
    doi_size = 0.5
    forward_grid_number = 100
    inverse_grid_number = 50

    # Sensor details
    number_of_sensors = 40
    transceiver = True
    number_of_transmitters = 40
    number_of_receivers = 40

    # Tx-Rx pairs
    txrx_pairs = []
    if transceiver:
        for i in range(number_of_transmitters):
            for j in range(number_of_receivers):
                if i != j:
                    txrx_pairs.append((i, j))
    else:
        for i in range(number_of_transmitters):
            for j in range(number_of_receivers):
                txrx_pairs.append((i, j))

    # Sensor positions
    sensor_x = [-1.5, -1.2, -0.9, -0.6, -0.3, 0,
         0.3, 0.6, 0.9, 1.2, 1.5, 1.5,
         1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
         1.5, 1.5, 1.5, 1.2, 0.9, 0.6,
         0.3, 0, -0.3, -0.6, -0.9, -1.2,
         -1.5, -1.5, -1.5, -1.5, -1.5,
         -1.5, -1.5, -1.5, -1.5, -1.5]
    sensor_y = [-1.5, -1.5, -1.5, -1.5, -1.5,
         -1.5, -1.5, -1.5, -1.5, -1.5,
         -1.5, -1.2, -0.9, -0.6, -0.3, 0,
         0.3, 0.6, 0.9, 1.2, 1.5, 1.5,
         1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
         1.5, 1.5, 1.5, 1.2, 0.9, 0.6,
         0.3, 0, -0.3, -0.6, -0.9, -1.2]
    sensor_positions = np.transpose(np.array([sensor_x, sensor_y]))

    # scatterer details
    object_permittivity = 3
