
class Config:

    frequency = 2.4e9

    # Room details
    geometry = "square"
    room_length = 3
    doi_size = 0.5
    grid_number = 100

    # Sensor details
    number_of_sensors = 40
    transceiver = True
    number_of_transmitters = 40
    number_of_receivers = 40

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

    # scatterer details
    object_permittivity = 3
