import os
import copy
import numpy as np

from forward_models.mom import MethodOfMomentModel

if __name__ == '__main__':

    script_dir = os.path.dirname(__file__)
    filepath = r"C:/Users/dsamr/OneDrive - HKUST Connect/MPHIL RESEARCH/PROJECTS/ISP/data/scatterer_data/mnist_scatterers_8.npz"
    data = np.load(filepath)
    scatterers = data["mnist_scatterers_8"]

    total_fields = []

    for i, scatterer in enumerate(scatterers):

        print(i)

        model = MethodOfMomentModel()
        sensor_positions = model.get_sensor_positions()
        receiver_positions = sensor_positions
        transmitter_positions = sensor_positions
        grid_positions = model.get_grid_positions()

        model.find_grids_with_object(grid_positions, scatterer)
        object_field = model.get_field_from_scattering(scatterer)
        direct_field = model.get_direct_field(transmitter_positions, receiver_positions)

        incident_field = model.get_incident_field(transmitter_positions, grid_positions)
        current = model.get_induced_current(object_field, incident_field)
        scattered_field = model.get_scattered_field(current, grid_positions, transmitter_positions)

        total_field = direct_field + scattered_field

        total_field_copy = copy.deepcopy(total_field)
        direct_field, scattered_field, total_field, txrx_pairs = model.transreceiver_manipulation(direct_field, scattered_field, total_field)

        total_fields.append(total_field)

    script_dir = os.path.dirname(__file__)
    filename = "mnist_total_field_8s"
    np.savez(os.path.join(script_dir, "field_data", filename), total_field_8s=total_fields)
    print("Data saved")
