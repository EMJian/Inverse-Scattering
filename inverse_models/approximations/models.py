import numpy as np

from config import Config


# noinspection PyShadowingNames
class ApproximateModels:

    def __init__(self):

        # Physical parameters
        self.frequency = Config.frequency
        self.wavelength = 3e8 / self.frequency
        self.wave_number = 2 * np.pi / self.wavelength

        # Room parameters
        self.m = Config.inverse_grid_number
        self.number_of_grids = self.m ** 2

        # Sensor parameters
        self.txrx_pairs = Config.txrx_pairs
        self.number_of_tx = Config.number_of_transmitters
        self.number_of_rx = Config.number_of_receivers

        # Other parameters
        self.nan_remove = Config.nan_remove

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

    def born_approx(self, direct_field, incident_field, total_forward_field, integral_values):

        def _get_born_model(incident_field, integral_values):
            A = np.zeros((len(self.txrx_pairs), self.number_of_grids), dtype=complex)
            for i, pair in enumerate(self.txrx_pairs):
                A[i, :] = self.wave_number ** 2 * \
                          np.multiply(integral_values[pair[1], :], np.transpose(incident_field[:, pair[0]]))
            return A

        def _get_born_data(direct_field, total_forward_field):
            direct_field = self.remove_nan_values(direct_field)
            data = total_forward_field - direct_field
            data = data.reshape(data.size, order='F')
            return data

        A = _get_born_model(incident_field, integral_values)
        data = _get_born_data(direct_field, total_forward_field)
        return A, data

    def rytov_approx(self, direct_field, incident_field, total_forward_field, integral_values):

        def _get_rytov_model(direct_field, incident_field, integral_values):
            A = np.zeros((len(self.txrx_pairs), self.number_of_grids), dtype=complex)
            for i, pair in enumerate(self.txrx_pairs):
                A[i, :] = self.wave_number ** 2 * \
                          np.divide(np.multiply(integral_values[pair[1], :],
                                                np.transpose(incident_field[:, pair[0]])),
                                    direct_field[pair[1], pair[0]])
            return A

        def _get_rytov_data(direct_field, total_forward_field):
            direct_field = self.remove_nan_values(direct_field)
            data = np.log(np.divide(total_forward_field, direct_field))
            data = data.reshape(data.size, order='F')
            return data

        A = _get_rytov_model(direct_field, incident_field, integral_values)
        data = _get_rytov_data(direct_field, total_forward_field)
        return A, data

    def prytov_approx(self, direct_field, incident_field, total_forward_power, incident_forward_power, integral_values):

        def _get_prytov_model(direct_field, incident_field, integral_values):
            A = np.zeros((len(self.txrx_pairs), self.number_of_grids), dtype=complex)
            for i, pair in enumerate(self.txrx_pairs):
                A[i, :] = np.real(self.wave_number ** 2
                                  * np.divide(np.multiply(integral_values[pair[1], :],
                                                          np.transpose(incident_field[:, pair[0]])),
                                              direct_field[pair[1], pair[0]]))
            return A

        def _get_prytov_data(total_forward_power, incident_forward_power):
            data = (total_forward_power - incident_forward_power) / (10 * np.log10(np.exp(2)))
            data = data.reshape(data.size, order='F')
            return data

        A = _get_prytov_model(direct_field, incident_field, integral_values)
        data = _get_prytov_data(total_forward_power, incident_forward_power)
        return A, data
