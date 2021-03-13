import os
import numpy as np
import scipy.io as sio

if __name__ == '__main__':

    path = r"/extras/compare"

    filename = os.path.join(path, "scattered_field.mat")
    matlab_sf = sio.loadmat(filename)
    matlab_sf = matlab_sf["E_s"]

    filename = os.path.join(path, "scattered_field.npy")
    python_sf = np.load(filename)

    print("Mean of difference between scattered field matrices: ", np.mean(matlab_sf - python_sf))
    print("Mean of difference between absolute values of scattered field matrices: ", np.mean(np.abs(matlab_sf) - np.abs(python_sf)))
    print("Mean of absolute difference between scattered field matrices: ", np.mean(np.abs(matlab_sf - python_sf)))

    filename = os.path.join(path, "incident_field.mat")
    matlab_if = sio.loadmat(filename)
    matlab_if = matlab_if["E_d"]

    filename = os.path.join(path, "incident_field.npy")
    python_if = np.load(filename)

    print("Mean of difference between incident field matrices: ", np.mean(matlab_if - python_if))
    print("Mean of difference between absolute values of incident field matrices: ",
          np.mean(np.abs(matlab_if) - np.abs(python_if)))
    print("Mean of absolute difference between incident field matrices: ", np.mean(np.abs(matlab_if - python_if)))

    filename = os.path.join(path, "total_field.mat")
    matlab_tf = sio.loadmat(filename)
    matlab_tf = matlab_tf["E_ds"]

    filename = os.path.join(path, "total_field.npy")
    python_tf = np.load(filename)

    print("Mean of difference between total field matrices: ", np.mean(matlab_tf - python_tf))
    print("Mean of difference between absolute values of total field matrices: ",
          np.mean(np.abs(matlab_tf) - np.abs(python_tf)))
    print("Mean of absolute difference between total field matrices: ", np.mean(np.abs(matlab_tf - python_tf)))
