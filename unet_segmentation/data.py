import tensorflow_datasets as tfds


class Dataset:

    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    print(dataset.shape)
    print(info)
