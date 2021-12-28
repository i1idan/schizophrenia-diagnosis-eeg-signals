import numpy as np
from typing import Tuple
from pyts.image import GramianAngularField
import scipy.io
import cv2
from sklearn.model_selection import train_test_split


def load_data(data_path: str = './DATA.mat', seed: int = 1234) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    # Load data
    data = scipy.io.loadmat(data_path)
    # extract-information
    H = data['H'].reshape(-1)  # healthy
    print(f"[INFO] H.shape: {H.shape}, each sample has (channels, values): {H[0].shape} !")
    S = data['S'].reshape(14)  # schizopherni
    print(f"[INFO] S.shape: {H.shape}, each sample has (channels, values): {S[0].shape} !")
    print(f"[INFO] S.shape: {S.shape}")
    CHAN = data['CHAN']  # channels
    print(f"[INFO] CHAN.shape: {CHAN.shape}")
    Fs = data['Fs']  # ferequency
    print(f"[INFO] Fs.shape: {Fs.shape}")
    # Define Static variable
    healthy_samples = H.shape[0]  # number of healthy samples
    schizo_samples = S.shape[0]  # number of confirmed schizophrenia samples
    channel_size = CHAN.shape[1]  # number of channels
    sub = 15000  # We use a small subset of information
    print(
        f"[INFO] healthy_samples: {healthy_samples}, schizo_samples: {schizo_samples}, channel_size: {channel_size}, and sub: {sub}")
    # Get normal samples
    normal = []

    for j in range(healthy_samples):
        for i in range(channel_size):
            x = H[j][i, :sub]
            x = x.reshape(1, -1)
            gasf = GramianAngularField(image_size=112, method='summation')
            x_gasf = gasf.fit_transform(x)
            normal.append(x_gasf[0])


    for j in range(healthy_samples):
        for i in range(channel_size):
            x = H[j][i, :sub]
            x = x.reshape(1, -1)
            gadf = GramianAngularField(image_size=112, method='difference')
            x_gadf = gadf.fit_transform(x)
            normal.append(x_gadf[0])
    normal = np.array(normal)
    print(f"[INFO] normal samples: {normal.shape}")

    # Get schizophrenia samples
    schizo = []

    for j in range(schizo_samples):
        for i in range(channel_size):
            x = S[j][i, :sub]
            x = x.reshape(1, -1)
            gasf = GramianAngularField(image_size=112, method='summation')
            x_gasf = gasf.fit_transform(x)
            schizo.append(x_gasf[0])

    for j in range(schizo_samples):
        for i in range(channel_size):
            x = S[j][i, :sub]
            x = x.reshape(1, -1)
            gadf = GramianAngularField(image_size=112, method='difference')
            x_gadf = gadf.fit_transform(x)
            schizo.append(x_gadf[0])
    schizo = np.array(schizo)
    print(f"[INFO] schizo samples: {schizo.shape}")
    # get data and labels
    labels = np.concatenate([np.zeros(len(normal)), np.ones(len(schizo))])
    data = np.concatenate([normal, schizo], axis=0)
    data = np.array([np.expand_dims(d, axis=-1) for d in data])
    print(f"[INFO] labels: {labels.shape}")
    print(f"[INFO] data: {data.shape}")

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=seed,
                                                        shuffle=True,
                                                        stratify=labels
                                                        )

    print(f"[INFO] x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
    print(f"[INFO] x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}")
    return (x_train, y_train), (x_test, y_test)
