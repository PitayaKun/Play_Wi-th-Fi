import os
import random
import time
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import pywt
from sources.loader import load_label
from sources.config import nulls, pilots, label_to_index
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def custom_phase_unwrap_rows(phase):
    unwrapped_phase = np.zeros_like(phase)

    for i in range(phase.shape[0]):
        unwrapped_phase[i, 0] = phase[i, 0]

        for j in range(1, phase.shape[1]):
            diff = phase[i, j] - phase[i, j - 1]

            if diff >= np.pi:
                unwrapped_phase[i, j] = unwrapped_phase[i, j - 1] + (diff - 2 * np.pi)
            elif diff < -np.pi:
                unwrapped_phase[i, j] = unwrapped_phase[i, j - 1] + (diff + 2 * np.pi)
            else:
                unwrapped_phase[i, j] = unwrapped_phase[i, j - 1] + diff

    return unwrapped_phase


def decompose(data_nparray, window_size, stride):
    row_length = data_nparray.shape[0]
    num_windows = ((row_length - window_size) // stride) + 1
    dec_data_list = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windowed_array = data_nparray[start:end, :]
        dec_data_list.append(windowed_array)

    return dec_data_list


def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(0, np.abs(x) - threshold)


def dwtd(signal, level=4, threshold=0.1, wavelet='sym4'):
    num_rows = signal.shape[0]
    # Perform wavelet denoising for each row
    denoised_signal = np.empty_like(signal)
    for i in range(num_rows):
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal[i], wavelet, level=level)
        # Apply thresholding to the detail coefficients
        for j in range(1, len(coeffs)):
            coeffs[j] = soft_threshold(coeffs[j], threshold)
        # Reconstruct the denoised signal
        denoised_signal[i] = pywt.waverec(coeffs, wavelet)
    return denoised_signal


class csi_dataset(Dataset):
    def __init__(self, dataset_path, bandwidth, window_size, stride, device,
                 test=False, amp=True, rm_nulls=False, rm_pilots=False):
        # Get a list of all dirs in the directory
        label_list = os.listdir(dataset_path)

        data = []
        gt = []

        # Load data
        for label in label_list:
            label_path = os.path.join(dataset_path, label)
            csi_list = load_label(label_path, int(bandwidth * 3.2))
            for csi in csi_list:
                if rm_nulls:
                    csi[:, nulls[bandwidth]] = 0
                if rm_pilots:
                    csi[:, pilots[bandwidth]] = 0
                if amp:
                    csi = np.abs(csi)
                    dec_csis = decompose(csi, window_size, stride)
                else:
                    dec_complex_csis = decompose(csi, window_size, stride)
                    dec_csis = [np.hstack((np.abs(arr), custom_phase_unwrap_rows(np.angle(arr, deg=True)))) for arr in dec_complex_csis]
                data += dec_csis
                gt_list = [label_to_index[label] for _ in range(len(dec_csis))]
                gt += gt_list

        # standardized_arrays = []
        # for arr in data:
        #     scaler = StandardScaler()
        #     standardized_arr = scaler.fit_transform(arr)
        #     standardized_arrays.append(standardized_arr)

        data = np.stack(data)

        num_samples, num_rows, num_features = data.shape
        data_reshaped = data.reshape(num_samples * num_rows, num_features)

        if test:
            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)
        else:
            scaler = StandardScaler()
            scaler.fit(data_reshaped)
            with open('scaler.pkl', 'wb') as file:
                pickle.dump(scaler, file)

        data_standardized = scaler.transform(data_reshaped)
        data_standardized = data_standardized.reshape(num_samples, num_rows, num_features)
        standardized_arrays = [data_standardized[i] for i in range(data_standardized.shape[0])]

        # denoised_data = []
        # for sample in standardized_data:
        #     denoised_data.append(dwtd(sample))
        # self.data = denoised_data

        self.data = standardized_arrays
        self.gt = gt
        self.dataset_path = dataset_path
        self.bandwidth = bandwidth
        self.window_size = window_size
        self.stride = stride
        self.device = device
        self.rm_nulls = rm_nulls
        self.rm_pilots = rm_pilots
        self.amp = amp

    def __getitem__(self, index):
        data = torch.tensor(self.data[index]).to(self.device).float()
        # if not self.amp:
        #     data = data.unsqueeze(0)
        gt = torch.tensor(self.gt[index]).to(self.device)
        return data, gt

    def __len__(self):
        return len(self.data)
