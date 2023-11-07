import os
import sources.config as cf
import numpy as np


def load(data_path, subcarrier_num):
    # Open the file in read binary mode ('rb') and read its content
    with open(data_path, 'rb') as file:
        data_bytes = file.read()
    # remove the first x packets
    data_bytes = data_bytes[3*cf.skip_packet:]
    frame_num = int(len(data_bytes) / (subcarrier_num * 4))
    csi_array = csi_decoder(data_bytes, frame_num, subcarrier_num)

    return csi_array


def load_label(label_path, subcarrier_num):
    # Get a list of all files in the directory
    file_list = os.listdir(label_path)
    csi_list = []
    # Loop through the list of files and read each file
    for file_name in file_list:
        file_path = os.path.join(label_path, file_name)
        # Check if the item in the directory is a file (not a subdirectory)
        if os.path.isfile(file_path):
            # Specify the full path of the file
            csi = load(file_path, subcarrier_num)
            csi_list.append(csi)

    return csi_list


def csi_decoder(data_bytes, frame_num, subcarrier_num):
    # Convert bytes to numpy array
    csi_np = np.frombuffer(
        data_bytes,
        dtype=np.int16,
        count=frame_num * subcarrier_num * 2
    )

    # Cast numpy 1-d array to matrix
    csi_np = csi_np.reshape((frame_num, subcarrier_num * 2))

    # Convert csi into complex numbers
    csi_cmplx = np.fft.fftshift(
        csi_np[:frame_num, ::2] + 1.j * csi_np[:frame_num, 1::2], axes=(1,)
    )

    return csi_cmplx


if __name__ == "__main__":
    label_name = input('input label to load: ')
    label_path = os.path.join(cf.trainSet_directory, label_name)
    csis = load_label(label_path, cf.subcarrier_num)
    print(len(csis))
