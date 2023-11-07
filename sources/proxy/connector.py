import socket
import datetime
import sources.config as cf
from sources.loader import csi_decoder
import torch
import numpy as np
import sources.model.MyNet as MyNet
import pickle
from sklearn.preprocessing import StandardScaler


# listen on specified port and send the packet to target socket
def capture_packets(monitor_socket, packet_size, capture_num):
    byte_packets = bytearray()
    for _ in range(capture_num):
        # Receive data from the Raspberry Pi
        data, addr = monitor_socket.recvfrom(packet_size)
        # print(datetime.datetime.now())
        byte_packets += data[-cf.subcarrier_num * 4:]
    return byte_packets


def save_scaler():
    # Create a UDP socket on the PC to send forwarded packets
    pc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pc_socket.bind((cf.pc_ip, cf.pc_port))

    try:
        pilot_num = cf.packet_num - cf.packet_stride

        pilot_packets = capture_packets(pc_socket, cf.packet_size, pilot_num)
        csi_list = []

        while len(csi_list) < 1000:
            # Receive data from the Raspberry Pi
            packet_bytes = capture_packets(pc_socket, cf.packet_size, cf.packet_stride)
            csi_buffer = pilot_packets + packet_bytes

            # Decode csi
            csi_cmplx = csi_decoder(csi_buffer, cf.packet_num, cf.subcarrier_num)
            csi_cmplx[:, cf.nulls[cf.bandwidth]] = 0
            # csi_cmplx[:, cf.pilots[cf.bandwidth]] = 0
            csi_amp = np.abs(csi_cmplx)

            csi_list.append(csi_amp)
            pilot_packets = csi_buffer[cf.packet_stride * cf.subcarrier_num * 4:]
            print(len(csi_list))

        data = np.stack(csi_list)

        num_samples, num_rows, num_features = data.shape
        data_reshaped = data.reshape(num_samples * num_rows, num_features)
        scaler = StandardScaler()
        scaler.fit(data_reshaped)
        with open('scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

    finally:
        pc_socket.close()


if __name__ == "__main__":
    # Initialization
    save_scaler()
    # Load model
    device = torch.device("cpu")
    # net = MyNet.Net(len(cf.label_to_index), 42).to(device)
    net = MyNet.ABLSTM(cf.subcarrier_num, 32, cf.packet_num, len(cf.label_to_index)).to(device)
    state_dict = torch.load(
        r'F:\Git_Repos\Play_Wi-th-Fi\sources\model\saved_models\window_size_43_stride_1\epoch_29_acc_92.8.pth')
    net.load_state_dict(state_dict)
    net.eval()
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Create a UDP socket on the PC to send forwarded packets
    pc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pc_socket.bind((cf.pc_ip, cf.pc_port))
    try:
        # Record the first (window size - stride) packets
        pilot_num = cf.packet_num - cf.packet_stride
        pilot_size = pilot_num * cf.subcarrier_num * 4
        pilot_packets = capture_packets(pc_socket, cf.packet_size, pilot_num)

        while True:
            # Receive data from the Raspberry Pi
            packet_bytes = capture_packets(pc_socket, cf.packet_size, cf.packet_stride)
            csi_buffer = pilot_packets + packet_bytes

            # Decode csi
            csi_cmplx = csi_decoder(csi_buffer, cf.packet_num, cf.subcarrier_num)
            csi_cmplx[:, cf.nulls[cf.bandwidth]] = 0
            # csi_cmplx[:, cf.pilots[cf.bandwidth]] = 0
            csi_amp = np.abs(csi_cmplx)

            standardized_amp = scaler.fit_transform(csi_amp)
            # Wrap csi
            csi_tensor = torch.tensor(standardized_amp).float().unsqueeze(0)

            # Classify
            with torch.no_grad():
                classification = net(csi_tensor)
            _, predicted = torch.max(classification.data, 1)
            # 计算Top-k精确度
            # _, indices = torch.topk(classification, 3)
            # print(datetime.datetime.now())
            print(predicted)
            pilot_packets = csi_buffer[cf.packet_stride * cf.subcarrier_num * 4:]

    except KeyboardInterrupt:
        # Close the socket
        pc_socket.close()
