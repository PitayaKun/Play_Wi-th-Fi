import socket
import numpy as np
from config import *
import time

pc_ip = '192.168.3.3'  # Replace with your PC's IP address
pc_port = 5501  # Replace with the port number on your PC
# Function to receive a fixed number of bytes
def receive_exact_length(target_socket, length):
    data = bytearray()
    while len(data) < length:
        chunk = target_socket.recv(length - len(data))
        if not chunk:
            print(len(data))
            raise ValueError("Connection closed prematurely")
        data += chunk
    return data
def capture_packets(monitor_socket):
    # Receive data from the Raspberry Pi
    data, addr = monitor_socket.recvfrom(1024)
    return data

if __name__ == "__main__":
    # Define the monitor address and port
    monitor_address = (pc_ip, pc_port)

    # Create a socket to connect to the monitor
    connector_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    connector_socket.bind(monitor_address)
    try:
        # connector_socket.connect(monitor_address)
        # print("Connected to monitor")

        # data_bytes = bytearray((subcarrier_num - 1) * 4)

        while True:
            # Receive the data as a bytearray
            # data_bytes = receive_exact_length(connector_socket, bytearray_size)

            packet_bytes = capture_packets(connector_socket)

            # # Convert CSI bytes to numpy array
            csi_np = np.frombuffer(
                packet_bytes,
                dtype=np.int16,
                count=packet_num * subcarrier_num * 2
            )
            # # Convert csi into complex numbers
            csi_complex = csi_np[::2] + 1.j * csi_np[1::2]

            print(time.time())
            # csi_amp = np.abs(csi_complex)
            # print(len(data_bytes))
            # TODO:prediction

    except KeyboardInterrupt:
        print("connector is closing...")
    finally:
        connector_socket.close()
