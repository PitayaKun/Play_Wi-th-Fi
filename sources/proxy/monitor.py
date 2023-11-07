import socket
from sources.config import *

# Create a UDP socket on the Raspberry Pi to listen for incoming packets
monitor_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
monitor_socket.bind((monitor_ip, monitor_port))

# Create a TCP socket on the PC to send forwarded packets
pc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# listen on specified port and send the packet to target socket
def forward_packets(recv_socket, send_socket, ip, port):
    # Receive data from the Raspberry Pi
    data, addr = recv_socket.recvfrom(packet_size)
    # Forward the received data to the PC
    send_socket.sendto(data, (ip, port))


if __name__ == "__main__":
    try:
        while True:
            forward_packets(monitor_socket, pc_socket, pc_ip, pc_port)
    except KeyboardInterrupt:
        # Close the sockets
        monitor_socket.close()
        pc_socket.close()
