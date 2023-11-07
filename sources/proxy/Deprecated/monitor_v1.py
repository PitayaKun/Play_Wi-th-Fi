# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# sudo tcpdump -i wlan0 dst port 5500 -XX

import subprocess
import socket
import datetime
from config import *

pc_ip = '192.168.3.3'  # Replace with your PC's IP address
pc_port = 5501  # Replace with the port number on your PC

def capture_packets(target_socket, ip, port):
    target_socket.setblocking(False)
    try:
        # Run tcpdump with the desired interface and capture packets to stdout
        tcpdump_command = ["tcpdump", "-i", "wlan0", "dst", "port", "5500", "-XX"]

        process = subprocess.Popen(tcpdump_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)

        # Initialize a buffer to collect csi of the current packet
        csi_buffer = bytearray(bytearray_size)

        # Read and process the output in real-time
        line_index = -1 - prefixed_lines
        packet_counter = 0
        for line in process.stdout:
            line_index += 1
            # Skip the first several lines of each packet
            if line_index == - prefixed_lines:
                print(line)
            if line_index < -1:
                continue
            # Get the data bytes of first subcarrier
            elif line_index == -1:
                data_bytes = process_line(line, True)
                csi_buffer[packet_counter * subcarrier_num * 4: (packet_counter * subcarrier_num + 1) * 4] = data_bytes
            else:
                data_bytes = process_line(line)
                start_index = (packet_counter * subcarrier_num + 1) * 4
                csi_buffer[start_index + line_index * bytes_per_line: start_index + (line_index + 1) * bytes_per_line] \
                    = data_bytes

            if line_index == lines_num - prefixed_lines - 1:
                packet_counter += 1
                if packet_counter == packet_num:
                    # Send the entire csi packet
                    target_socket.sendto(csi_buffer, (ip, port))
                    print(datetime.datetime.now())
                    # Reset packets counter
                    packet_counter = 0

                # Reset lines counter
                line_index = -1 - prefixed_lines

    except KeyboardInterrupt:
        # Handle keyboard interrupt (e.g., Ctrl+C) to stop tcpdump gracefully
        process.terminate()


def process_line(packet_line, first_subcarrier_flag=False):
    # Extract the hexadecimal values from the lines
    data_part = packet_line.split()
    if first_subcarrier_flag:
        data_str = ''.join(data_part[-2])
    else:
        data_str = ''.join(data_part[1:-1])
    data_bytes = bytes.fromhex(data_str)

    return data_bytes


if __name__ == "__main__":
    monitor_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    capture_packets(monitor_socket, pc_ip, pc_port)

    # Close the sockets when done
    monitor_socket.close()
