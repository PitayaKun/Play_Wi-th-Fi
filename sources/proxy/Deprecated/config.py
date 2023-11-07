bandwidth = 80
subcarrier_num = int(bandwidth * 3.2)
packet_num = 1
bytes_per_line = 16
prefixed_lines = 5
lines_num = 69
bytearray_size = packet_num * subcarrier_num * 4

# settings of monitor
monitor_ip = '192.168.3.7'
monitor_port = 1209
