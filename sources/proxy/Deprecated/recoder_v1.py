# import os
# import time
#
# from connector import capture_packets, pc_socket
# from datetime import datetime
# import sources.config as cf
#
#
# def save_packets(data_num, save_path, packet_size, subcarrier_num):
#     with open(save_path, 'wb') as file:
#         for _ in range(data_num):
#             # Receive data from the Raspberry Pi
#             packet_bytes = capture_packets(pc_socket, packet_size)
#             # Write the bytes data to the file
#             file.write(packet_bytes[-subcarrier_num * 4:])
#
#
# def generate_save_path(dir_path):
#     # Get the current date and time
#     current_datetime = datetime.now()
#
#     # Format the date and time as a string (you can customize the format)
#     formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
#
#     # Combine the directory path and the formatted datetime to create the new directory path
#     save_path = os.path.join(dir_path, formatted_datetime)
#
#     return save_path
#
#
# if __name__ == "__main__":
#     dataset_name = input('label name: ')
#     dataset_path = os.path.join(cf.trainSet_directory, dataset_name)
#     if not os.path.exists(dataset_path):
#         os.makedirs(dataset_path)
#
#     while True:
#         time.sleep(3)
#         print('start recording...')
#         sample_path = generate_save_path(dataset_path)
#         save_packets(cf.frame_num, sample_path, cf.packet_size, cf.subcarrier_num)
#         print('sample saved')
#
#         # Create a flag to exit recorder
#         flag = input('input 1 to continue or 0 to exit:')
#
#         if flag == '0':
#             break
#
#     print('end recording')
