import os
import time
import tkinter as tk
import socket
import random
from sources.proxy.connector import capture_packets
from datetime import datetime
import sources.config as cf


def random_label_generator(input_list):
    while True:
        yield random.choice(input_list)


def save_packets(data_num, save_path, packet_size):
    # Create a UDP socket on the PC to receive forwarded packets
    pc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pc_socket.bind((cf.pc_ip, cf.pc_port))

    try:
        csi_bytes = capture_packets(pc_socket, packet_size, data_num)
        with open(save_path, 'wb') as file:
            # Write the bytes data to the file
            file.write(csi_bytes)
    finally:
        pc_socket.close()


def generate_save_path(dir_path):
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time as a string (you can customize the format)
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Combine the directory path and the formatted datetime to create the new directory path
    save_path = os.path.join(dir_path, formatted_datetime)

    return save_path


if __name__ == "__main__":
    label_list = list(cf.label_to_index.keys())
    for label in label_list:
        label_path = os.path.join(cf.trainSet_directory, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)

    label_generator = random_label_generator(label_list)

    while True:
        for label in label_list:
            # random_label = next(label_generator)
            random_label = label
            print("label:                                              " + random_label)
            time.sleep(0.3)
            print("recording starts")
            # record
            dataset_path = os.path.join(cf.trainSet_directory, random_label)
            sample_path = generate_save_path(dataset_path)
            save_packets(cf.frame_num, sample_path, cf.packet_size)
            print("waiting..........")
            time.sleep(2)

    # # Create a tkinter window
    # window = tk.Tk()
    # window.title("Random Label Generator")
    #
    # # Get the screen width and height
    # screen_width = window.winfo_screenwidth()
    # screen_height = window.winfo_screenheight()
    #
    # # Calculate the window position to center it on the screen
    # window_width = 800
    # window_height = 400
    # x = (screen_width - window_width) // 2
    # y = (screen_height - window_height) // 2
    #
    # # Set the window position and size
    # window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    #
    # # Create a label to display the random element
    # element_label = tk.Label(window, text="", font=("Helvetica", 60))
    # element_label.pack(padx=20, pady=20)
    #
    # # State variable to track whether to display the element or clear the window
    # display_label = False
    # stop_flag = False
    #
    # # Function to toggle between displaying the element and clearing the window
    # def toggle_display():
    #     global display_label, stop_flag
    #     if not stop_flag:
    #         if display_label:
    #             random_label = next(label_generator)
    #             element_label.config(text=random_label)
    #
    #
    #
    #             # Schedule to clear the window after 1 second (1000 milliseconds)
    #             window.after(1, toggle_display)
    #
    #         else:
    #             # record
    #             dataset_path = os.path.join(cf.trainSet_directory, random_label)
    #             sample_path = generate_save_path(dataset_path)
    #             print(datetime.now())
    #             save_packets(cf.frame_num, sample_path, cf.packet_size, cf.subcarrier_num)
    #             print(datetime.now())
    #             print("record success!")
    #
    #             element_label.config(text="")
    #             # Schedule to display the next element after 1 second (1000 milliseconds)
    #             window.after(1000, toggle_display)
    #         display_label = not display_label
    #
    # # Function to start generating random elements
    # def start_generation():
    #     global stop_flag
    #     stop_flag = False
    #     toggle_display()
    #
    # # Function to stop generating random elements
    # def stop_generation():
    #     global stop_flag
    #     stop_flag = True
    #
    #
    # # Create Start and Stop buttons
    # start_button = tk.Button(window, text="Start", command=start_generation)
    # start_button.pack(pady=10)
    # stop_button = tk.Button(window, text="Stop", command=stop_generation)
    # stop_button.pack(pady=10)
    #
    # window.mainloop()

    # while True:
    #     time.sleep(3)
    #     print('start recording...')
    #     sample_path = generate_save_path(dataset_path)
    #     save_packets(cf.frame_num, sample_path, cf.packet_size, cf.subcarrier_num)
    #     print('sample saved')
    #
    #     # Create a flag to exit recorder
    #     flag = input('input 1 to continue or 0 to exit:')
    #
    #     if flag == '0':
    #         break
