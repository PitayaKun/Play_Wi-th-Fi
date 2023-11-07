# Define the IP address and port to listen on
monitor_ip = '0.0.0.0'  # Listen on all available network interfaces
monitor_port = 5500

# Define the PC's IP address and port to forward packets to
pc_ip = '192.168.3.3'  # Replace with your PC's IP address
pc_port = 5501  # Replace with the port number on your PC

# Params of monitor and connector
bandwidth = 20
subcarrier_num = int(bandwidth * 3.2)
packet_size = 18 + 4 * subcarrier_num
packet_num = 43
packet_stride = 3

# Params of recorder
frame_num = 53
skip_packet = 8

# Dirs of dataset
trainSet_directory = 'F:/Git_Repos/Play_Wi-th-Fi/sources/trainSet'
testSet_directory = 'F:/Git_Repos/Play_Wi-th-Fi/sources/testSet'

# Indexes of Null and Pilot OFDM subcarriers
# https://www.oreilly.com/library/view/80211ac-a-survival/9781449357702/ch02.html
nulls = {
    20: [x + 32 for x in [
        -32, -31, -30, -29,
        31, 30, 29, 0
    ]],

    40: [x + 64 for x in [
        -64, -63, -62, -61, -60, -59, -1,
        63, 62, 61, 60, 59, 1, 0
    ]],

    80: [x + 128 for x in [
        -128, -127, -126, -125, -124, -123, -1,
        127, 126, 125, 124, 123, 1, 0
    ]],

    160: [x + 256 for x in [
        -256, -255, -254, -253, -252, -251, -129, -128, -127, -5, -4, -3, -2, -1,
        255, 254, 253, 252, 251, 129, 128, 127, 5, 4, 3, 3, 1, 0
    ]]
}

pilots = {
    20: [x + 32 for x in [
        -21, -7,
        21, 7
    ]],

    40: [x + 64 for x in [
        -53, -25, -11,
        53, 25, 11
    ]],

    80: [x + 128 for x in [
        -103, -75, -39, -11,
        103, 75, 39, 11
    ]],

    160: [x + 256 for x in [
        -231, -203, -167, -139, -117, -89, -53, -25,
        231, 203, 167, 139, 117, 89, 53, 25
    ]]
}

# parameters of training
batch_size = 64
lr = 1e-3
epochs = 30
log_interval = 20
save_model_dir = './saved_models/'
label_to_index = {"attack": 0, "dash": 1, "jump": 2, "parry": 3, "static": 4}
# label_to_index = {"attack": 0, "parry": 1, "static": 2, "jump": 3}