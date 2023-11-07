import time
import torch
import model.MyNet as MyNet
from proxy.recoder import generate_save_path
import config as cf
import os

# Load model
device = torch.device("cpu")
net = MyNet.ABLSTM(256, 128, len(cf.label_to_index)).to(device)
state_dict = torch.load(r'F:\Git_Repos\Play_Wi-th-Fi\sources\model\saved_models\window_size_40_stride_1\epoch_3_acc_75.44444444444444.pth')
net.load_state_dict(state_dict)
net.eval()
testdata = torch.rand(40, 256)
testdata = testdata.unsqueeze(0)
start_time = time.time()

with torch.no_grad():
    classification = net(testdata)
_, predicted = torch.max(classification.data, 1)


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")