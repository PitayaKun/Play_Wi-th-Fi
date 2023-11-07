import datetime
import os.path
from datetime import time
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import utils
import sources.config as cf
import MyNet

# device selection
device = torch.device("cuda")
# device = torch.device("cpu")

# set seed
utils.set_seed(12)

# params search(window_size & stride)
n = 50
params_list = []

for i in range(40, n + 1):
    for j in range(1, 2):
        params_list.append((i, j))

for window_size, stride in params_list:
    params_str = "window_size_" + str(window_size) + "_stride_" + str(stride)
    params_path = os.path.join(cf.save_model_dir, params_str)

    # load train dataset
    train_dataset = utils.csi_dataset(
        cf.trainSet_directory,
        cf.bandwidth,
        window_size=window_size,
        stride=stride,
        device=device,
        amp=True,
        rm_nulls=True,
        rm_pilots=False)

    # load test dataset
    test_dataset = utils.csi_dataset(
        cf.testSet_directory,
        cf.bandwidth,
        window_size=window_size,
        stride=stride,
        device=device,
        test=True,
        amp=True,
        rm_nulls=True,
        rm_pilots=False)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=True)

    # 创建网络，优化器和损失函数
    # net = MyNet.Net(len(cf.label_to_index), window_size).to(device)
    net = MyNet.ABLSTM(cf.subcarrier_num, 32, window_size, len(cf.label_to_index)).to(device)
    # net = MyNet.CSINet(len(cf.label_to_index), window_size).to(device)
    # net = MyNet.LSTMModel(
    #     input_size=256,
    #     hidden_size=128,
    #     num_layers=window_size,
    #     output_size=cf.label_num,
    #     device=device
    # ).to(device)
    # net = MyNet.MLPModel(window_size*256, 1024, cf.label_num).to(device)
    optimizer = Adam(net.parameters(), cf.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 80
    # 开始训练
    for e in range(cf.epochs):
        net.train()
        count = 0
        total_losses = 0
        for batch_id, data in enumerate(train_loader):
            # 分离xy
            inputs, labels = data

            # 本次训练的batch大小（可能不足batch_size)
            num_batch = len(labels)
            count += num_batch
            optimizer.zero_grad()

            # calculate loss
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_losses += loss
            loss.backward()
            optimizer.step()

            # 训练次数达到log_interval时输出
            current_datetime = datetime.datetime.now()
            if (batch_id + 1) % cf.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\taverage_losses: {:.6f}".format(
                    current_datetime, e + 1, count, len(train_dataset), total_losses / (batch_id + 1)
                )
                print(mesg)

        # 一次epoch结束后对测试集进行评估
        with torch.no_grad():
            net.eval()
            correct = 0
            correct_attack = 0
            correct_dash = 0
            correct_jump = 0
            correct_parry = 0
            correct_static = 0
            for batch_id, data in enumerate(test_loader):
                # 分离xy
                inputs, labels = data
                # y和x之间计算损失
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct_attack += torch.sum((predicted == labels) & (labels == 0)).item()
                correct_dash += torch.sum((predicted == labels) & (labels == 1)).item()
                correct_jump += torch.sum((predicted == labels) & (labels == 2)).item()
                correct_parry += torch.sum((predicted == labels) & (labels == 3)).item()
                correct_static += torch.sum((predicted == labels) & (labels == 4)).item()
                correct += (predicted == labels).sum().item()
            # log
            print(f'attack:{correct_attack}')
            print(f'dash:{correct_dash}')
            print(f'jump:{correct_jump}')
            print(f'parry:{correct_parry}')
            print(f'static:{correct_static}')

            accuracy = 100 * correct / len(test_dataset)
            print(f'Epoch {e + 1}, Accuracy: {accuracy:.4f}%')
            # save checkpoint if meet requirements
            if accuracy > best_acc:
                if not os.path.exists(params_path):
                    os.mkdir(params_path)
                best_acc = accuracy
                # 保存模型
                net.eval().cpu()
                save_model_filename = "epoch_" + str(e + 1) + "_acc_" + str(accuracy) + ".pth"
                save_model_path = os.path.join(params_path, save_model_filename)
                torch.save(net.state_dict(), save_model_path)
                print("\nDone, trained model saved at", save_model_path)
                net.to(device).train()



