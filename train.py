# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import dataset
from model import LeNet
from torch.optim.lr_scheduler import StepLR
import os

USE_MULTI_GPU = True

# 检测机器是否有多张显卡
if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device_ids = [0, 1]
else:
    MULTI_GPU = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

write = SummaryWriter('result')

batch_size = 102400
epoch_num = 200

# 定义dataloader
cifar_train = dataset.Cifar10Dataset('./cifar10/train', transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
cifar_test = dataset.Cifar10Dataset('./cifar10/test', transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
cifar_train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
cifar_test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)


net = LeNet()
if MULTI_GPU:
    net = nn.DataParallel(net,device_ids=device_ids)
net.to(device)
criteon = nn.CrossEntropyLoss()

optimizer=optim.Adam(net.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
if MULTI_GPU:
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    scheduler = nn.DataParallel(scheduler, device_ids=device_ids)

# print(net)
for epoch in range(epoch_num):
    for batchidx, (label, img) in enumerate(cifar_train_loader):
        net.train()
        logits = net(img.to(device))
        loss = criteon(logits, label.long().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.module.step()
        scheduler.module.step()
    print("epoch:{} loss:{}".format(epoch, loss.item()))
    write.add_scalar(tag='train_loss', global_step=epoch, scalar_value=loss.item())

    net.eval()
    with torch.no_grad():
        total_num = 0
        total_correct = 0
        for label, img in cifar_test_loader:
            logits = net(img.to(device))
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(label.to(device), pred).float().sum()
            total_num += img.size(0)
        acc = total_correct / total_num
        write.add_scalar(
            tag="eval_acc",
            global_step=epoch,
            scalar_value=acc
        )
        if MULTI_GPU:
            print("EVAL--epoch:{}  acc:{} Lr:{}".format(
                epoch,
                acc,
                optimizer.module.state_dict()['param_groups'][0]['lr'])
            )
            write.add_scalar(
                tag="Learning Rate",
                global_step=epoch,
                scalar_value=optimizer.module.state_dict()['param_groups'][0]['lr']
            )
        else:
            print("EVAL--epoch:{}  acc:{} Lr:{}".format(
                epoch,
                acc,
                optimizer.state_dict()['param_groups'][0]['lr'])
            )
            write.add_scalar(
                tag="Learning Rate",
                global_step=epoch,
                scalar_value=optimizer.state_dict()['param_groups'][0]['lr']
            )