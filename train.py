import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import dataset
from model import LeNet
from torch.optim.lr_scheduler import StepLR

write = SummaryWriter('result')

batch_size = 71680
epoch_num = 200

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

# label, img = iter(cifar_train_loader).next()

criteon=nn.CrossEntropyLoss()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
net=LeNet().to(device)
# write.add_graph(net)
optimizer=optim.Adam(net.parameters(), lr=1e-2)
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

# print(net)
for epoch in range(epoch_num):
    for batchidx, (label, img) in enumerate(cifar_train_loader):
        net.train()
        logits = net(img.to(device))
        loss = criteon(logits, label.long().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
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
        print("EVAL--epoch:{}  acc:{} Lr:{}".format(epoch, acc, optimizer.state_dict()['param_groups'][0]['lr']))
        write.add_scalar(tag="eval_acc", global_step=epoch, scalar_value=acc)
        write.add_scalar(tag="Learning Rate", global_step=epoch, scalar_value=optimizer.state_dict()['param_groups'][0]['lr'])