# -*- coding: utf-8 -*-
import imageio
import numpy as np
import pickle
import os

if not os.path.exists("./cifar10/train"):
    os.makedirs("./cifar10/train")
if not os.path.exists("./cifar10/test"):
    os.makedirs("./cifar10/test")

def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


# 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
for j in range(1, 6):
    dataName = "data_batch_" + str(j)  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。
    Xtr = load_file('./cifar10/'+dataName)
    print(dataName + " is loading...")

    for i in range(0, 10000):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
        img = img.transpose(1, 2, 0)  # 读取image
        picName = './cifar10/train/' + str(Xtr['labels'][i]) + '_' + str(i + (j - 1)*10000) + '.jpg'  # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
        imageio.imwrite(picName, img)
    print(dataName + " loaded.")

print("test_batch is loading...")

# 生成测试集图片
testXtr = load_file('./cifar10/test_batch')
for i in range(0, 10000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = './cifar10/test/' + str(testXtr['labels'][i]) + '_' + str(i) + '.jpg'
    imageio.imwrite(picName, img)
print("test_batch loaded.")
