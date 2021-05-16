# LeNet-5的Pytorch实现

## 简介

这里利用pytorch实现了LeNet-5

目录如下

> │  dataset.py				定义了数据集读取的类
> │  make_dataset.py	将CIFAR10数据集转为普通的文件夹形式
> │  model.py				模型定义
> │  train.py					训练文件夹

## 使用

1. 进入项目目录

   `cd ./LeNet-5_by_Pytorch `

2. 下载数据集

   下载[CIFAR10数据集](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)到`./cifa10`文件夹下

3. 运行`make_dataset.py`

   `python3 make_dataset.py`

4. 开始训练

   `python3 train.py`

## 相关

若在使用中有疑问或发现代码有错误，都欢迎联系我或提issue😉

<a target="_blank" href="http://mail.qq.com/cgi-bin/qm_share?t=qm_mailme&email=RSYaJi0gKycsKwU0NGsmKig" style="text-decoration:none;"><img src="http://rescdn.qqmail.com/zh_CN/htmledition/images/function/qm_open/ico_mailme_01.png"/></a>

