import torch
import numpy as np
import torch.nn.functional as F  # 激励函数都在这

# # numpy数据转成pytorch数据
# np_data=np.arange(6).reshape((2,3))
# torch_data=torch.from_numpy(np_data)
# # torch.mm()表示矩阵乘法
# tensor2=torch.mm(torch_data,torch_data.T)
# # 对象.numpy()表示pytorch数据类型转成numpy数据类型
# tensor2.numpy()
# # abs 绝对值计算
# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)  # 转换成32位浮点 tensor
# torch.abs(tensor)
# # sin   三角函数 sin
# torch.sin(tensor)
# # mean  均值
# torch.mean(tensor)
#
# # 求梯度(1)
# from torch.autograd import Variable
# tensor = torch.FloatTensor([[1,2],[3,4]])
# variable = Variable(tensor, requires_grad=True)
# # 把张量变成变量，tensor不可以反向传播，variable可以，requires_grad：获得梯度
# v_out = torch.mean(variable*variable)
# v_out.backward()
# print(variable.grad)
#
#
# # 求梯度(2)
# b = torch.FloatTensor([[1, 2], [3, 4]])
# b.requires_grad_(True)# 设置是否可以对b求梯度
# c = torch.FloatTensor([[12, 31], [21, 51]])
# c.requires_grad_(True)
# d = torch.mm(b,c)
# d.backward(torch.FloatTensor([[1,0.1],[1,1]]))# 这里可以设置反向传播获得的梯度的倍数
# print(b.grad)
# print(b.data)
#
#
# # 激励函数
# x = torch.linspace(-5, 5, 200)
# x_np = x.data.numpy()
# # 几种常用的 激励函数
# y_relu = F.relu(x).data.numpy()
# y_sigmoid = torch.sigmoid(x).data.numpy()
# y_tanh = torch.tanh(x).data.numpy()
# y_softmax = F.softmax(x, dim=0)
# # softmax函数在nn.functional里面。用dim指明维度，dim=0表示按列计算；dim=1表示按行计算。一维的情况dim=0，dim=1会报错。
#
#
# # 搭建神经网络
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# # torch.unsqueeze()这个函数主要是对数据维度进行扩充。
# # 给dim的位置加上维数为一的维度，比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1,3）。
# # a.unsqueeze(dim) 就是在a中指定位置dim加上一个维数为1的维度。
# # 还有一种形式就是b=torch.unsqueeze(a，dim) a就是在a中指定位置dim加上一个维数为1的维度。
# y = x.pow(2) + 0.2*torch.rand(x.size())
# # plt.scatter(x.data.numpy(), y.data.numpy())
# # plt.show()
# class Net(torch.nn.Module):
# # 搭建神经网络时要继承torch.nn.Module这个类
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()                         # 继承 __init__ 功能，官方操作，必须要这么做。
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出。
#         self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出
#     # 这里torch.nn.Linear其实是一个类，实例出来的对象其实是一个函数。
#     # 此处包含了搭建层时的信息，n_feature为输入的特征数，n_hidden为隐藏层神经元数， n_output为输出层神经元个数。
#     # 如果需要加深层数，则self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
#     def forward(self,x):
#         x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
#         x = torch.sigmoid(self.predict(x))  # 输出值
#         return x
#     # 前向传播的过程
# net = Net(n_feature=1, n_hidden=10, n_output=1)   # 权重初始化在这步自动完成。
# # optimizer 是训练的工具，运用哪种梯度下降的优化器。
# optimizer = torch.optim.SGD(net.parameters(), lr=0.2,weight_decay=0.01)
# # 定义随机梯度下降优化器。传入 net 的所有参数, lr为学习率，weight_decay为L2正则化中的lambda参数。pytorch只支持L2正则化。
# # 其他的梯度下降法：
# # opt_Momentum= torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8) # momentum 动量加速,在SGD函数里指定momentum的值即可
# # opt_RMSprop= torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)  # RMSprop 指定参数alpha
# # opt_Adam= torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))   # Adam 参数betas=(0.9, 0.99)
#
# loss_func = torch.nn.MSELoss()                # 回归问题的损失函数。预测值和真实值的误差计算公式 (均方差函数)。回归用于预测。
# # loss_func = torch.nn.CrossEntropyLoss()     # 分类问题的损失函数，nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合。分类用于识别。
# for t in range(100):
#     prediction = net(x)              # 喂给 net 训练数据 x, 输出预测值，net也是一个实例出来的函数。
#     loss = loss_func(prediction, y)  # 计算预测与真实之间的误差，注意预测和真实的前后位置
#
#     optimizer.zero_grad()   # 清空上一步的梯度值
#     loss.backward()         # 误差反向传播, 计算参数更新值
#     optimizer.step()        # 将参数更新值施加到 net 的 parameters 上，实行梯度下降。
#
#
# # 分类问题，网络搭建过程同上，网络结构（2，10，2）
# # 伪数据
# n_data = torch.ones(100, 2)         # 数据的基本形态
# x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
# y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
# x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
# y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )
# # 注意 x, y 数据的数据形式是一定要像下面一样
# x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
# y = torch.cat((y0, y1),).type(torch.LongTensor)    # LongTensor = 64-bit integer
#
# torch.normal()函数：返回一个张量，张量里面的随机数是从相互独立的正态分布中随机生成的。
# 举例：torch.normal(means=torch.arange(1, 11), std=torch.arange(1, 0, -0.1))
#  1.5104   #是从均值为1，标准差为1的正态分布中随机生成的
#  1.6955   #是从均值为2，标准差为0.9的正态分布中随机生成的
#  2.4895   ...
#  4.9185
#  4.9895
#  6.9155
#  7.3683
#  8.1836
#  8.7164
#  9.8916
#
#
# # torch.cat((A,B),dim)函数：同numpy.concatnate((A,B)...), axis)
# # 对于参数dim,axis的理解：dim或axis=i时，操作的是从外往里数第i个中括号。
# # 举例：
# a=np.ones((2,3,4))
# print(a)
# # [[[1. 1. 1. 1.]
# #   [1. 1. 1. 1.]
# #   [1. 1. 1. 1.]]
# #  [[1. 1. 1. 1.]
# #   [1. 1. 1. 1.]
# #   [1. 1. 1. 1.]]]
# print(a.sum(axis=0))
# # [[2. 2. 2. 2.]
# #  [2. 2. 2. 2.]
# #  [2. 2. 2. 2.]]
# print(a.sum(axis=1))
# # [[3. 3. 3. 3.]
# #  [3. 3. 3. 3.]]
# print(a.sum(axis=2))
# # [[4. 4. 4.]
# #  [4. 4. 4.]]
#
#
# # 搭建神经网络方法2：
# net2 = torch.nn.Sequential(
#     torch.nn.Linear(1, 10),     # a0到z1
#     torch.nn.ReLU(),            # z1到a1
#     torch.nn.Linear(10, 1)      # a1到z2
# )torch.nn.Sequential
#
#
# # 保存提取：当神经网络训练完成时，需要保存起来；当用于做实验或者继续训练时再提取出来。
# # 保存，再for循环训练完成后：
# torch.save(net,'net.pkl')# 保存整张神经网络
# torch.save(net1.state_dict(), 'net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)
#
# # 提取：1.全部提取
# net2 = torch.load('net.pkl')    # 提取整张神经网络
# prediction = net2(x)            # 用神经网络做预测
# 2.提取参数放到新网络中
# net3 = torch.nn.Sequential(
#         torch.nn.Linear(1, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 1)
#     )
# # 将保存的参数复制到 net3
# net3.load_state_dict(torch.load('net_params.pkl'))
# prediction = net3(x)
#
#
# # 批训练
# import torch.utils.data as Data
#
# BATCH_SIZE=5                        # 定义batch-size的大小，不整除也无所谓，自动训练剩下的那些数据
# x = torch.linspace(1, 10, 10)       # x data (torch tensor)
# y = torch.linspace(10, 1, 10)       # y data (torch tensor)
#
# torch_dataset = Data.TensorDataset(x,y)# 生成训练数据是x，实际值是y的数据集
# loader = Data.DataLoader(
#     dataset=torch_dataset,      # torch TensorDataset format
#     batch_size=BATCH_SIZE,      # mini batch size
#     shuffle=True,               # 要不要打乱数据 (打乱比较好)
# )# 把 dataset 放入 DataLoader
# for epoch in range(3):   # 训练全部数据 3 次
#     for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
#         # 假设这里就是你训练的地方...
#         # 打出来一些数据
#         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
#               batch_x.numpy(), '| batch y: ', batch_y.numpy())
# # 关于enumerate()函数：
# # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
# seq = ['one', 'two', 'three']
# for i, element in enumerate(seq):
#     print(i, element)
#
#
# # 卷积神经网络（GPU版在训练过程的variable数据，测试过程的variable数据，实例化神经网络，计算预测的位置加cuda）
# import torch
# import torch.nn as nn
# import torch.utils.data as Data
# import torchvision
# from torch.autograd import Variable
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
#
# # Hyper Parameters
# EPOCH = 10           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
# BATCH_SIZE = 64
# LR = 0.001          # 学习率
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Mnist数据集
# train_data = torchvision.datasets.MNIST(
#     root='./mnist/',    # 下载到或者提取数据集的位置
#     train=True,  # true训练，false测试
#     transform=torchvision.transforms.ToTensor(),
#     # transform参数：转换 PIL.Image（像素数据）或者numpy.ndarray成torch.FloatTensor (C x H x W)形式
#     # 训练的时候（0-255）正规化到 [0.0, 1.0] 区间
#     download=True,          # 没下载就下载, 下载了就不用再下了
# )
# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False,
#                                        transform=torchvision.transforms.ToTensor())
#
# # # 打印出数据集的一张图片
# # print(train_data.data.size())
# # print(train_data.targets.size())
# # plt.imshow(train_data.data[0].numpy(),cmap='gray')
# # plt.show()
#
# #装载数据
# train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
# # shuffle参数：是否打乱
# test_loader=Data.DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=False)
#
# # 建立卷积神经网络
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
#             nn.Conv2d(
#                 in_channels=1,      # 输入通道数
#                 out_channels=16,    # 输出通道数，也是卷积核个数
#                 kernel_size=5,      # 卷积核大小，5*5
#                 stride=1,           # 卷积步长
#                 padding=2,          # 补白
#             ),      # output shape (16, 28, 28)
#             nn.ReLU(),    # 激活函数
#             nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
#         )
#         self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
#             nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
#             nn.ReLU(),  # activation
#             nn.MaxPool2d(2),  # output shape (32, 7, 7)
#         )
#         self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         # view的作用为展平后输入全连接层。第二个卷积层输出的形状为(batch_size, 32, 7, 7)。
#         # 展平多维的卷积图成 (batch_size, 32 * 7 * 7),自动考虑batchsize。
#         x = self.out(x)
#         return x
#
# # 实例化神经网络
# cnn=CNN().to(device)           # GPU版在这一行加cuda()
# # print(cnn)
# optimizer=torch.optim.Adam(cnn.parameters(), lr=LR)
# loss_func = nn.CrossEntropyLoss()
#
# # 训练过程
# def train(epoch):
#     for step, (b_x, b_y) in enumerate(train_loader):    # 分配 batch data, normalize x when iterate train_loader
#         b_x,b_y=Variable(b_x).to(device),Variable(b_y).to(device)             # GPU版在这句话的数据位置加cuda
#         output = cnn(b_x)               # cnn output
#         loss = loss_func(output, b_y)   # cross entropy loss
#         optimizer.zero_grad()           # clear gradients for this training step
#         loss.backward()                 # backpropagation, compute gradients
#         optimizer.step()
#         if step%200==0:
#             print('Train Epoch:{}[{}/{}({:.0f}%)]'.format(epoch,step*len(b_x),len(train_loader.dataset),100.*step/len(train_loader)))
#             print('Loss:{:.6f}'.format(loss.data))
#             print('')
#
# # 测试过程
# def test():
#     test_loss=0
#     correct=0
#     with torch.no_grad():
#         for x,y in test_loader:
#             x,y=Variable(x).to(device),Variable(y).to(device)     # GPU版在这句话的数据位置加cuda
#             output=cnn(x)
#             test_loss+=loss_func(output,y).data
#             pred=output.data.max(axis=1,keepdim=True)[1].to(device)    # GPU版在这句话加cuda（）
#             # pytorch的max函数不只返回最大值，还返回最大值的索引值。
#             # torch.max()[0]， 只返回最大值的每个数
#             # troch.max()[1]， 只返回最大值的每个索引
#             # torch.max()[1].data 只返回variable中的数据部分（去掉Variable containing:）
#             # torch.max()[1].data.numpy() 把数据转化成numpy ndarry
#             # torch.max()[1].data.numpy().squeeze() 把数据条目中维度为1 的删除掉
#             correct+=pred.eq(y.data.view_as(pred)).sum()
#             # eq函数：比较相等。返回一个相等元素为1，不等元素为0的张量。
#             # sum计算个数。
#             # view_as函数：将张量的形状变成指定的形状。
#         test_loss/=len(test_loader.dataset)
#         print('Test:Loss:{:.4f}'.format(test_loss))
#         print('Accuracy:{}/{}({:.0f}%)'.format(correct,len(test_loader.dataset),
#                                                100.*correct/len(test_loader.dataset)))
#         print('')
#
#
# for epoch in range(1,51):
#     train(epoch)
#     test()
# #另：因为gpu上的数据无法用matplotlib，numpy等进行可视化，所以需要把gpu上的数据，放回cpu。方法：
# pred_y=pred_y.cpu()
#
#
# # dropout正则化
# import matplotlib.pyplot as plt
#
# torch.manual_seed(1)    # reproducible
#
# N_SAMPLES = 20
# N_HIDDEN = 300
#
# # training data
# x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
# y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
#
# # test data
# test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
# test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
#
# # # show data
# # plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
# # plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
# # plt.legend(loc='upper left')
# # plt.ylim((-2.5, 2.5))
# # plt.show()
#
# class Net_ofit(torch.nn.Module):
#     def __init__(self, n_hidden):
#         super(Net_ofit, self).__init__()
#         self.hidden1 = torch.nn.Linear(1, n_hidden)
#         self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
#         self.hidden3 = torch.nn.Linear(n_hidden, 1)
#
#     def forward(self,x):
#         x = F.relu(self.hidden1(x))  # 激励函数(隐藏层的线性值)
#         x = F.relu(self.hidden2(x))  # 输出值
#         x = self.hidden3(x)
#         return x
#
#
# class Net_dropout(torch.nn.Module):
#     def __init__(self, n_hidden):
#         super(Net_dropout, self).__init__()
#         self.hidden1 = torch.nn.Linear(1, n_hidden)
#         self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
#         self.hidden3 = torch.nn.Linear(n_hidden, 1)
#         self.drop = torch.nn.Dropout(0.5)           # dropout这么调，失活率0.5
#
#     def forward(self, x):
#         x = F.relu(self.drop(self.hidden1(x)))      # dropout这么用
#         x = F.relu(self.drop(self.hidden2(x)))
#         x = self.hidden3(x)
#         return x
#
# net1=Net_ofit(N_HIDDEN)
# net2=Net_dropout(N_HIDDEN)
# optimizer_ofit = torch.optim.Adam(net1.parameters(), lr=0.01)
# optimizer_drop = torch.optim.Adam(net2.parameters(), lr=0.01)
# loss_func = torch.nn.MSELoss()
#
# for t in range(500):
#     pred_ofit = net1(x)
#     pred_drop = net2(x)
#
#     loss_ofit = loss_func(pred_ofit, y)
#     loss_drop = loss_func(pred_drop, y)
#
#     optimizer_ofit.zero_grad()
#     optimizer_drop.zero_grad()
#     loss_ofit.backward()
#     loss_drop.backward()
#     optimizer_ofit.step()
#     optimizer_drop.step()
#     if t % 50 == 0:     # 每 50 步画一次图
#         net1.eval()
#         net2.eval()     # 神经网络在测试的时候要关闭dropout，eval()进入测试模式。
#         test_pred_ofit = net1(test_x)
#         test_pred_drop = net2(test_x)
#         plt.scatter(x.data.numpy(),y.data.numpy(),c='magenta', s=50, alpha=0.5)
#         plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5)
#         plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-')
#         plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--')
#         plt.text(0,-1.2,'overfitting loss=%.4f'%loss_func(test_pred_ofit.data,test_y.data))
#         plt.text(0, -1.5, 'drop loss=%.4f'%loss_func(test_pred_drop.data, test_y.data))
#         plt.ylim(-2.5,2.5)
#         plt.pause(0.1)
#         net1.train()
#         net2.train()    # 预测结束，改回训练模式。
#
#
# # 定义自己的数据集，需要写三个魔法方法。加载自定义数据集时要调用Dataset,DataLoader
# import torchvision
# from PIL import Image
# import os
# from torchvision import transforms
# from torch.utils.data import Dataset,DataLoader
#
# batch_size=64
# trans = transforms.Compose([
#         transforms.Resize(224),
#         # resize
#         transforms.RandomHorizontalFlip(),
#         # 水平翻转，默认概率为0.5
#         transforms.ToTensor(),
#         # 因为输入神经网络的数据类型需要为tensor，所以，这里可以设置将数据类型转换。
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         # 这个是imagenet的均值和方差
#     ])
#
# class Mydata(Dataset):
# # 继承Dataset类
#     def __init__(self,root,transforms=None):
#     # 这个魔法方法通常设定为将类实例化为对象时，需要输入数据的相对路径。
#         self.root_dir = root
#         self.transform=transforms
#
#     def __getitem__(self, idx):
#     # 这个魔法方法在设定数据的索引方法，返回数据和标签（顺序不要弄错）。
#         a,b=0,0
#         label_list=os.listdir(self.root_dir)                # os.listdir()方法：用于返回指定的文件夹包含的文件或文件夹的名字的列表。
#         for i in label_list:
#             l=os.listdir(os.path.join(self.root_dir, i))    # os.path.join()方法：将多个路径组合后返回
#             a+=len(l)
#             if idx>a:
#                 b=a
#             else:
#                 img_name=l[idx-b-1]
#                 img_item_path = os.path.join(self.root_dir,i,img_name)
#                 img = Image.open(img_item_path)
#                 if self.transform is not None:
#                     img=self.transform(img)
#                 return img,label_list.index(i)  # 将数据转换成tensor的数据类型后返回，注意数据在前面，标签在后面
#
#     def __len__(self):
#     # 这个魔法方法返回全体数据的大小
#         a=0
#         for i in os.listdir(self.root_dir):
#             a+=len(os.listdir(os.path.join(self.root_dir, i)))
#         return a
#
# train_data=Mydata('data/train',transforms=trans)       # 加载完成后，train_data为一个由数据总数个的元组组成的可迭代对象，元组数据为(tensor，标签)
# test_data=Mydata('data/test',transforms=trans)
# train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,pin_memory=True)
# test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,pin_memory=True)
# DataLoader类的目的是为了将数据打包好，分批次的送进神经网络。shuffle参数为是否打乱数据。
#
#
# # 可视化工具，tensorboard的使用
# from torch.utils.tensorboard import SummaryWriter
# from PIL import Image
# from torchvision import transforms
#
# writer=SummaryWriter('tensorboard')     # 传入写入可视化的文件夹
# for i in range(100):
#     writer.add_scalar(tag='y=x', scalar_value=i, global_step=i)
#     # tag参数：标题
#     # scalar_value参数：想要可视化的值，一般是loss，对应可视化后的y轴。
#     # global_step (int): 默认为None，训练的步数。对应可视化后的x轴。
#     # 想要打开tensorboard文件时，终端输入tensorboard --logdir=文件路径 --port=端口号
#     # （设置端口号避免和其他人一样）
#
# trans = transforms.Compose([transforms.ToTensor()])
# img=Image.open('data/train/airplane/batch_1_num_29.jpg')
# img=trans(img)
# writer.add_image(tag='test',img_tensor=img)
# # tag(string): 标题
# # img_tensor(torch.Tensor, numpy.array, or string / blobname): 图片数据为tensor或array形式
# # global_step(int): Global step value to record
# # 想要查看时，同样在终端输入tensorboard --logdir=文件路径 --port=6006
# # （设置端口号避免和其他人一样）
# writer.close()          # 写入完成后要加这句话。
#
#
# # 调整学习率的方法
# from torch.optim import lr_scheduler as lr_s
# # 第一种：等间隔调整学习率
# scheduler = lr_s.StepLR(optimizer, step_size=30, gamma=0.1)
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()
# # 假设初始学习率lr = 0.05，则：
# # lr = 0.05     if epoch < 30
# # lr = 0.005    if 30 <= epoch < 60
# # lr = 0.0005   if 60 <= epoch < 90
# # scheduler.step()这句话为调整学习率。
#
# # 第二种：设定间隔调整学习率
# scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()
# # 假设初始学习率lr = 0.05，则：
# # lr = 0.05     if epoch < 30
# # lr = 0.005    if 30 <= epoch < 80
# # lr = 0.0005   if epoch >= 80
#
# # 第三种：lr=lr*gamma**epoch
# scheduler=lr_s.ExponentialLR(optimizer,gamma=0.1)
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()
#
#
# # 显示部分测试集数据及预测
# def visualize_model(model):
#     model.eval()
#     plt.figure()
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             outputs = model(inputs)
#             preds=outputs.data.max(axis=1)[1]
#             for i,preds in enumerate(preds[0:6]):
#                 plt.subplot(3, 2, i+1)
#                 plt.imshow(np.transpose(inputs[i].numpy(),(1,2,0)))
#                 # np.transpose()函数：因为plt只能显示形状为（h,w,3）的图片，而pytorch的tensor为（3，h,w）
#                 # 所以结合np.transpose()函数改变序列才可以显示。
#                 plt.title("prediction:{}\naccuracy:{}".format(test_data.classes[preds],test_data.classes[labels[i]]))
#             break
#         plt.show()
#
# visualize_model(model)
#
#
# # 迁移学习（以resnet18为例）
# # 迁移学习主要有以下两种手段：
# # 1模型微调：冻结预训练模型靠近输入的部分卷积层，训练剩下的卷积层和全连接层。（用于大数据）
# # 2特征提取：冻结预训练模型的全部卷积层，只训练自己定制的全连接层。（用于少量数据），优化器中只传入全连接层的参数。
# from torchvision import models
# import torch.nn as nn
# import torch
# # 在冻结部分卷积层前，先看一下网络结构，确定网络的前几个child不训练。children()这个方法，就是将网络的模块化。
# # model=models.resnet18(pretrained=True)
# # list1=list(model.children())
# # for i in list1:
# #     print(i)
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.res_head = models.resnet18(pretrained=True)
#         for i,child in enumerate(self.res_head.children()):
#             # 冻结网络中前7个child的参数
#             if i<6:
#                 for param in child.parameters():
#                     param.requires_grad = False
#             else:
#                 break
#         # 修改全连接层
#         self.res_head.fc = nn.Linear(512, 100)
#
#     def forward(self, x):
#         x = self.res_head(x)
#         return x
#
# model=Net()
# print(model.res_head.layer2[0].conv1.weight.requires_grad)
# print(model.res_head.layer3[0].conv1.weight.requires_grad)
# # 运用filter()方法和lambda表达式，筛选出requires_grad为True的参数进行训练。
# optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
# # filter(function, iterable)函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
# # 接收两个参数，第一个为判断函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False。
# # 最后将返回 True 的元素装进一个filter对象中，可用list()转换。
