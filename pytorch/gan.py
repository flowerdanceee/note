# coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import cv2
from torchvision.utils import save_image
import os

# 创建文件夹
if not os.path.exists('./img'):
    os.mkdir('./img')

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

def to_img(x):
    out = x.view(-1,224, 224)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out


batch_size = 32
num_epoch = 100
z_dimension = 100
# 图像预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
])

class Mydata(Dataset):
    def __init__(self,root,transforms=None):
        self.root_dir = root
        self.transform=transforms

    def __getitem__(self, idx):
        label_list=os.listdir(self.root_dir)                # os.listdir()方法：用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        img_name=label_list[idx]
        img_item_path = os.path.join(self.root_dir,img_name)
        img = cv2.imread(img_item_path)
        img=img[:,:,0]
        if self.transform is not None:
            img=self.transform(img)
        return img,1  # 将数据转换成tensor的数据类型后返回，注意数据在前面，标签在后面

    def __len__(self):
        a=os.listdir(self.root_dir)
        return len(a)

data=Mydata('C_front',img_transform)
dataloader = DataLoader(dataset=data,batch_size=batch_size,shuffle=True,pin_memory=True)


# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,3,(3,3),(1,1),(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x=self.conv(x)
        x=x.view(-1,7*7*512)
        x=self.fc(x)
        return x


# ###### 定义生成器 Generator #####
# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 1024),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(1024, 4096),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(4096, 4096),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(4096, 7*7*512),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(7*7*512, 7 * 7 * 512),  # 线性变换
            nn.ReLU(True),  # relu激活
        )
        self.convTranspose=nn.Sequential(
            nn.ConvTranspose2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, (5,5), stride = (2,2), padding = (2,2), output_padding = (1,1)),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, (5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, (5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, (5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, (5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(3),
            nn.ConvTranspose2d(3, 3, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(3),
            nn.ConvTranspose2d(3, 3, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(3),
            nn.ConvTranspose2d(3, 1, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            nn.ConvTranspose2d(1, 1, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x = self.gen(x)
        x=x.view(-1,512,7,7)
        x=self.convTranspose(x)
        x=255*torch.sigmoid(x)
        return x


# 创建对象
D = Discriminator().to(device)
G = Generator().to(device)
# z = Variable(torch.randn(100)).to(device)  # 随机生成一些噪声
# fake_img = G(z).detach()


# 首先需要定义loss的度量方式  （二分类的交叉熵）
# 其次定义 优化函数,优化函数的学习率为0.0003
criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
d_optimizer = torch.optim.SGD(D.parameters(), lr=0.01)
g_optimizer = torch.optim.SGD(G.parameters(), lr=0.01)

# ##########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        real_img = Variable(img).to(device)  # 将tensor变成Variable放入计算图中
        real_label = Variable(torch.ones(num_img)).to(device)  # 定义真实的图片label为1
        fake_label = Variable(torch.zeros(num_img)).to(device) # 定义假的图片的label为0
        real_label = torch.unsqueeze(real_label, 1)
        fake_label = torch.unsqueeze(fake_label, 1)


        # ########判别器训练train#####################
        # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
        # 计算真实图片的损失
        real_out = D(real_img)  # 将真实图片放入判别器中
        real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好
        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss

        # 计算假的图片的损失
          # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
        z = Variable(torch.randn(num_img, z_dimension)).to(device)  # 随机生成一些噪声
        fake_img = G(z)
        fake_out = D(fake_img.detach())  # 判别器判断假的图片，
        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss

        # 损失函数和优化
        d_loss = (d_loss_real + d_loss_fake) / 2  # 损失包括判真损失和判假损失
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数

        # ==================训练生成器============================
        # ###############################生成网络的训练###############################
        # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
        # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
        # 反向传播更新的参数是生成网络里面的参数，
        # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
        # 这样就达到了对抗的目的
        # 计算假的图片的损失
        fake_out = D(fake_img)
        g_loss = criterion(fake_out, real_label)  # 判别器输出结果越靠近1，生成器的效果越好。
        # 所以用生成器的输出通过判别器得到的结果和1来计算生成器的损失。

        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数


        # 打印中间的损失
        # if (i + 1) % 100 == 0:
        print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f}'
              'D real: {:.6f},D fake: {:.6f}'.format(
            epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
            real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
        ))
        #     save_image(real_images, './img/real_images.png')
        fake_images = to_img(fake_img.cpu().data)

    if not os.path.exists('./img/fake_images-{}'.format(epoch + 1)):
        os.mkdir('./img/fake_images-{}'.format(epoch + 1))
    for i,img in enumerate(fake_images):
        img=img.numpy()
        cv2.imwrite( './img/fake_images-{}/{}.png'.format(epoch + 1,i),img)

# 保存模型
torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')