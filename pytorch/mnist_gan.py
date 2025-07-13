import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# 定义生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 28 * 28)

    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        x = nn.LeakyReLU(0.2)(self.fc2(x))
        x = nn.Tanh()(self.fc3(x))
        return x


# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        x = nn.LeakyReLU(0.2)(self.fc2(x))
        x = nn.Sigmoid()(self.fc3(x))
        return x


# 定义训练过程
def train_gan(generator, discriminator, dataloader, device, num_epochs=50, lr=0.0002, betas=(0.5, 0.999)):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    fixed_noise = torch.randn(64, 100, device=device)

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # 训练判别器
            optimizer_d.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # 真实图像的损失
            real_outputs = discriminator(real_images)
            real_loss = criterion(real_outputs, real_labels)

            # 生成图像的损失
            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)

            # 总损失
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

            # 打印训练过程中的损失值
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], D_loss: {:.4f}, G_loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item()))
