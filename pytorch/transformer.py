import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    # initializers
    def __init__(self, c=4):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(6, 2 * c, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(2 * c)
        self.deconv1_2 = nn.ConvTranspose2d(2, 2 * c, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(2 * c)
        self.deconv2 = nn.ConvTranspose2d(4 * c, 2 * c, 3, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(2 * c)
        self.deconv3 = nn.ConvTranspose2d(2 * c, c, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(c)
        self.deconv4 = nn.ConvTranspose2d(c, 1, 4, 2, 1)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        # 7
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        # 14
        x = F.tanh(self.deconv4(x))
        # 28

        return x


# Enable CUDA device if available
device = torch.device("cpu")

generator = Generator().to(device)
# 打印模型参数数量
total_params = sum(p.numel() for p in generator.parameters())
# total_params: 104*c*c+357*c+1
print(f"Total number of parameters: {total_params}")
