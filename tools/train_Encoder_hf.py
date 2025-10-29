import os
import sys

# 获取当前文件所在目录的上一级的上一层目录
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

# 拼接相对路径并添加到 sys.path
sys.path.append(os.path.join(base_dir, "MST/simulation/train_code"))
sys.path.append(os.path.join(base_dir, "DiffBIR"))
sys.path.append(base_dir)

# 将 base_dir 设为工作目录
os.chdir(base_dir)

import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from MST.simulation.train_code.architecture import *
from MST.simulation.train_code.utils import *

# 启用cuDNN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from DiffBIR.utils.common import (wavelet_decomposition,
                                  wavelet_decomposition_msi)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.max_pool = nn.AdaptiveMaxPool2d((2, 2))

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 2, bias=False)

        self.SiLU = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.SiLU(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.SiLU(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class DoubleConvWoBN(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.double_conv(x) + self.res_conv(x)


class ChannelEncoder(nn.Module):
    def __init__(self):
        super(ChannelEncoder, self).__init__()
        self.conv1 = DoubleConvWoBN(in_channels=28, out_channels=21)
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            DoubleConvWoBN(in_channels=21, out_channels=9),
        )
        self.conv3 = DoubleConvWoBN(in_channels=9, out_channels=3)
        self.conv_out = DoubleConvWoBN(in_channels=3, out_channels=3)
        self.conv_res = nn.Sequential(
            DoubleConvWoBN(in_channels=28, out_channels=3),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        )

        self.ca1 = ChannelAttention(28, 2)
        self.ca2 = ChannelAttention(21, 2)
        self.ca3 = ChannelAttention(9, 2)
        self.ca_res = ChannelAttention(28, 2)

    def forward(self, x):

        res = self.conv_res(x * self.ca_res(x))

        x = x * self.ca1(x)
        x = self.conv1(x)

        x = x * self.ca2(x)
        x = self.conv2(x)

        x = x * self.ca3(x)
        x = self.conv3(x)

        x = self.conv_out(x + res)
        return x


class ChannelDecoder(nn.Module):
    def __init__(self):
        super(ChannelDecoder, self).__init__()
        self.conv1 = DoubleConvWoBN(in_channels=3, out_channels=9)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=21, kernel_size=2, stride=2),
            nn.SiLU(inplace=True),
            DoubleConvWoBN(in_channels=21, out_channels=21),

        )
        self.conv3 = DoubleConvWoBN(in_channels=21, out_channels=28)
        self.conv_out = DoubleConvWoBN(in_channels=28, out_channels=28)
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=2),
            DoubleConvWoBN(in_channels=3, out_channels=28),
        )

        self.ca3 = ChannelAttention(28, 2)
        self.ca2 = ChannelAttention(21, 2)
        self.ca1 = ChannelAttention(9, 2)

        self.ca_res = ChannelAttention(28, 2)

    def forward(self, x):

        res = self.conv_res(x)
        res = res * self.ca_res(res)

        x = self.conv1(x)
        x = x * self.ca1(x)

        x = self.conv2(x)
        x = x * self.ca2(x)

        x = self.conv3(x)
        x = x * self.ca3(x)

        x = self.conv_out(x + res)

        return x


class ChannelVAE(nn.Module):
    def __init__(self):
        super(ChannelVAE, self).__init__()
        self.encoder = ChannelEncoder()
        self.decoder = ChannelDecoder()

    def forward(self, x):
        en = self.encoder(x)
        return self.decoder(en)



# 读取命令行参数
if len(sys.argv) < 2:
    print("Usage: python script.py <device>")
    sys.exit(1)

cuda_device = sys.argv[1]  # 获取命令行参数中的设备信息
print(cuda_device)


# 定义自定义数据集类
class MSIDataset(Dataset):
    def __init__(self, list_file_path):
        self.data_list = []
        with open(list_file_path, "r") as f:
            for line in f:
                file_record = eval(line.strip())  # 将字典形式的字符串转换为字典
                self.data_list.append(file_record)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        record = self.data_list[idx]

        # 使用np.load读取npy文件，并转换为PyTorch张量
        gt = torch.from_numpy(np.load(record['gt'])).float()
        msi = torch.from_numpy(np.load(record['msi'])).float()

        return gt, msi

# list.txt 文件路径
list_file_path = "./datasets/MSIdatasets/list.txt"

# 实例化数据集
train_data = MSIDataset(list_file_path)


def torch_psnr(img, ref):  # input [batch_size, channels, height, width]
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[1]  # number of channels
    psnr = 0
    for i in range(nC):
        # calculate MSE for each channel
        mse = torch.mean((img[:, i, :, :] - ref[:, i, :, :]) ** 2, dim=(1, 2))
        psnr += 10 * torch.log10((255 * 255) / mse)
    return torch.mean(psnr) / nC


def main():

    # 创建SummaryWriter来写入TensorBoard日志文件
    writer = SummaryWriter()

    # 创建DiffSCI模型并移至GPU
    ChannelVAE_model = ChannelVAE().to(cuda_device)

    def print_model_parameters(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.numel())

    ChannelVAE_model = torch.load("weights/model_SeVAE_hf_endecoder_c21_bu2_c9_DoubleConvWoBN_resca_silu_2024-09-04_psnr32.7162.pt", map_location="cpu").to(cuda_device)
    print("Total parameters:")
    print_model_parameters(ChannelVAE_model)

    print("Total number of parameters: {:,}".format(sum(p.numel() for p in ChannelVAE_model.parameters() if p.requires_grad)))

    # 定义均方误差损失函数
    criterion = nn.MSELoss()

    # 定义DataLoader，设置batch_size合适的大小
    batch_size = 24
    num_constraint_epochs = 24
    warmup_steps = 5
    lambda_cons = 0.5

    num_epochs = 120
    avg_loss = 0

    dataloader = DataLoader(train_data, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=2)

    # 定义优化器和学习率调度器
    optimizer = optim.Adam(ChannelVAE_model.parameters(), lr=0.0005)

    def get_scheduler(optimizer, dst_rate, num_epochs):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # 在warmup阶段，学习率逐渐增加到初始学习率
                return ((current_step + 1) / warmup_steps+1)/2
            elif current_step < num_constraint_epochs:
                # 在warmup之后，保持初始学习率不变
                return 1
            else:
                return np.power(
                    dst_rate,
                    (current_step + 1 - num_constraint_epochs) / (num_epochs - num_constraint_epochs),
                )

        # 创建学习率调度器
        scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler

    # 创建学习率调度器
    scheduler = get_scheduler(optimizer=optimizer, dst_rate=0.5, num_epochs=num_epochs)

    # 循环迭代训练过程
    for epoch in range(num_epochs):
        total_loss = 0
        total_psnr = 0
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}",
            position=0,
            leave=True,
        )

        for batch_idx, (inputs_gt, inputs_msi) in progress_bar:
            optimizer.zero_grad()
            inputs_gt = inputs_gt.to(cuda_device)  # 获取一个batch的输入数据
            inputs_msi = inputs_msi.to(cuda_device)  # 获取一个batch的输入数据

            inputs_msi_hf, inputs_msi_lf = wavelet_decomposition_msi(inputs_msi, 3)
            inputs_gt_hf = inputs_gt - inputs_msi_lf


            # 根据epoch阶段选择不同的损失函数
            if epoch < num_constraint_epochs:
                # 将预测的光谱数据合成为RGB图像

                pred_gt_hf = ChannelVAE_model.encoder(inputs_gt_hf)
                restore_msi = ChannelVAE_model(inputs_msi_hf) + inputs_msi_lf
                restore_gt = ChannelVAE_model.decoder(pred_gt_hf) + inputs_msi_lf

                loss = (
                    criterion(
                        pred_gt_hf,
                        F.interpolate(
                            wavelet_decomposition(torch.stack((inputs_gt_hf[:, 23], inputs_gt_hf[:, 15], inputs_gt_hf[:, 6]), dim=1), 3)[0],
                            size=(inputs_gt_hf.shape[2] * 2, inputs_gt_hf.shape[2] * 2),
                            mode="bilinear",
                            align_corners=False,
                        ),
                    )
                    * np.min([(num_constraint_epochs - epoch) / 8, 1])
                    * lambda_cons
                    + (criterion(restore_msi, inputs_msi)+ criterion(restore_gt, inputs_gt)) / 2
                )
                del pred_gt_hf
            else:
                restore_msi = ChannelVAE_model(inputs_msi_hf) + inputs_msi_lf
                restore_gt = ChannelVAE_model(inputs_gt_hf) + inputs_msi_lf

                loss = (criterion(restore_msi, inputs_msi) + criterion(restore_gt, inputs_gt)) / 2

            del restore_msi, inputs_msi_hf, inputs_msi_lf, inputs_gt_hf

            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()

            with torch.no_grad():
                total_psnr += torch_psnr(restore_gt, inputs_gt)

            # 实时显示损失
            avg_loss = total_loss / (batch_idx + 1)
            avg_psnr = total_psnr / (batch_idx + 1)
            progress_bar.set_postfix({"loss": avg_loss, "psnr": avg_psnr.item(), "lr": scheduler.get_last_lr()[0]}, refresh=True)

        # 在训练过程中将损失数据写入TensorBoard
        writer.add_scalar("train/Loss", avg_loss, epoch)
        writer.add_scalar("train/PSNR", avg_psnr, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # 触发学习率调度器
        scheduler.step()

    # 设置模型为评估模式
    ChannelVAE_model.eval()

    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluation", position=0, leave=True)

    with torch.no_grad():
        total_psnr= 0
        for batch_idx, (inputs_gt, inputs_msi) in progress_bar:
            optimizer.zero_grad()
            inputs_gt = inputs_gt.to(cuda_device)  # 获取一个batch的输入数据
            inputs_msi = inputs_msi.to(cuda_device)  # 获取一个batch的输入数据

            inputs_msi_hf, inputs_msi_lf = wavelet_decomposition_msi(inputs_msi, 3)
            inputs_gt_hf = inputs_gt - inputs_msi_lf

            restore_gt = ChannelVAE_model(inputs_gt_hf) + inputs_msi_lf
            loss = criterion(restore_gt, inputs_gt)

            total_loss += loss.item()
            total_psnr += torch_psnr(restore_gt, inputs_gt)

            # 实时显示损失
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({"loss": avg_loss}, refresh=True)

    eval_loss = total_loss / len(dataloader)
    eval_psnr = total_psnr / len(dataloader)
    print(f"Eventually Evaluation Loss: {eval_loss}")

    # 创建一个带有日期和平均损失的文件名
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # 获取当前日期
    filename = f"./weights/model_SeVAE_hf3_endecoder_c21_bu2_c9_DConvWoBN_resca_silu_{current_date}_psnr{eval_psnr:.4f}.pt"

    # 保存模型
    torch.save(ChannelVAE_model, filename)
    print(f"Saved to '{filename}'...")

    # 关闭SummaryWriter
    writer.close()


while True:
    main()
