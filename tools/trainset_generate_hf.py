# --- Setup base dir & env, then import deps (condensed) ---

# Stdlib
import os, sys
from pathlib import Path


def get_basedir(up: int = 2) -> Path:
    """Return dir `up` levels above running .py/.ipynb."""
    try:
        p = Path(__file__).resolve()  # .py
    except NameError:
        try:
            import ipynbname  # notebook

            p = Path(ipynbname.path()).resolve()
        except Exception:
            p = (Path.cwd() / "_dummy").resolve()  # fallback
    for _ in range(up):
        p = p.parent
    return p


# only initialize BASE_DIR once
BASE_DIR = globals().get("BASE_DIR")
if not isinstance(BASE_DIR, Path) or not BASE_DIR.exists():
    BASE_DIR = get_basedir()

sys.path[:0] = [
    str(BASE_DIR),
    str(BASE_DIR / "packages"),
    str(BASE_DIR / "packages" / "DiffBIR"),
    str(BASE_DIR / "packages" / "MST" / "simulation" / "train_code"),
]
os.chdir(BASE_DIR)
print(f"Set BASE = {BASE_DIR}")
os.environ.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"})

import h5py
import scipy.io as sio
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from packages.DiffBIR.utils.common import (
    wavelet_decomposition_msi,
)
from packages.MST.simulation.train_code.architecture import model_generator
from packages.MST.simulation.train_code.utils import (
    LoadTraining,
    init_mask,
    shuffle_crop,
    init_meas,
    LoadTest,
)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

train_size = 16400
batch_size = 64


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


ChannelVAE_model = torch.load(
    str(BASE_DIR)
    + "/weights/model_SeVAE_hf3_endecoder_c21_bu2_c9_DConvWoBN_resca_silu_2024-09-05_psnr49.5199.pt",
    map_location="cpu",
    weights_only=False,
)
ChannelVAE_model = ChannelVAE_model.cuda().eval()


# DAUHST_model = model_generator("mst_l", str(BASE_DIR)+"/packages/MST/simulation/test_code/model_zoo/mst/mst_l.pth").cuda()
DAUHST_model = model_generator(
    "dauhst_3stg",
    str(BASE_DIR)
    + "/packages/MST/simulation/test_code/model_zoo/dauhst_3stg/dauhst_3stg.pth",
).cuda()
DAUHST_model.eval()

# 创建文件夹用于保存图片
os.makedirs(str(BASE_DIR) + "/datasets/Images_Diff_hf", exist_ok=True)
os.makedirs(str(BASE_DIR) + "/datasets/Images_Diff_hf/gt", exist_ok=True)
os.makedirs(str(BASE_DIR) + "/datasets/Images_Diff_hf/base", exist_ok=True)


def compute_weights(source_lambdas, target_lambdas):
    dist_matrix = torch.abs(target_lambdas.unsqueeze(1) - source_lambdas.unsqueeze(0))
    weights = torch.exp(-dist_matrix) ** 4
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights


def resample_channels(data_tensor, source_range, target_range, cuda_device):
    source_lambdas = torch.linspace(
        source_range[0], source_range[1], source_range[2]
    ).to(cuda_device)
    target_lambdas = torch.linspace(
        target_range[0], target_range[1], target_range[2]
    ).to(cuda_device)
    weights = compute_weights(source_lambdas, target_lambdas)
    data_resampled = torch.einsum("ij,jhw->ihw", weights, data_tensor)
    return data_resampled


def load_ICLV_dataset(data_dir, cuda_device):
    images = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".mat"):
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing file: {file_path}")

            with h5py.File(file_path, "r") as f:
                data = f["rad"][:]

            data_tensor = (
                torch.tensor(data).flip(dims=[1]).to(cuda_device).float().unsqueeze(0)
            )

            # 计算上分位数
            upper_quantile = np.percentile(data_tensor.cpu().numpy(), 99.5)
            data_tensor = data_tensor / (upper_quantile * 1.4)

            source_range = [400, 700, 31]
            target_range = [450, 650, 28]
            data_tensor_resized = resample_channels(
                data_tensor.squeeze(0), source_range, target_range, cuda_device
            ).unsqueeze(0)
            data_tensor_resized = F.interpolate(
                data_tensor_resized, size=(1024, 1024), mode="bilinear"
            )
            data_tensor_resized[data_tensor_resized < 0.0] = 0.0
            data_tensor_resized[data_tensor_resized > 1.0] = 1.0

            images.append(data_tensor_resized[0].permute(1, 2, 0).cpu().numpy())

    return images


def load_Harvard_dataset(data_dir, cuda_device):
    images = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".mat"):
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing file: {file_path}")

            # 加载数据并进行初始处理
            data = sio.loadmat(file_path)["ref"]
            data_tensor = (
                torch.tensor(data).permute(2, 0, 1).to(cuda_device).float().unsqueeze(0)
            )

            # 计算上分位数
            upper_quantile = np.percentile(data_tensor.cpu().numpy(), 99.5)
            data_tensor = data_tensor / (upper_quantile * 1.4)

            source_range = [420, 720, 31]
            target_range = [450, 650, 28]
            data_tensor_resized = resample_channels(
                data_tensor.squeeze(0), source_range, target_range, cuda_device
            ).unsqueeze(0)
            data_tensor_resized = F.interpolate(
                data_tensor_resized, size=(1024, 1024), mode="bilinear"
            )
            data_tensor_resized[data_tensor_resized < 0.0] = 0.0
            data_tensor_resized[data_tensor_resized > 1.0] = 1.0

            images.append(data_tensor_resized[0].permute(1, 2, 0).cpu().numpy())

    return images


def load_NTIRE2022_dataset(data_dir, cuda_device):
    images = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".mat"):
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing file: {file_path}")

            with h5py.File(file_path, "r") as f:
                data = f["cube"][:]

            data_tensor = (
                torch.tensor(data).permute(0, 2, 1).to(cuda_device).float().unsqueeze(0)
            )
            # 计算上分位数
            upper_quantile = np.percentile(data_tensor.cpu().numpy(), 99.5)
            data_tensor = data_tensor / (upper_quantile * 1.4)

            # 直方图展示

            source_range = [400, 700, 31]
            target_range = [450, 650, 28]
            data_tensor_resized = resample_channels(
                data_tensor.squeeze(0), source_range, target_range, cuda_device
            ).unsqueeze(0)
            data_tensor_resized = F.interpolate(
                data_tensor_resized, size=(1024, 1024), mode="bilinear"
            )
            data_tensor_resized[data_tensor_resized < 0.0] = 0.0
            data_tensor_resized[data_tensor_resized > 1.0] = 1.0

            images.append(data_tensor_resized[0].permute(1, 2, 0).cpu().numpy())

    return images


def load_HSIGene_dataset(data_dir, cuda_device):
    images = []

    # 遍历目录中的所有.mat文件
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".mat"):
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing file: {file_path}")

            # 加载数据
            data = sio.loadmat(file_path)["data"]
            data = np.array(data).transpose(2, 0, 1)  # 转置为 (channels, height, width)

            # 使用指定的范围选择14个通道
            data_tensor = (
                torch.tensor(data[5 : 5 + 14]).cuda().unsqueeze(0).float() / 255
            )  # 假设目标数据是 14 个通道

            # 创建插值的权重
            original_channels = torch.linspace(0, 1, 14)
            target_channels = torch.linspace(0, 1, 28)

            # 初始化插值后的张量
            interpolated_data = torch.zeros(1, 28, 256, 256).cuda()

            # 对每个目标通道进行插值
            for i in range(28):
                # 查找两个最接近的原始通道
                idx = (
                    np.searchsorted(
                        original_channels.cpu().numpy(),
                        target_channels[i].cpu().numpy(),
                    )
                    - 1
                )
                idx = max(0, min(idx, 12))  # 确保索引在有效范围内

                # 线性插值权重
                if idx < 13:
                    weight_low = 1 - (target_channels[i] - original_channels[idx]) / (
                        original_channels[idx + 1] - original_channels[idx]
                    )
                    weight_high = 1 - weight_low

                    # 插值
                    interpolated_data[0, i] = (
                        data_tensor[0, idx] * weight_low
                        + data_tensor[0, idx + 1] * weight_high
                    )
            interpolated_data = F.interpolate(
                interpolated_data, size=(512, 512), mode="bilinear"
            )
            interpolated_data[interpolated_data < 0.0] = 0.0
            interpolated_data[interpolated_data > 1.0] = 1.0

            # 将插值后的数据放入图像列表
            images.append(interpolated_data[0].permute(1, 2, 0).cpu().numpy())

    return images


def add_noise(input_meas_batch, noise_probabilities):
    """
    为input_meas_batch添加噪声。

    参数:
    - input_meas_batch: 需要添加噪声的张量
    - noise_probabilities: 一个包含四个概率的列表，对应于不加噪声、添加0.001噪声、添加0.004噪声和添加0.01噪声的概率

    返回:
    - 添加噪声后的input_meas_batch
    """
    assert abs(noise_probabilities.sum() - 1) < 1e-4, "噪声概率之和必须为1"

    noise_levels = [0.00005, 0.0004, 0.001, 0.004, 0.01]
    # noise_levels = [0, 0.0001, 0.0003, 0.001, 0.004]
    noise_choice = np.random.choice(len(noise_levels), p=noise_probabilities)
    noise_std = noise_levels[noise_choice]

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, input_meas_batch.shape).astype(
            np.float32
        )
        noise = torch.from_numpy(noise).cuda()
        input_meas_batch += noise

    return input_meas_batch


# 原始数据集路径
harvard_data_dir = (
    str(BASE_DIR) + "/exp/Harvard Statistics of Real-World Hyperspectral Images"
)
ntire2022_data_dir = str(BASE_DIR) + "/exp/NTIRE2022Valid"
iclv_data_dir = str(BASE_DIR) + "/exp/ICLV"
hsi_data_dir = str(BASE_DIR) + "/package/HSIGene/save_uncond/mats"

# 添加数据集 （生成数据集Rebuttal实验用，采集数据集性能会提升，生成数据集不会提升）
# train_set = []
# train_set += load_Harvard_dataset(harvard_data_dir, 'cuda')
# train_set += load_NTIRE2022_dataset(ntire2022_data_dir, "cuda")
# train_set += load_ICLV_dataset(iclv_data_dir, "cuda")
# train_set += load_HSIGene_dataset(hsi_data_dir, "cuda")

train_set = LoadTraining(str(BASE_DIR) + "/packages/MST/datasets/cave_1024_28/")
# mask3d_batch_train, input_mask_train = init_mask(str(BASE_DIR)+"/packages/MST/datasets/TSA_simu_data/", "Phi", batch_size)
mask3d_batch_train, input_mask_train = init_mask(
    str(BASE_DIR) + "/packages/MST/datasets/TSA_simu_data/", "Phi_PhiPhiT", batch_size
)


# 打开文件用于写入图片文件名列表
with open("./datasets/Images_Diff_hf/img_files.txt", "w") as file_list:
    for batch_idx in tqdm(
        range(0, train_size, batch_size), desc="Processing batches", unit="batch"
    ):
        gt_batch = shuffle_crop(train_set, batch_size)
        gt_batch = Variable(gt_batch).cuda().float()

        # 定义噪声添加概率
        # noise_probabilities = np.array([0.65, 0.25, 0.05, 0.03, 0.02])
        noise_probabilities = np.array([0.80, 0.10, 0.050, 0.0375, 0.0125])

        # 添加噪声
        gt_batch_noise = add_noise(gt_batch, noise_probabilities)

        # input_meas_batch = init_meas(gt_batch_noise, mask3d_batch_train, "H")
        input_meas_batch = init_meas(gt_batch_noise, mask3d_batch_train, "Y")

        with torch.no_grad():
            initial_msi = DAUHST_model(input_meas_batch, input_mask_train)

            # 使用自定义的概率生成随机数
            probabilities = [0.5, 0.15, 0.35]
            values = [3, 4, 5]
            random_number = np.random.choice(values, p=probabilities)

            inputs_msi_hf, inputs_msi_lf = wavelet_decomposition_msi(
                initial_msi, random_number
            )
            inputs_gt_hf = gt_batch - inputs_msi_lf

            encoded_base = ChannelVAE_model.encoder(inputs_msi_hf)
            encoded_gt = ChannelVAE_model.encoder(inputs_gt_hf)

        RANGE_MAX = 0.85
        RANGE_MIN = 0.15

        range_channel = torch.tensor(
            [
                encoded_base[i].max() - encoded_base[i].min()
                for i in range(encoded_base.shape[0])
            ]
        ).cuda()
        max_val_channel = (
            torch.tensor(
                [
                    encoded_base[i].max()
                    + range_channel[i] / (RANGE_MAX - RANGE_MIN) * (1 - RANGE_MAX)
                    for i in range(encoded_base.shape[0])
                ]
            )
            .cuda()
            .view(batch_size, 1, 1, 1)
        )
        min_val_channel = (
            torch.tensor(
                [
                    encoded_base[i].min()
                    - range_channel[i] / (RANGE_MAX - RANGE_MIN) * (RANGE_MIN)
                    for i in range(encoded_base.shape[0])
                ]
            )
            .cuda()
            .view(batch_size, 1, 1, 1)
        )

        encoded_base = (encoded_base - min_val_channel) / (
            max_val_channel - min_val_channel
        )
        encoded_gt = (encoded_gt - min_val_channel) / (max_val_channel - min_val_channel)

        # # 获取原始测量
        # encoded_meas = init_meas(gt_batch, mask3d_batch_train, "Y")

        # # 计算两个测量之间的差
        # meas_difference = init_meas(initial_msi, mask3d_batch_train, "Y") - encoded_meas

        # # 添加一个维度以匹配 (batch, ch, h, w) 的格式
        # encoded_meas = encoded_meas.unsqueeze(1)  # 在 channel 维度插入
        # meas_difference = meas_difference.unsqueeze(1)  # 在 channel 维度插入

        # # 合并到一个 tensor 中，shape 将为 (batch, ch, h, w)
        # encoded_meas = torch.cat(
        #     (encoded_meas, meas_difference), dim=1
        # )  # 在 channel 维度合并

        # 将Tensor转换为PIL图像
        def tensor_to_image(base, gt, index):  # meas,
            deal_with_tiff(
                base, str(BASE_DIR) + "/datasets/Images_Diff_hf/base/image_", index
            )
            deal_with_tiff(gt, str(BASE_DIR) + "/datasets/Images_Diff_hf/gt/image_", index)

            # # 保存 meas 为 .npy 文件
            # np.save(
            #     str(BASE_DIR)
            #     + f"/datasets/Images_Diff_hf/meas/meas_{batch_idx + index}.npy",
            #     meas.cpu().numpy(),
            # )
            # 写入图片文件名列表到 img_files.txt 文件
            file_list.write(
                f"{str(BASE_DIR)}/datasets/Images_Diff_hf/base/image_{batch_idx + index}.tif {str(BASE_DIR)}/datasets/Images_Diff_hf/gt/image_{batch_idx + index}.tif\n"
            )

        import tifffile as tiff

        def deal_with_tiff(arg0, arg1, index):
            """
            This function converts a tensor to an image and saves it in TIFF format.

            Args:
                arg0: The tensor to be converted.
                arg1: The base path for saving the image.
                index: The index of the image.
            """

            # Convert the tensor to a NumPy array.
            arg0 = arg0.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().detach().numpy()

            # Save the image in TIFF format using LZW compression.
            tiff.imwrite(
                f"{arg1}{batch_idx + index}.tif",
                (arg0 * 65535).astype(np.uint16),
                dtype=np.uint16,
                compression="zlib",
            )

        # 循环遍历encoded_gt并保存为图片
        for i in range(encoded_base.shape[0]):
            tensor_to_image(encoded_base[i], encoded_gt[i], i)  # encoded_meas[i],


batch_size = 10
# mask3d_batch, input_mask = init_mask(str(BASE_DIR)+"/packages/MST/datasets/TSA_simu_data/", "Phi", batch_size)
mask3d_batch, input_mask = init_mask(
    str(BASE_DIR) + "/packages/MST/datasets/TSA_simu_data/", "Phi_PhiPhiT", 10
)

test_data = LoadTest(str(BASE_DIR) + "/packages/MST/datasets/TSA_simu_data/Truth/")
test_data = test_data.cuda().float()
# input_meas = init_meas(test_data, mask3d_batch, "H")
input_meas = init_meas(test_data, mask3d_batch, "Y")

with torch.no_grad():
    initial_msi = DAUHST_model(input_meas, input_mask)

    inputs_msi_hf, inputs_msi_lf = wavelet_decomposition_msi(initial_msi, 3)
    inputs_gt_hf = test_data.float().cuda() - inputs_msi_lf

    encoded_base_raw = ChannelVAE_model.encoder(inputs_msi_hf)
    encoded_gt_raw = ChannelVAE_model.encoder(inputs_gt_hf)

    RANGE_MAX = 0.85
    RANGE_MIN = 0.15

    range_channel = torch.tensor(
        [
            encoded_base_raw[i].max() - encoded_base_raw[i].min()
            for i in range(encoded_base_raw.shape[0])
        ]
    ).cuda()
    max_val_channel = (
        torch.tensor(
            [
                encoded_base_raw[i].max()
                + range_channel[i] / (RANGE_MAX - RANGE_MIN) * (1 - RANGE_MAX)
                for i in range(encoded_base_raw.shape[0])
            ]
        )
        .cuda()
        .view(batch_size, 1, 1, 1)
    )
    min_val_channel = (
        torch.tensor(
            [
                encoded_base_raw[i].min()
                - range_channel[i] / (RANGE_MAX - RANGE_MIN) * (RANGE_MIN)
                for i in range(encoded_base_raw.shape[0])
            ]
        )
        .cuda()
        .view(batch_size, 1, 1, 1)
    )

    normalized_encoded_base = (encoded_base_raw - min_val_channel) / (
        max_val_channel - min_val_channel
    )
    normalized_encoded_gt = (encoded_gt_raw - min_val_channel) / (
        max_val_channel - min_val_channel
    )

    # # 获取原始测量
    # encoded_meas = init_meas(test_data, mask3d_batch, "Y")

    # # 计算两个测量之间的差
    # meas_difference = init_meas(initial_msi, mask3d_batch, "Y") - encoded_meas

    # # 添加一个维度以匹配 (batch, ch, h, w) 的格式
    # encoded_meas = encoded_meas.unsqueeze(1)  # 在 channel 维度插入
    # meas_difference = meas_difference.unsqueeze(1)  # 在 channel 维度插入

    # # 合并到一个 tensor 中，shape 将为 (batch, ch, h, w)
    # encoded_meas = torch.cat(
    #     (encoded_meas, meas_difference), dim=1
    # )  # 在 channel 维度合并

torch.save(normalized_encoded_base, str(BASE_DIR) + "/datasets/psnr_val_base.pt")
torch.save(normalized_encoded_gt, str(BASE_DIR) + "/datasets/psnr_val_gt.pt")
# torch.save(encoded_meas, str(BASE_DIR) + "/datasets/psnr_val_meas.pt")

with torch.no_grad():
    x = DAUHST_model(input_meas, input_mask).cpu()

torch.save(initial_msi, str(BASE_DIR) + "/datasets/psnr_raw_base.pt")
torch.save(test_data, str(BASE_DIR) + "/datasets/psnr_raw_gt.pt")
