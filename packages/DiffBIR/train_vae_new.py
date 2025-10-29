import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from utils.common import instantiate_from_config, wavelet_decomposition

# 启用cuDNN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from model import AutoencoderKL
from PIL import Image, ImageDraw, ImageFont
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm


def co_augment_data(lq, gt):
    """
    Augments the input images `lq` and `gt` by applying rotation, flipping, and random crop and resize.

    Args:
        lq (torch.Tensor): The low-quality image batch of shape (B, C, H, W).
        gt (torch.Tensor): The ground truth image batch of shape (B, C, H, W).

    Returns:
        torch.Tensor, torch.Tensor: The augmented low-quality and ground truth image batches.
    """
    B, _, h, w = lq.shape

    # Random horizontal flip (50% chance)
    flip_h = torch.bernoulli(torch.full((B,), 0.5)).bool()
    lq[flip_h] = torch.flip(lq[flip_h], [3])
    gt[flip_h] = torch.flip(gt[flip_h], [3])

    # Random vertical flip (50% chance)
    flip_v = torch.bernoulli(torch.full((B,), 0.5)).bool()
    lq[flip_v] = torch.flip(lq[flip_v], [2])
    gt[flip_v] = torch.flip(gt[flip_v], [2])

    # Random rotation (0, 90, 180, 270 degrees)
    k = torch.randint(0, 4, (B,))
    for i in range(B):
        lq[i] = torch.rot90(lq[i], k[i].item(), [1, 2])
        gt[i] = torch.rot90(gt[i], k[i].item(), [1, 2])

    # Random cropping and resizing
    scale_factors = torch.empty(B).uniform_(2/3, 1)
    new_h = (h * scale_factors).int()
    new_w = (w * scale_factors).int()

    lq_resized, gt_resized = [], []
    for i in range(B):
        # Randomly choose top-left corner for cropping
        top = torch.randint(0, h - new_h[i] + 1, (1,)).item()
        left = torch.randint(0, w - new_w[i] + 1, (1,)).item()

        lq_cropped = lq[i:i+1, :, top:top + new_h[i], left:left + new_w[i]]
        gt_cropped = gt[i:i+1, :, top:top + new_h[i], left:left + new_w[i]]

        # Resize back to original size
        lq_resized.append(F.interpolate(lq_cropped, size=(h, w), mode='bilinear', align_corners=False))
        gt_resized.append(F.interpolate(gt_cropped, size=(h, w), mode='bilinear', align_corners=False))

    lq_resized = torch.cat(lq_resized, dim=0)
    gt_resized = torch.cat(gt_resized, dim=0)

    return lq_resized, gt_resized

def log_txt_as_img(wh, xc):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = []
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def torch_psnr(img, ref):  # input [batch_size, channels, height, width]
    img = (img * 255.0).round()
    ref = (ref * 255.0).round()
    nC = img.shape[1]  # number of channels
    psnr = 0
    for i in range(nC):
        # calculate MSE for each channel
        mse = torch.mean((img[:, i, :, :] - ref[:, i, :, :]) ** 2, dim=(1, 2))
        psnr += 10 * torch.log10((255 * 255) / mse)
    return torch.mean(psnr) / nC


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    vae_cfg = {"embed_dim": 4, "ddconfig": {"double_z": True, "z_channels": 4, "resolution": 256, "in_channels": 3, "out_ch": 3, "ch": 128, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2, "attn_resolutions": [], "dropout": 0.0}}

    vae = AutoencoderKL(**vae_cfg)
    vae.load_state_dict(torch.load(cfg.train.vae_resume, map_location="cpu"), strict=True)

    # Setup optimizer:
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg.train.learning_rate)

    def get_scheduler(optimizer, warmup_steps):
        def lr_lambda(current_step):
            return ((current_step + 1) / warmup_steps + 2) / 3 if current_step < warmup_steps else 1

        # 创建学习率调度器
        scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler

    # 创建学习率调度器
    lr_scheduler = get_scheduler(optimizer=opt, warmup_steps=cfg.train.warmup_steps)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(dataset=dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, drop_last=True)
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} images from {dataset.file_list}")
        psnr_val_base = torch.load("/home/newdisk/btsun/project/Predict-and-Subspace-Refine/datasets/psnr_val_base.pt").cpu().float()
        psnr_val_gt = torch.load("/home/newdisk/btsun/project/Predict-and-Subspace-Refine/datasets/psnr_val_gt.pt").cpu().float()

    # Prepare models for training:
    vae.to(device)
    vae.train()
    vae, opt, loader = accelerator.prepare(vae, opt, loader)
    pure_vae: AutoencoderKL = accelerator.unwrap_model(vae)

    # Variables for monitoring/logging purposes:
    global_step = 0
    iter_step = 0
    epoch = 0
    max_steps = cfg.train.train_steps
    step_loss_vae = []
    epoch_loss_vae = []
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    opt.zero_grad()

    loss_vae_list = []

    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, _ in loader:
            with torch.no_grad():
                gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
                lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)

                lq, gt = co_augment_data(lq, gt)
                lq = lq * 2 - 1

                gt = gt - lq.mean(dim=(1, 2, 3), keepdim=True)
                del lq

            # 训练VAE的Decoder
            loss_vae_it = torch.nn.functional.mse_loss(gt, vae(gt)[0])

            accelerator.backward(loss_vae_it / cfg.train.learn_iter_num)
            loss_vae_list.append(loss_vae_it.item())
            del loss_vae_it, gt

            iter_step = iter_step + 1
            pbar.update(1)

            if iter_step % cfg.train.learn_iter_num == 0 and iter_step > 0:
                opt.step()
                opt.zero_grad()
                accelerator.wait_for_everyone()
                iter_step = 0
            else:
                continue

            global_step += 1

            loss_vae = np.mean(loss_vae_list)
            loss_vae_list.clear()
            step_loss_vae.append(loss_vae)
            epoch_loss_vae.append(loss_vae)

            # 更新学习率
            lr_scheduler.step()

            pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss_vae: {loss_vae:.6f}")

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss_vae = accelerator.gather(torch.tensor(step_loss_vae, device=device).unsqueeze(0)).mean().item()
                step_loss_vae.clear()
                if accelerator.is_local_main_process:
                    writer.add_scalar("loss_vae/loss_vae_simple_step", avg_loss_vae, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0 and accelerator.is_local_main_process:
                checkpoint = pure_vae.state_dict()
                ckpt_path = f"{ckpt_dir}/{global_step:07d}_vae.pt"
                torch.save(checkpoint, ckpt_path)
                del checkpoint

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                with torch.no_grad():
                    if accelerator.is_local_main_process:
                        pure_vae.eval()
                        mean_base = psnr_val_base.to(device).mean(dim=(1, 2, 3), keepdim=True)
                        for tag, image in [
                            ("image/img_lq", psnr_val_base),
                            ("image/img_lq_decoded", pure_vae((psnr_val_base.to(device) - mean_base) * 2)[0] / 2 + mean_base),
                            ("image/img_gt", psnr_val_gt),
                            ("image/img_gt_decoded", pure_vae((psnr_val_gt.to(device) - mean_base) * 2)[0] / 2 + mean_base),
                            ("image/img_gt_error", 2 * torch.abs(psnr_val_gt.to(device) - (pure_vae((psnr_val_gt.to(device) - mean_base) * 2)[0] / 2 + mean_base))),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)

                        writer.add_scalar("psnr/psnr_val_gt", torch_psnr((pure_vae((psnr_val_gt.to(device) - mean_base) * 2)[0] / 2 + mean_base).clamp(0, 1), psnr_val_gt.to(device)), global_step)
                        writer.add_scalar("psnr/psnr_val_base", torch_psnr((pure_vae((psnr_val_base.to(device) - mean_base) * 2)[0] / 2 + mean_base).clamp(0, 1), psnr_val_base.to(device)), global_step)
                        del mean_base
                pure_vae.train()

            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss_vae = accelerator.gather(torch.tensor(epoch_loss_vae, device=device).unsqueeze(0)).mean().item()
        epoch_loss_vae.clear()
        if accelerator.is_local_main_process:
            writer.add_scalar("loss_vae/loss_vae_simple_epoch", avg_epoch_loss_vae, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    # 获取当前脚本文件的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置当前工作目录为脚本文件所在目录
    os.chdir(script_dir)
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/train_vae.yaml")
    args = parser.parse_args()
    main(args)
