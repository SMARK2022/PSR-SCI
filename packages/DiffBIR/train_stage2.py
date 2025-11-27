import os
from pathlib import Path
from argparse import ArgumentParser

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import numpy as np  # noqa: E402  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from einops import rearrange  # noqa: E402
from model import ControlLDM, Diffusion  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402
from tqdm import tqdm  # noqa: E402
from utils.common import instantiate_from_config  # noqa: E402
from utils.sampler import SpacedSampler  # noqa: E402


from lpips import LPIPS  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402


# Stdlib
def get_basedir(up: int = 3) -> Path:
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


# 启用cuDNN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# --- Setup base dir & env, then import deps (condensed) ---

# only initialize BASE_DIR once
BASE_DIR = globals().get("BASE_DIR")
if not isinstance(BASE_DIR, Path) or not BASE_DIR.exists():
    BASE_DIR = get_basedir()


def augment_data(images: torch.Tensor) -> torch.Tensor:
    """
    Applies data augmentation to a batch of images.

    Args:
        images: A batch of images with shape (batch_size, 3, 256, 256).

    Returns:
        A batch of augmented images with shape (batch_size, 3, 256, 256).
    """

    # # Randomly apply noise to each image with a 25% probability.
    noises = torch.randn_like(images).to(images.device) * 0.01
    rand_idx = torch.rand(images.size(0)) < 0.1
    images[rand_idx] += noises[rand_idx]

    # Randomly apply noise to each image with a 25% probability.
    noises = torch.randn_like(images).to(images.device) * 0.04
    rand_idx = torch.rand(images.size(0)) < 0.1
    images[rand_idx] += noises[rand_idx]

    # Randomly apply a 2x2 blur to each image with a 25% probability.
    blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 0.5))
    rand_idx = torch.rand(images.size(0)) < 0.2
    images[rand_idx] = blur(images[rand_idx])

    # Randomly apply a 2x2 blur to each image with a 25% probability.
    blur = transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.5, 0.5))
    rand_idx = torch.rand(images.size(0)) < 0.1
    images[rand_idx] = blur(images[rand_idx])

    return images


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

    return lq, gt


def calc_loss(pred, gt, lpips_model, weights):
    """
    Calculate a weighted combination of L1, L2, and LPIPS loss.

    Args:
        pred (torch.Tensor): Predicted images.
        gt (torch.Tensor): Ground truth images.
        lpips_model: Pretrained LPIPS model.
        weights (dict): Weights for each loss component, e.g., {"l1": 1.0, "l2": 1.0, "lpips": 0.1}.

    Returns:
        torch.Tensor: Weighted loss value.
    """
    # L1 Loss
    pred = (pred + 1) / 2
    gt = (gt + 1) / 2

    l1_loss = torch.mean(torch.abs(pred - gt))
    # print(f"L1 Loss: {l1_loss.item()}")

    # L2 Loss
    l2_loss = torch.mean((pred - gt) ** 2)
    # print(f"L2 Loss: {l2_loss.item()}")

    # LPIPS Loss
    lpips_loss = lpips_model(pred, gt).mean()
    # print(f"LPIPS Loss: {lpips_loss.item()}")

    # Weighted combination of losses
    total_loss = (
        weights["l1"] * l1_loss +
        weights["l2"] * l2_loss +
        weights["lpips"] * lpips_loss
    )
    print(f"Total Loss: {total_loss.item()}")

    return total_loss


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
    img = (img * 256).round()
    ref = (ref * 256).round()
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
    set_seed(480)
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
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused = cldm.load_pretrained_sd(sd)
    if accelerator.is_local_main_process:
        print(f"strictly load pretrained SD weight from {cfg.train.sd_path}\n" f"unused weights: {unused}")
    del sd

    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_local_main_process:
            print(f"strictly load controlnet weight from checkpoint: {cfg.train.resume}")
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_local_main_process:
            print(f"strictly load controlnet weight from pretrained SD\n" f"weights initialized with newly added zeros: {init_with_new_zero}\n" f"weights initialized from scratch: {init_with_scratch}")

    # 加载 VAE
    if cfg.train.vae_resume:
        cldm.load_vae_from_ckpt(torch.load(cfg.train.vae_resume, map_location="cpu"))
        if accelerator.is_local_main_process:
            print(f"strictly load VAE weight from checkpoint: {cfg.train.vae_resume}")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # Example usage of calc_loss
    lpips_model = LPIPS(net='alex').to(device)  # Load LPIPS model with AlexNet backbone
    weights = {"l1": 0.8, "l2": 0.8, "lpips": 0.05}

    # 创建三个常量，分别表示是否训练 UNet、ControlNet 和 CLIP
    TRAIN_CONTROLNET = True

    # 创建训练参数列表
    train_params = []

    if TRAIN_CONTROLNET:
        train_params.append({"params": cldm.controlnet.parameters(), "lr": cfg.train.learning_rate})

    # 根据训练参数列表创建优化器
    opt = torch.optim.AdamW(train_params)

    def get_scheduler(optimizer, warmup_steps):
        def lr_lambda(current_step):
            return ((current_step + 1) / warmup_steps + 2)/3 if current_step < warmup_steps else 1

        # 创建学习率调度器
        scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler

    # 创建学习率调度器
    lr_scheduler = get_scheduler(optimizer=opt, warmup_steps=cfg.train.warmup_steps)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} images from {dataset.file_list}")
    psnr_val_base = torch.load(str(BASE_DIR) +"/datasets/psnr_val_base.pt").cpu().float()
    psnr_val_gt = torch.load(str(BASE_DIR) +"/datasets/psnr_val_gt.pt").cpu().float()

    # Prepare models for training:
    cldm.train().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)

    # Variables for monitoring/logging purposes:
    global_step = 0
    iter_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch_loss = []
    epoch = 0
    sampler = SpacedSampler(diffusion.betas)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    opt.zero_grad()

    loss_list = []

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_local_main_process,
            unit="batch",
            total=len(loader),
        )
        for gt, lq,  prompt in loader:

            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            # meas = rearrange(meas, "b h w c -> b c h w").contiguous().float().to(device)
            # lq = augment_data(lq)

            gt = (gt - lq.mean(dim=(1, 2, 3)).to(device)[:, None, None, None]) * 2
            lq = lq - lq.mean(dim=(1, 2, 3)).to(device)[:, None, None, None] + 0.5

            with torch.no_grad():
                z_0 = cldm.module.vae_encode(gt)
                cond = cldm.module.prepare_condition(lq, [""] * lq.shape[0])
            # z_0 = cldm.vae_encode(gt)
            # cond = cldm.prepare_condition(lq, meas)
            lq = torch.tensor(0)
            del  lq, prompt

            t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)

            loss_it, x_refine = diffusion.p_losses(cldm, z_0, t, cond)

            loss_it += calc_loss(cldm.module.vae_decode(x_refine), gt, lpips_model, weights)

            accelerator.backward(loss_it / cfg.train.learn_iter_num)
            loss_list.append(loss_it.item())
            loss_it = torch.tensor(0)
            del loss_it, t, gt, x_refine

            N_logs = 5
            z_0 = z_0[:N_logs]
            cond["c_img"] = cond["c_img"][:N_logs]

            iter_step = iter_step + 1
            pbar.update(1)

            if iter_step % cfg.train.learn_iter_num == 0:
                opt.step()
                opt.zero_grad()
                accelerator.wait_for_everyone()
                iter_step = 0
            else:
                continue

            global_step += 1

            if (global_step % cfg.train.image_every == 0 or global_step == 1):
                cldm.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    # z_decode在不同设备上计算
                    z_decode_local = (
                        cldm.module.vae_decode(
                        # cldm.vae_decode(
                            sampler.sample(
                                model=cldm,
                                device=device,
                                steps=100,
                                batch_size=z_0.shape[0],
                                x_size=cond["c_img"].shape[1:],
                                cond=cond,
                                uncond=None,
                                cfg_scale=1.0,
                                x_T=diffusion.q_sample(
                                    cond["c_img"],
                                    torch.full((cond["c_img"].shape[0],), diffusion.num_timesteps - 1, dtype=torch.long, device=device),
                                    torch.randn(cond["c_img"].shape, dtype=torch.float32, device=device),
                                ),
                                progress=accelerator.is_local_main_process,
                                progress_leave=False,
                            )
                        )
                        / 2
                        + 0.5
                    )

                    # Gather和释放
                    z_decode = accelerator.gather(z_decode_local)
                    z_0_gathered = accelerator.gather(cldm.module.vae_decode(z_0) / 2 + 0.5)
                    c_img_gathered = accelerator.gather(cldm.module.vae_decode(cond["c_img"]) / 2 + 0.5)
                    # z_0_gathered = accelerator.gather(cldm.vae_decode(z_0) / 2 + 0.5)
                    # c_img_gathered = accelerator.gather(cldm.vae_decode(cond["c_img"]) / 2 + 0.5)

                    if accelerator.is_local_main_process:
                        torch.cuda.empty_cache()

                    if accelerator.is_local_main_process:
                        # 逐步计算和释放显存
                        for tag, image in [
                            (
                                "image_train/train_samples",
                                z_decode,
                            ),
                            (
                                "image_train/train_gt",
                                z_0_gathered,
                            ),
                            (
                                "image_train/train_base",
                                c_img_gathered,
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=5), global_step)
                            del image  # 释放内存
                            torch.cuda.empty_cache()

                        # 计算并释放显存
                        writer.add_scalar(
                            "psnr/psnr_train_delta",
                            torch_psnr(
                                z_decode,
                                z_0_gathered,
                            )
                            - torch_psnr(
                                c_img_gathered,
                                z_0_gathered,
                            ),
                            global_step,
                        )
                    del z_decode, z_0_gathered, c_img_gathered  # 释放内存
                cldm.train()


            del z_0, cond
            loss = np.mean(loss_list)
            loss_list.clear()
            step_loss.append(loss)
            epoch_loss.append(loss)

            # 更新学习率
            lr_scheduler.step()

            pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss:.6f}")

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                if accelerator.is_local_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)

            # 保存检查点，根据常量决定是否保存 UNet、ControlNet 和 CLIP
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0 and accelerator.is_local_main_process:
                # 保存 ControlNet 检查点
                if TRAIN_CONTROLNET:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/controlnet_{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

                # 删除 checkpoint 释放内存
                del checkpoint

            torch.cuda.empty_cache()
            if global_step % cfg.train.image_every == 0 or global_step == 1:
                cldm.eval()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    psnr_val_base = psnr_val_base.to(device)

                    mean_base = psnr_val_base.mean(dim=(1, 2, 3)).to(device)[:, None, None, None]
                    log_cond = cldm.module.prepare_condition(psnr_val_base - mean_base + 0.5, [""] * psnr_val_base.shape[0])
                    # log_cond = cldm.prepare_condition(psnr_val_base - mean_base + 0.5, psnr_val_meas)
                    psnr_val_base = psnr_val_base.to("cpu")

                    z_decode = (
                        cldm.module.vae_decode(
                        # cldm.vae_decode(
                            sampler.sample(
                                model=cldm,
                                device=device,
                                steps=100,
                                batch_size=psnr_val_base.shape[0],
                                x_size=log_cond["c_img"].shape[1:],
                                cond=log_cond,
                                uncond=None,
                                cfg_scale=1.0,
                                x_T=diffusion.q_sample(
                                    log_cond["c_img"],
                                    torch.full((psnr_val_base.shape[0],), diffusion.num_timesteps - 1, dtype=torch.long, device=device),
                                    torch.randn(log_cond["c_img"].shape, dtype=torch.float32, device=device),
                                ),
                                progress=accelerator.is_local_main_process,
                                progress_leave=False,
                            )
                        )
                        / 2
                        + mean_base
                    )

                    # z_decode = wavelet_decomposition(z_decode, 3)[0] + wavelet_decomposition(psnr_val_base, 3)[1]

                    if accelerator.is_local_main_process:
                        torch.cuda.empty_cache()

                    del log_cond
                    psnr_val_gt = psnr_val_gt.to(device)
                    psnr_val_base = psnr_val_base.to(device)

                    if accelerator.is_local_main_process:
                        for tag, image in [
                            (
                                "image_val/update",
                                4 * torch.abs(z_decode - psnr_val_base),
                            ),
                            (
                                "image_val/error",
                                4 * torch.abs(z_decode - psnr_val_gt),
                            ),
                            (
                                "image_val/samples",
                                z_decode,
                            ),
                            (
                                "image_val/gt",
                                psnr_val_gt,
                            ),
                            (
                                "image_val/base",
                                psnr_val_base,
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=5), global_step)
                            del image

                        writer.add_scalar(
                            "psnr/psnr_val_sample",
                            torch_psnr(
                                z_decode,
                                psnr_val_gt,
                            ),
                            global_step,
                        )
                        writer.add_scalar(
                            "psnr/psnr_val_baseline",
                            torch_psnr(
                                psnr_val_base,
                                psnr_val_gt,
                            ),
                            global_step,
                        )
                        writer.add_scalar(
                            "psnr/psnr_val_delta",
                            torch_psnr(
                                z_decode,
                                psnr_val_gt,
                            )
                            - torch_psnr(
                                psnr_val_base,
                                psnr_val_gt,
                            ),
                            global_step,
                        )
                    del z_decode
                    psnr_val_base = psnr_val_base.to("cpu")
                    psnr_val_gt = psnr_val_gt.to("cpu")
                cldm.train()

            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0)).mean().item()
        epoch_loss.clear()
        if accelerator.is_local_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    # 获取当前脚本文件的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置当前工作目录为脚本文件所在目录
    os.chdir(script_dir)
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/train_stage2.yaml")
    args = parser.parse_args()

    main(args)
