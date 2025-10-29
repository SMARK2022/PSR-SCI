import logging
import os
import random

import numpy as np
import scipy.io as sio
import torch
from .ssim_torch import ssim


def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(f"{mask_path}/mask.mat")
    mask = mask["mask"]
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    return mask3d.expand([batch_size, nC, H, W]).cuda().float()


def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(f"{mask_path}/mask_3d_shift.mat")
    mask_3d_shift = mask["mask_3d_shift"]
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1
    # print(Phi_batch.shape, Phi_s_batch.shape)
    return Phi_batch, Phi_s_batch


def LoadTraining(path: str) -> list:
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print("training sences:", len(scene_list))
    for i in range(len(scene_list)):
        # for i in range(5):
        scene_path = path + scene_list[i]
        scene_num = int(scene_list[i].split(".")[0][5:])
        if scene_num <= 205:
            if "mat" not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict["img_expand"] / 65536.0
            elif "img" in img_dict:
                img = img_dict["img"] / 65536.0
            img = img.astype(np.float32)
            imgs.append(img)
            print(f"Sence {i} is loaded. {scene_list[i]}")
    return imgs


def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)["img"]
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


def LoadMeasurement(path_test_meas):
    img = sio.loadmat(path_test_meas)["simulation_test"]
    test_data = img
    test_data = torch.from_numpy(test_data)
    return test_data


# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255 * 255) / mse)
    return psnr / nC


def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def time2file_name(time):
    year = time[:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    return f"{year}_{month}_{day}_{hour}_{minute}_{second}"


def shuffle_crop(train_data, batch_size, crop_size=256, argument=True):
    if argument:
        return generate_augmented_dataset(train_data, batch_size, crop_size)
    index = np.random.choice(range(len(train_data)), batch_size)
    processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
    for i in range(batch_size):
        h, w, _ = train_data[index[i]].shape
        x_index = np.random.randint(0, h - crop_size)
        y_index = np.random.randint(0, w - crop_size)
        processed_data[i, :, :, :] = train_data[index[i]][x_index : x_index + crop_size, y_index : y_index + crop_size, :]
    return torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))


def crop_and_resize(img, crop_size, prob_list, scale_list):
    """
    Crop and resize the image based on predefined probabilities and scales.
    
    Args:
        img (numpy.ndarray): The input image.
        crop_size (int): The target crop size.
        prob_list (list of float): List of probabilities for different scales.
        scale_list (list of float): List of scaling factors corresponding to the probabilities.
    
    Returns:
        numpy.ndarray: The cropped and resized image.
    """
    h, w, _ = img.shape
    prob = np.random.rand()

    # Select the crop factor based on the probability list
    cumulative_prob = 0.0
    crop_factor = 1.0
    for p, s in zip(prob_list, scale_list):
        cumulative_prob += p
        if prob < cumulative_prob:
            crop_factor = s
            break

    crop_h = int(crop_size * crop_factor)
    crop_w = int(crop_size * crop_factor)

    # If crop size is larger than the current dimensions, skip cropping
    if crop_h > h or crop_w > w:
        # Random flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1)  # Horizontal flip
        # Random rotation
        k = np.random.randint(0, 4)
        img = np.rot90(img, k=k, axes=(0, 1))
        img = img.copy()
        crop_h = crop_size
        crop_w = crop_size

    x_index = np.random.randint(0, h - crop_h + 1)
    y_index = np.random.randint(0, w - crop_w + 1)

    cropped_img = img[x_index: x_index + crop_h, y_index: y_index + crop_w, :]
    resized_img = np.array(
        torch.nn.functional.interpolate(
            torch.from_numpy(cropped_img).unsqueeze(0).permute(0, 3, 1, 2).float(),
            size=(crop_size, crop_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)
    )

    return resized_img


def generate_augmented_dataset(train_data, batch_size, crop_size):
    """
    Generates an augmented dataset by processing input training data with specified batch size and crop size.

    Args:
        train_data (list): The input training data.
        batch_size (int): The size of each batch.
        crop_size (int): The size for cropping the images.

    Returns:
        torch.Tensor: The processed augmented dataset in batches.
    """
    prob_list = [0.4, 0.3, 0.3]
    scale_list = [1, 2, 4]
    gt_batch = []

    # The first half data use the original data and augment with arguement_1
    index = np.random.choice(range(len(train_data)), batch_size // 3*2)
    for i in range(batch_size // 3 * 2):
        img = train_data[index[i]]
        cropped_img = crop_and_resize(img, crop_size, prob_list, scale_list)
        cropped_img = torch.from_numpy(np.transpose(cropped_img, (2, 0, 1))).cuda().float()  # Convert to C,H,W
        gt_batch.append(arguement_1(cropped_img))

    # The next sixth data use splicing and augment with arguement_2
    for _ in range(batch_size - (batch_size // 3 * 2) - (batch_size // 8)):
        sample_list = np.random.randint(0, len(train_data), 4)
        spliced_data = np.zeros((4, crop_size // 2, crop_size // 2, 28), dtype=np.float32)
        for j in range(4):
            img = train_data[sample_list[j]]
            cropped_img = crop_and_resize(img, crop_size // 2, prob_list, scale_list)
            spliced_data[j] = cropped_img
        spliced_data = torch.from_numpy(np.transpose(spliced_data, (0, 3, 1, 2))).cuda().float()
        gt_batch.append(arguement_2(spliced_data))

    # The remaining data use random scale cropping and augment with arguement_3
    for _ in range(batch_size // 8):
        index = np.random.choice(range(len(train_data)), 1)[0]
        img = train_data[index]
        cropped_img = crop_and_resize(img, crop_size, prob_list, scale_list)
        cropped_img = torch.from_numpy(np.transpose(cropped_img, (2, 0, 1))).cuda().float()  # Convert to C,H,W
        gt_batch.append(arguement_3(cropped_img, crop_size))

    return torch.stack(gt_batch, dim=0)


def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for _ in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for _ in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for _ in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x


def arguement_2(generate_gt):
    c, h, w = generate_gt.shape[1], 256, 256
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c, h, w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img


def arguement_3(x, crop_size):
    """
    Randomly scales the image by a factor between 0.5 to 3 and then resizes it back to the original crop size.
    :param x: c,h,w
    :param crop_size: the target crop size
    :return: c,h,w
    """
    scale_factor = random.uniform(0.5, 3.0)
    scaled_size = int(crop_size * scale_factor)
    x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(scaled_size, scaled_size), mode="bilinear", align_corners=True).squeeze(0)
    x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(crop_size, crop_size), mode="bilinear", align_corners=True).squeeze(0)
    return x


def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)
        return torch.mul(H, mask3d_batch) if mul_mask else H
    return meas


def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i : step * i + col] = inputs[:, i, :, :]
    return output


def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i : step * i + col - (nC - 1) * step]
    return output


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = f"{model_path}/log.txt"
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def init_mask(mask_path, mask_type, batch_size):
    mask3d_batch = generate_masks(mask_path, batch_size)
    if mask_type == "Phi":
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == "Phi_PhiPhiT":
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == "Mask":
        input_mask = mask3d_batch
    elif mask_type is None:
        input_mask = None
    return mask3d_batch, input_mask


def init_meas(gt, mask, input_setting):
    if input_setting == "H":
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == "HM":
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == "Y":
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas


def checkpoint(model, epoch, model_path, logger):
    model_out_path = f"{model_path}/model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_out_path)
    logger.info(f"Checkpoint saved to {model_out_path}")
