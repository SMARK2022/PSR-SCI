from typing import Tuple, overload

import torch
from torch.nn import functional as F


class Guidance:

    def __init__(self, scale: float, t_start: int, t_stop: int, space: str, repeat: int, rgb_subscale: float = 0) -> "Guidance":
        """
        Initialize restoration guidance.

        Args:
            scale (float): Gradient scale (denoted as `s` in our paper). The larger the gradient scale, 
                the closer the final result will be to the output of the first stage model.
            t_start (int), t_stop (int): The timestep to start or stop guidance. Note that the sampling 
                process starts from t=1000 to t=0, the `t_start` should be larger than `t_stop`.
            space (str): The data space for computing loss function (rgb or latent).

        Our restoration guidance is based on [GDP](https://github.com/Fayeben/GenerativeDiffusionPrior).
        Thanks for their work!
        """
        self.scale = scale
        self.rgb_subscale = rgb_subscale
        self.t_start = t_start
        self.t_stop = t_stop
        self.target = None
        self.space = space
        self.repeat = repeat
    
    def load_target(self, target: torch.Tensor) -> None:
        self.target = target

    def __call__(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # avoid propagating gradient out of this scope
        pred_x0 = pred_x0.detach().clone()
        target_x0 = target_x0.detach().clone()
        return self._forward(target_x0, pred_x0, t)
    
    @overload
    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        ...


class MSEGuidance(Guidance):

    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # inputs: [-1, 1], nchw, rgb
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = (pred_x0 - target_x0).pow(2).mean((1, 2, 3)).sum()
        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()

class MeasMSEGuidance(Guidance):
    def load_guidance(self, target: torch.Tensor, masks: torch.Tensor, max_val_channel, min_val_channel, inputs_msi_lf, decoder: torch.nn.Module) -> None:
        self.target = target
        self.mask3d_batch = masks
        self.max_val_channel, self.min_val_channel, self.inputs_msi_lf = max_val_channel, min_val_channel, inputs_msi_lf
        self.decoder = decoder
        self.rgb_target = None
        self.bias = 0

    def load_bias(self, bias: torch.Tensor):
        self.bias = bias

    def load_rgb_target(self, rgb_target: torch.Tensor):
        self.rgb_target = rgb_target

    def _forward(self, target: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # inputs: [-1, 1], nchw, rgb
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            pred_x0 = (pred_x0) / 2 + self.bias

            def shift(inputs, step=2):
                [bs, nC, row, col] = inputs.shape
                output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
                for i in range(nC):
                    output[:, i, :, step * i: step * i + col] = inputs[:, i, :, :]
                return output

            def gen_meas_torch(data_batch, mask3d_batch):
                temp = shift(mask3d_batch * data_batch, 2)
                meas = torch.sum(temp, 1)
                return meas
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(gen_meas_torch(self.decoder(pred_x0 * (self.max_val_channel - self.min_val_channel) + self.min_val_channel), self.mask3d_batch)[0].detach().cpu().numpy())
            # plt.colorbar()
            # plt.tight_layout()
            # plt.savefig(f"/home/newdisk/btsun/project/Predict-and-Subspace-Refine/visual/pred_meas_step_{t}.png")
            # plt.close()

            # plt.figure()
            # plt.imshow(pred_x0[0].permute(1, 2, 0).detach().cpu().numpy())
            # plt.tight_layout()
            # plt.savefig(f"/home/newdisk/btsun/project/Predict-and-Subspace-Refine/visual/pred_x0_step_{t}.png")
            # plt.close()
            loss = (gen_meas_torch(self.decoder((pred_x0) * (self.max_val_channel - self.min_val_channel) + self.min_val_channel) + self.inputs_msi_lf, self.mask3d_batch) - target).pow(2).mean((1, 2)).sum()
            if not self.rgb_target is None:
                loss = loss + (pred_x0 - self.rgb_target).pow(2).mean((1, 2, 3)).sum() * self.rgb_subscale
        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()


class WeightedMSEGuidance(Guidance):

    def _get_weight(self, target: torch.Tensor) -> torch.Tensor:
        # convert RGB to G
        rgb_to_gray_kernel = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        target = torch.sum(target * rgb_to_gray_kernel.to(target.device), dim=1, keepdim=True)
        # initialize sobel kernel in x and y axis
        G_x = [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]
        G_y = [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]
        G_x = torch.tensor(G_x, dtype=target.dtype, device=target.device)[None]
        G_y = torch.tensor(G_y, dtype=target.dtype, device=target.device)[None]
        G = torch.stack((G_x, G_y))

        target = F.pad(target, (1, 1, 1, 1), mode='replicate') # padding = 1
        grad = F.conv2d(target, G, stride=1)
        mag = grad.pow(2).sum(dim=1, keepdim=True).sqrt()

        n, c, h, w = mag.size()
        block_size = 2
        blocks = mag.view(n, c, h // block_size, block_size, w // block_size, block_size).permute(0, 1, 2, 4, 3, 5).contiguous()
        block_mean = blocks.sum(dim=(-2, -1), keepdim=True).tanh().repeat(1, 1, 1, 1, block_size, block_size).permute(0, 1, 2, 4, 3, 5).contiguous()
        block_mean = block_mean.view(n, c, h, w)
        weight_map = 1 - block_mean

        return weight_map

    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # inputs: [-1, 1], nchw, rgb
        with torch.no_grad():
            w = self._get_weight((target_x0 + 1) / 2)
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = ((pred_x0 - target_x0).pow(2) * w).mean((1, 2, 3)).sum()
        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()
