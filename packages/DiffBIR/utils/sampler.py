import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchsde
from torch import nn
from torch.nn import functional as F
from ..model.cldm import ControlLDM
from ..model.gaussian_diffusion import extract_into_tensor
from tqdm import tqdm

sys.path.append('/home/newdisk/btsun/project/Predict-and-Subspace-Refine/DiffBIR/')
from DiffBIR.utils.common import gaussian_weights, sliding_windows
from DiffBIR.utils.cond_fn import Guidance


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedSampler(nn.Module):
    """
    Implementation for spaced sampling schedule proposed in IDDPM. This class is designed
    for sampling ControlLDM.
    
    https://arxiv.org/pdf/2102.09672.pdf
    """

    def __init__(self, betas: np.ndarray) -> "SpacedSampler":
        super().__init__()
        self.num_timesteps = len(betas)
        self.original_betas = betas
        self.original_alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        self.context = {}

    def register(self, name: str, value: np.ndarray) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def make_schedule(self, num_steps: int) -> None:
        # calcualte betas for spaced sampling
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
        used_timesteps = space_timesteps(self.num_timesteps, str(num_steps))
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.original_alphas_cumprod):
            if i in used_timesteps:
                # marginal distribution is the same as q(x_{S_t}|x_0)
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        assert len(betas) == num_steps
        self.timesteps = np.array(sorted(list(used_timesteps)), dtype=np.int32) # e.g. [0, 10, 20, ...]

        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # print(f"sampler sqrt_alphas_cumprod: {np.sqrt(alphas_cumprod)[-1]}")
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )

        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Implement the posterior distribution q(x_{t-1}|x_t, x_0).
        
        Args:
            x_start (torch.Tensor): The predicted images (NCHW) in timestep `t`.
            x_t (torch.Tensor): The sampled intermediate variables (NCHW) of timestep `t`.
            t (torch.Tensor): Timestep (N) of `x_t`. `t` serves as an index to get 
                parameters for each timestep.
        
        Returns:
            posterior_mean (torch.Tensor): Mean of the posterior distribution.
            posterior_variance (torch.Tensor): Variance of the posterior distribution.
            posterior_log_variance_clipped (torch.Tensor): Log variance of the posterior distribution.
        """
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def apply_cond_fn(
        self,
        model: ControlLDM,
        pred_x0: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        upscale: float,
        cond_fn: Guidance,
        beta1: float = 0.8,  # 动量因子
    ) -> torch.Tensor:
        t_now = int(t[0].item()) + 1
        if not (cond_fn.t_stop < t_now and t_now < cond_fn.t_start):
            # stop guidance
            self.context["g_apply"] = False
            return pred_x0

        # 初始化动量变量
        if "momentum" not in self.context:
            self.context["momentum"] = torch.zeros_like(pred_x0)

        grad_rescale = 1 / extract_into_tensor(self.posterior_mean_coef1, index, pred_x0.shape)
        # grad_rescale = torch.max(100 - grad_rescale, torch.tensor(4.0, device=grad_rescale.device))
        grad_rescale = torch.max(grad_rescale, torch.tensor(4.0, device=grad_rescale.device))
        # grad_rescale = torch.min(grad_rescale, torch.tensor(50.0, device=grad_rescale.device))

        # grad_rescale = 32.0

        # print(grad_rescale)
        momentum = self.context["momentum"]
        torch.cuda.empty_cache()

        # apply guidance for multiple times
        loss_vals = []
        try:
            for _ in range(cond_fn.repeat):
                # set target and pred for gradient computation
                target, pred = None, None
                if cond_fn.space == "latent":
                    target = model.vae_encode(cond_fn.target)
                    pred = pred_x0
                elif cond_fn.space == "rgb":
                    # We need to backward gradient to x0 in latent space, so it's required
                    # to trace the computation graph while decoding the latent.
                    with torch.enable_grad():
                        target = cond_fn.target
                        pred_x0_rg = pred_x0.detach().clone().requires_grad_(True)
                        pred = model.vae_decode(pred_x0_rg)
                        if upscale > 1.0:
                            pred = F.interpolate(pred, size=(int(pred.shape[-2] / upscale), int(pred.shape[-1] / upscale)), mode="bicubic", antialias=True)
                        assert pred.requires_grad
                else:
                    raise NotImplementedError(cond_fn.space)

                # compute gradient
                delta_pred, loss_val = cond_fn(target, pred, t_now)
                loss_vals.append(loss_val)

                # update pred_x0 w.r.t gradient
                if cond_fn.space == "latent":
                    delta_pred_x0 = delta_pred
                    # 动量更新和权重衰减
                    momentum = beta1 * momentum + (1 - beta1) * delta_pred_x0 * grad_rescale
                    pred_x0 = pred_x0 + momentum  # AdamW-like update
                elif cond_fn.space == "rgb":
                    pred.backward(delta_pred)
                    delta_pred_x0 = pred_x0_rg.grad
                    # 动量更新和权重衰减
                    momentum = beta1 * momentum + (1 - beta1) * delta_pred_x0 * grad_rescale
                    pred_x0 = pred_x0 + momentum  # AdamW-like update
                else:
                    raise NotImplementedError(cond_fn.space)

            # 解引用并释放计算图
            pred_x0_rg.grad = None
            pred_x0_rg.detach_()
            del delta_pred_x0, pred_x0_rg, pred, target

        except Exception as e:  # 捕获所有异常
            # Release resources and clear the computation graph
            if 'pred_x0_rg' in locals():
                pred_x0_rg.grad = None
                pred_x0_rg.detach_()
                del pred_x0_rg  # 仅删除定义过的变量

            # 使用条件检查确保变量存在再进行删除
            if 'delta_pred_x0' in locals():
                del delta_pred_x0
            if 'pred' in locals():
                del pred
            if 'target' in locals():
                del target
            print(f"An error occurred: {e}")  # 打印错误信息

        # 清除缓存，释放 GPU 内存
        torch.cuda.empty_cache()

        # 保存动量以供下次迭代使用
        self.context["momentum"] = momentum
        self.context["g_apply"] = True
        self.context["g_loss"] = float(np.mean(loss_vals))

        return pred_x0

    def predict_noise(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.:
            model_output = model(x, t, cond)
        else:
            # apply classifier-free guidance
            model_cond = model(x, t, cond)
            model_uncond = model(x, t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        return model_output

    @torch.no_grad()
    def predict_noise_tiled(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
        tile_size: int,
        tile_stride: int
    ):
        _, _, h, w = x.shape
        tiles = tqdm(sliding_windows(h, w, tile_size // 8, tile_stride // 8), unit="tile", leave=False)
        eps = torch.zeros_like(x)
        count = torch.zeros_like(x, dtype=torch.float32)
        weights = gaussian_weights(tile_size // 8, tile_size // 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=x.device)
        for hi, hi_end, wi, wi_end in tiles:
            tiles.set_description(f"Process tile ({hi} {hi_end}), ({wi} {wi_end})")
            tile_x = x[:, :, hi:hi_end, wi:wi_end]
            tile_cond = {
                "c_img": cond["c_img"][:, :, hi:hi_end, wi:wi_end],
                "c_txt": cond["c_txt"]
            }
            if uncond:
                tile_uncond = {
                    "c_img": uncond["c_img"][:, :, hi:hi_end, wi:wi_end],
                    "c_txt": uncond["c_txt"]
                }
            tile_eps = self.predict_noise(model, tile_x, t, tile_cond, tile_uncond, cfg_scale)
            # accumulate noise
            eps[:, :, hi:hi_end, wi:wi_end] += tile_eps * weights
            count[:, :, hi:hi_end, wi:wi_end] += weights
        # average on noise (score)
        eps.div_(count)
        return eps

    @torch.no_grad()
    def p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        upscale: float,
        cfg_scale: float,
        cond_fn: Optional[Guidance],
        tiled: bool,
        tile_size: int,
        tile_stride: int
    ) -> torch.Tensor:
        if tiled:
            eps = self.predict_noise_tiled(model, x, t, cond, uncond, cfg_scale, tile_size, tile_stride)
        else:
            eps = self.predict_noise(model, x, t, cond, uncond, cfg_scale)
        pred_x0 = self._predict_xstart_from_eps(x, index, eps)
        if cond_fn:
            assert not tiled, f"tiled sampling currently doesn't support guidance"
            pred_x0 = self.apply_cond_fn(model, pred_x0, t, index, upscale, cond_fn)
        model_mean, model_variance, _ = self.q_posterior_mean_variance(pred_x0, x, index)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (index != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        x_prev = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        batch_size: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        upscale:float = 1.0,
        cond_fn: Optional[Guidance]=None,
        tiled: bool=False,
        tile_size: int=-1,
        tile_stride: int=-1,
        x_T: Optional[torch.Tensor]=None,
        progress: bool=True,
        progress_leave: bool=True,
    ) -> torch.Tensor:
        self.make_schedule(steps)
        self.to(device)
        if x_T is None:
            # TODO: not convert to float32, may trigger an error
            img = torch.randn((batch_size, *x_size), device=device)
        else:
            img = x_T
        timesteps = np.flip(self.timesteps) # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, leave=progress_leave, disable=not progress)
        for i, step in enumerate(iterator):
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)
            img = self.p_sample(
                model, img, ts, index, cond, uncond,upscale, cfg_scale, cond_fn,
                tiled, tile_size, tile_stride
            )
            if cond_fn and self.context["g_apply"]:
                loss_val = self.context["g_loss"]
                desc = f"Spaced Sampler With Guidance, Loss: {loss_val:.6f}"
                # print(f"Spaced Sampler With Guidance, Loss: {loss_val:.6f}")
            else:
                desc = "Spaced Sampler"
            iterator.set_description(desc)
        return img

    @torch.no_grad()
    def sample_dpmpp_2m(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        batch_size: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        cond_fn: Optional[Guidance] = None,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: Optional[torch.Tensor] = None,
        progress: bool = True,
        progress_leave: bool = True,
    ) -> torch.Tensor:

        original_sigmas_log = torch.tensor(((1 - self.original_alphas_cumprod) / self.original_alphas_cumprod) ** 0.5, device=device).log().float()

        def append_zero(x):
            return torch.cat([x, x.new_zeros([1])])

        def get_sigmas_karras(n, sigma_min=0.029167532920837402, sigma_max=14.614642143249512, rho=7.0, device="cpu"):
            """Constructs the noise schedule of Karras et al. (2022)."""
            ramp = torch.linspace(0, 1, n)
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            return append_zero(sigmas).to(device)

        sigmas = get_sigmas_karras(steps)

        used_timesteps = torch.tensor([torch.argmin(torch.abs(original_sigmas_log - sigma.log().float())) for sigma in sigmas])
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.original_alphas_cumprod):
            if i in used_timesteps:
                # marginal distribution is the same as q(x_{S_t}|x_0)
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod

        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # print(f"sampler sqrt_alphas_cumprod: {np.sqrt(alphas_cumprod)[-1]}")
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

        self.to(device)
        if x_T is None:
            img = torch.randn((batch_size, *x_size), device=device)
        else:
            img = x_T

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_denoised = None

        total_steps = len(sigmas) - 1
        iterator = tqdm(sigmas[:-1], total=total_steps, leave=progress_leave, disable=not progress)
        for i, sigma in enumerate(iterator):

            ts = torch.full((batch_size,), torch.argmin(torch.abs(original_sigmas_log - sigma.log().float())), device=device, dtype=torch.long)
            denoised = img - sigma * self.predict_noise(model, img, ts, cond, uncond, cfg_scale)
            if not cond_fn is None:
                denoised = self.apply_cond_fn(model, denoised, ts, torch.full_like(ts, fill_value=total_steps - i - 1), cond_fn)

            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if old_denoised is None or sigmas[i + 1] == 0:
                img = (sigma_fn(t_next) / sigma_fn(t)) * img - (-h).expm1() * denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                img = (sigma_fn(t_next) / sigma_fn(t)) * img - (-h).expm1() * denoised_d
            old_denoised = denoised

            if cond_fn and self.context["g_apply"]:
                loss_val = self.context["g_loss"]
                desc = f"Spaced Sampler With Guidance, Loss: {loss_val:.6f}"
            else:
                desc = "Spaced Sampler"
            iterator.set_description(desc)
        return img

    @torch.no_grad()
    def sample_dpmpp_2m_sde(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        batch_size: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        seed: torch.Tensor = None,
        eta: float = 1.0,
        s_noise: float = 0.5,
        cond_fn: Optional[Guidance] = None,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: Optional[torch.Tensor] = None,
        progress: bool = True,
        progress_leave: bool = True,
    ) -> torch.Tensor:

        del self.context
        self.context = {}

        original_sigmas_log = torch.tensor(((1 - self.original_alphas_cumprod) / self.original_alphas_cumprod) ** 0.5, device=device).log().float()
        solver_type = "heun"

        def append_zero(x):
            return torch.cat([x, x.new_zeros([1])])

        def get_sigmas_karras(n, sigma_min=0.029167532920837402, sigma_max=14.614642143249512, rho=7.0, device="cpu"):
            """Constructs the noise schedule of Karras et al. (2022)."""
            ramp = torch.linspace(0, 1, n)
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            return append_zero(sigmas).to(device)

        sigmas = get_sigmas_karras(steps).to(device)

        used_timesteps = torch.tensor([torch.argmin(torch.abs(original_sigmas_log - sigma.log().float())) for sigma in sigmas])
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.original_alphas_cumprod):
            if i in used_timesteps:
                # marginal distribution is the same as q(x_{S_t}|x_0)
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod

        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # print(f"sampler sqrt_alphas_cumprod: {np.sqrt(alphas_cumprod)[-1]}")
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

        self.to(device)
        if x_T is None:
            img = torch.randn((batch_size, *x_size), device=device)
        else:
            img = x_T

        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        noise_sampler = BrownianTreeNoiseSampler(img, sigma_min, sigma_max, seed=seed, cpu=True)
        old_denoised = None

        total_steps = len(sigmas) - 1
        iterator = tqdm(sigmas[:-1], total=total_steps, leave=progress_leave, disable=not progress)
        for i, sigma in enumerate(iterator):
            ts = torch.full((batch_size,), torch.argmin(torch.abs(original_sigmas_log - sigma.log().float())), device=device, dtype=torch.long)
            denoised = img - sigma * self.predict_noise(model, img, ts, cond, uncond, cfg_scale)
            if not cond_fn is None:
                denoised = self.apply_cond_fn(model, denoised, ts, torch.full_like(ts, fill_value=total_steps - i - 1), cond_fn)

            if sigmas[i + 1] == 0:
                # Denoising step
                img = denoised
                # img = self.apply_cond_fn(model, img, ts, torch.full_like(ts, fill_value=total_steps - i - 1), cond_fn)
            else:
                # DPM-Solver++(2M) SDE
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                eta_h = eta * h

                img = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * img + (-h - eta_h).expm1().neg() * denoised
                # img = self.apply_cond_fn(model, img, ts, torch.full_like(ts, fill_value=total_steps - i - 1), cond_fn)

                if old_denoised is not None:
                    r = h_last / h
                    if solver_type == "heun":
                        img = img + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                    elif solver_type == "midpoint":
                        img = img + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

                if eta:
                    img = img + noise_sampler(sigmas[i], sigmas[i + 1]).to(device) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

            old_denoised = denoised
            h_last = h

            if cond_fn and self.context["g_apply"]:
                loss_val = self.context["g_loss"]
                desc = f"Spaced Sampler With Guidance, Loss: {loss_val:.6f}"
            else:
                desc = "Spaced Sampler"
            iterator.set_description(desc)

        del self.context
        self.context = {}
        torch.cuda.empty_cache()
        return img


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs) for s in seed]
        else:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if self.cpu_tree:
            w = torch.stack([tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device) for tree in self.trees]) * (self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)

        return w if self.batched else w[0]

class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()
