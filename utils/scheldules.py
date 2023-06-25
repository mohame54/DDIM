import math
import numpy as np
import torch
from typing import Union

def get_beta_schedule(beta_schedule: str,
                      *,
                      beta_start: float,
                      beta_end: float,
                      num_diffusion_timesteps: int
                      ):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float32,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosv2":
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), beta_end))
        betas = np.array(betas)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class DDIMSampler:

    def __init__(self,
                 schedule_name: str,
                 diff_train_steps: int,
                 beta_start: float = 0.001,
                 beta_end: float = 0.2):
        betas = get_beta_schedule(schedule_name,
                                  beta_start=beta_start,
                                  beta_end=beta_end,
                                  num_diffusion_timesteps=diff_train_steps)
        self.betas = torch.tensor(betas).to(torch.float32)
        self.alpha = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

        self.timesteps = np.arange(0, diff_train_steps)[::-1]
        self.num_train_steps = diff_train_steps
        self._num_inference_steps = 20
        self.eta = 0

    def _get_variance(self,
                      timestep: Union[torch.Tensor, int],
                      prev_timestep: Union[torch.Tensor, int] ):
        alpha_t = self.alpha_cumprod[timestep]
        alpha_prev = self.alpha_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_t = (1 - alpha_t)
        beta_prev = (1 - alpha_prev)
        return (beta_prev / beta_t) / (1 - alpha_t / alpha_prev)

    @staticmethod
    def treshold_sample(sample: torch.Tensor,
                        threshold: float = 0.9956,
                        max_clip: float = 1):
        batch_size, channels, height, width = sample.shape
        dtype = sample.dtype
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * height * width)

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, threshold, dim=1)
        s = torch.clamp(
            s, min=1, max=max_clip
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]

        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, height, width)
        sample = sample.to(dtype)

        return sample

    def set_infer_steps(self,
                        num_steps: int,
                        device: torch.DeviceObjType):
        self._num_inference_steps = num_steps
        step_ratio = self.num_train_steps // self._num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    @torch.no_grad()
    def p_sample(self,
                 x_t: torch.Tensor,
                 t_now: Union[torch.Tensor, int],
                 pred_net):
        prev_timestep = t_now - self.num_train_steps // self._num_inference_steps
        alpha_t = self.alpha_cumprod[t_now]
        alpha_prev = self.alpha_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        var = self._get_variance(t_now, prev_timestep)
        eps = torch.randn_like(x_t).to(x_t.device)
        t_now = (torch.ones((x_t.shape[0],),
                            device=x_t.device,
                            dtype=torch.int32) * t_now).to(x_t.device)
        eta_t = pred_net(x_t, t_now)

        x0_t = (x_t - eta_t * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

        c1 = self.eta * var.sqrt()
        c2 = ((1 - alpha_prev) - c1 ** 2).sqrt()
        x_tminus = alpha_prev.sqrt() * x0_t + c2 * eta_t + c1 * eps
        return x_tminus, x0_t

    def q_sample(self,
                 x_t: torch.Tensor,
                 timesteps: Union[torch.Tensor, int]):

        alpha_t = self.alpha_cumprod[timesteps].to(timesteps.device)
        alpha_t = alpha_t.flatten().to(x_t.device)[:, None, None, None]
        eps = torch.randn(*list(x_t.shape)).to(x_t.device)
        x_t = alpha_t.sqrt() * x_t + (1 - alpha_t).sqrt() * eps
        return x_t, eps
