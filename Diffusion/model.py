import copy
import torch
import numpy as np
from utils import DDIMSampler
import os
from typing import List, Tuple, Optional, Union
from .modules import DiffUnet


class DiffusionModel:
    def __init__(self,
                 main_net: DiffUnet,
                 ema_net: Optional[DiffUnet] = None,
                 num_steps: int = 100,
                 input_res: Union[Tuple[int, int], List[int]] = (32, 32),
                 emma: float = 0.999,
                 noise_sch_name: str = 'cosv2',
                 **noise_sch_kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps_net = main_net.to(self.device)
        self.ema_net = ema_net if ema_net is not None else copy.deepcopy(main_net)
        self.ema_net = self.ema_net.to(self.device)
        self.ema_net.eval()
        self.steps = num_steps
        self.res = (3,) + input_res if isinstance(input_res, tuple) else [3] + input_res
        self.num_steps = num_steps
        self.scheduler = DDIMSampler(noise_sch_name,
                                     diff_train_steps=num_steps,
                                     **noise_sch_kwargs)
        self.emma = emma

    @torch.no_grad()
    def generate(self,
                 num_samples: int,
                 num_infer_steps: int,
                 pred_net: Optional[str] = 'ema',
                 return_list: bool = False,
                 x_t: Optional[torch.Tensor] = None):
        shape = (num_samples,) + self.res if isinstance(self.res, tuple) else [num_samples] + self.res
        x_t = torch.randn(*shape).to(self.device) if x_t is None else x_t
        self.scheduler.set_infer_steps(num_infer_steps, x_t.device)
        pred_net = getattr(self, pred_net + "_net")
        xs = [x_t.cpu()]
        for step in range(num_infer_steps):
            t = self.scheduler.timesteps[step]
            x_t, _ = self.scheduler.p_sample(x_t, t, pred_net)
            xs.append(x_t.cpu())
        return xs[-1] if not return_list else xs

    @staticmethod
    def inverse_transform(img):
        """ Inverse transform the images after generation"""
        img = (img + 1) / 2
        img = np.clip(img, 0.0, 1.0)
        img = np.transpose(img, (1, 2, 0)) if len(img.shape) == 3 else np.transpose(img, (0, 2, 3, 1))
        return img

    @staticmethod
    def transform(img):
        """Transform the image before training converting the pixels values from [0, 255] to [-1, 1]"""
        img = img.to(torch.float32) / 127.5
        img = img - 1
        if len(img.shape) == 3:  # one sample
            img = torch.permute(img, (2, 0, 1))
        else:  # batch of samples
            img = torch.permute(img, (0, 3, 1, 2))
        return img

    def train_loss(self,
                   input_batch: torch.Tensor,
                   loss_type: Optional[str] = 'l1_loss',
                   **losskwargs):
        """Training loss"""
        bs, _, _, _ = input_batch.shape
        t = torch.randint(0, self.num_steps, size=(bs,))
        x_t, eps = self.scheduler.q_sample(input_batch, t)
        t = t.int().to(input_batch.device)
        eps_pred = self.eps_net(x_t, t)
        loss = getattr(torch.nn.functional, loss_type)(eps_pred, eps, **losskwargs)
        return loss

    def update_emma(self):
        for p_ema, p in zip(self.ema_net.parameters(), self.eps_net.parameters()):
            p_ema.data = (1 - self.emma) * p.data + p_ema.data * self.emma

    def train(self):
        self.eps_net.train()

    def eval(self):
        self.eps_net.eval()

    def parameters(self):
        return self.eps_net.parameters()

    def save(self,
             file_name: str):
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        ema_path = file_name + '/ema.pt'
        net_path = file_name + "/eps.pt"
        torch.save(self.ema_net.state_dict(), ema_path)
        torch.save(self.eps_net.state_dict(), net_path)

    def load(self,
             path_nets: str):
        pathes = [os.path.join(path_nets, p) for p in os.listdir(path_nets) if ("ema" in p or "eps" in p)]
        for index in range(len(pathes)):
            if "ema" in pathes[index]:
                break
        ema_p = pathes[index]
        eps_p = pathes[int(not index)]
        map_loc = 'cpu' if not torch.cuda.is_available() else 'cuda'
        self.eps_net.load_state_dict(torch.load(eps_p, map_location=map_loc))
        self.ema_net.load_state_dict(torch.load(ema_p, map_location=map_loc))
