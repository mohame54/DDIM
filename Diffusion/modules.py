from torch import nn
import torch
from typing import Optional, List, Union, Tuple
import torch.nn.functional as f
import math

MAX_FREQ = 1000


# sinusoidal embedding like in Transformers
def get_timestep_embedding(timesteps: torch.Tensor,
                           embedding_dim: int):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(MAX_FREQ) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def normalize(in_channels: int):
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6, affine=True)


def nonlinearity(name: str):
    return getattr(nn, name)()


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_chs: int,
                 out_chs: int,
                 temb_dim: int,
                 act: Optional[str] = 'SiLU',
                 dropout: float = 0.2):
        super().__init__()
        self.time_proj = nn.Sequential(nonlinearity(act),
                                       nn.Linear(temb_dim, out_chs))

        dims = [in_chs] + 2 * [out_chs]
        blocks = []
        for i in range(1, 3):
            blc = nn.Sequential(normalize(dims[i - 1]),
                                nonlinearity(act),
                                nn.Conv2d(dims[i - 1], dims[i], 3, padding=1), )
            if i > 1:
                blc = nn.Sequential(normalize(dims[i - 1]),
                                    nonlinearity(act),
                                    nn.Dropout(dropout),
                                    nn.Conv2d(dims[i - 1], dims[i], 3, padding=1), )
            blocks.append(blc)
        self.blocks = nn.ModuleList(blocks)
        self.short_cut = False
        if in_chs != out_chs:
            self.short_cut = True
            self.conv_short = nn.Conv2d(in_chs, out_chs, 1)

    def forward(self,
                x: torch.Tensor,
                temb: torch.Tensor):
        h = x
        for i, blc in enumerate(self.blocks):
            h = blc(h)
            if i == 0:
                h = h + self.time_proj(temb)[:, :, None, None]
        if self.short_cut:
            x = self.conv_short(x)
        return x + h


# Attention Module
class AttnBlock(nn.Module):
    def __init__(self,
                 in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self,
                x: torch.tensor):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (c ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


# Downsize Block
class DownBlock(nn.Module):
    def __init__(self,
                 out_chs: int,
                 with_conv: Optional[bool] = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.down_conv = nn.Conv2d(out_chs, out_chs, 3, stride=2)
        else:
            self.down_conv = nn.AvgPool2d(2, 2)

    def forward(self,
                x: torch.Tensor):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.down_conv(x)
        else:
            x = self.down_conv(x)
        return x


# Upsample BLock
class UpBlock(nn.Module):
    def __init__(self,
                 out_chs: int,
                 with_conv: Optional[bool] = True,
                 mode: Optional[str] = 'nearest'):
        super().__init__()
        self.with_conv = with_conv
        self.mode = mode
        if with_conv:
            self.up_conv = nn.Conv2d(out_chs, out_chs, 3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode=self.mode)
        if self.with_conv:
            x = self.up_conv(x)
        return x


# Unet Model
class DiffUnet(nn.Module):
    def __init__(self,
                 chs: Optional[int] = 32,
                 chs_mult: List[int] = [2, 2, 4, 4, 8],
                 attn_res: List[int] = [16, 8, 4],
                 block_depth: Optional[int] = 2,
                 act: Optional[str] = 'SiLU',
                 temb_dim: Optional[int] = 256,
                 with_conv: Optional[bool] = True,
                 res: Optional[int] = 64,
                 dropout: Optional[float] = 0.3):
        super().__init__()
        self.chs = chs
        self.conv_in = nn.Conv2d(3, chs, 3, padding=1)
        self.time_proj = nn.Sequential(nn.Linear(chs, temb_dim),
                                       nonlinearity(act),
                                       nn.Linear(temb_dim, temb_dim))
        chs_mult = [1] + chs_mult
        # down block
        down_dims = []  # to store the down features
        downs = []
        for i in range(1, len(chs_mult) - 1):
            in_ch = chs * chs_mult[i - 1]
            out_ch = chs * chs_mult[i]
            down = nn.Module()
            down.res = nn.ModuleList([ResidualBlock(in_ch, out_ch, temb_dim, act, dropout)] +
                                     [ResidualBlock(out_ch, out_ch, temb_dim, act, dropout) for _ in
                                      range(1, block_depth)])
            attn = AttnBlock(out_ch) if res in attn_res else nn.Identity()
            down.attn = attn
            down.down_blc = DownBlock(out_ch, with_conv)
            downs.append(down)
            down_dims.append(out_ch)
            res = res // 2

        self.downs = nn.ModuleList(downs)

        # mid block
        last_ch_dim = chs * chs_mult[-1]
        self.mid_res1 = ResidualBlock(out_ch,
                                      last_ch_dim,
                                      temb_dim, act, dropout)
        self.mid_attn = AttnBlock(last_ch_dim)
        self.mid_res2 = ResidualBlock(last_ch_dim,
                                      last_ch_dim,
                                      temb_dim, act, dropout)

        # up block
        down_dims = down_dims[1:] + [last_ch_dim]
        ups = []
        for i, skip_ch in zip(reversed(range(1, len(chs_mult) - 1)), reversed(down_dims)):
            out_ch = chs * chs_mult[i]
            in_ch = out_ch + skip_ch
            up = nn.Module()

            up.res = nn.ModuleList([ResidualBlock(in_ch, out_ch, temb_dim, act, dropout)] +
                                   [ResidualBlock(out_ch * 2, out_ch, temb_dim, act, dropout) for _ in
                                    range(1, block_depth)])
            attn = AttnBlock(out_ch) if res in attn_res else nn.Identity()
            up.attn = attn
            up.up_blc = UpBlock(skip_ch, with_conv) if i != 0 else nn.Identity()
            ups.append(up)
            res = int(res * 2)
        self.ups = nn.ModuleList(ups)
        self.out = nn.Sequential(normalize(out_ch),
                                 nonlinearity(act),
                                 nn.Conv2d(out_ch, 3, 3, padding=1))
        self.res = res

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor):
        t = get_timestep_embedding(timestep, self.chs)
        t = self.time_proj(t)
        h = self.conv_in(x)
        hs = []
        # Down
        for blc in self.downs:
            for res_block in blc.res:
                h = res_block(h, t)
                h = blc.attn(h)
                hs.append(h)
            h = blc.down_blc(h)
        # Mid
        h = self.mid_res1(h, t)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t)
        # Up
        for blc in self.ups:
            h = blc.up_blc(h)
            for res_block in blc.res:
                h = torch.cat([h, hs.pop()], axis=1)
                h = res_block(h, t)
                h = blc.attn(h)
        return self.out(h)
