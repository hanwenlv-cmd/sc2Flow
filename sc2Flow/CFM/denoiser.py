import torch
import torch.nn as nn
from model import sc2Flow
from tqdm import tqdm
from utils import list2vector
import matplotlib.pyplot as plt
class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = sc2Flow()

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels, step=None):

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e # xt
        v = (x - z) / (1 - t).clamp_min(self.t_eps) # target v

        x_pred = self.net(x=z, t=t.flatten(), labels=labels) # directly predict x
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps) # but use v to calculate loss

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1)).mean()

        return loss

    @torch.no_grad()
    def generate(self, labels):
        g_id = labels['g_id']
        device = g_id.device
        bsz,l = g_id.size(0), g_id.size(1)
        z = torch.randn(bsz,l,device=device) * self.noise_scale
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
            if i % int(self.steps/5)==0:
                list2vector(g_id[0].cpu(), z[0].cpu(), title=t[0],y_lim=(-0.5,6.1),i=t[0]) #visualize
        # last step euler
        list2vector(g_id[0].cpu(), z[0].cpu(), title=t[0], y_lim=(-0.5, 6.1), i=t[0])
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        # conditional
        x_cond = self.net(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        return v_cond

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
