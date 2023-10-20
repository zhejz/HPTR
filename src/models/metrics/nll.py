# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from omegaconf import ListConfig
from typing import Dict, Optional
from torch import Tensor, tensor
from torch.nn import functional as F
from torchmetrics.metric import Metric
from torch.distributions import MultivariateNormal


def compute_nll_mtr(dmean: Tensor, cov: Tensor) -> Tensor:
    dx = dmean[..., 0]
    dy = dmean[..., 1]
    sx = cov[..., 0, 0]
    sy = cov[..., 1, 1]
    rho = torch.tanh(cov[..., 1, 0])  # mtr uses clamp to [-0.5, 0.5]
    one_minus_rho2 = 1 - rho ** 2
    log_prob = (
        torch.log(sx)
        + torch.log(sy)
        + 0.5 * torch.log(one_minus_rho2)
        + 0.5 / one_minus_rho2 * ((dx / sx) ** 2 + (dy / sy) ** 2 - 2 * rho * dx * dy / (sx * sy))
    )
    return log_prob


class NllMetrics(Metric):
    full_state_update = False

    def __init__(
        self,
        prefix: str,
        winner_takes_all: str,
        p_rand_train_agent: float,
        n_decoders: int,
        n_pred: int,
        l_pos: str,
        n_step_add_train_agent: ListConfig,
        focal_gamma_conf: ListConfig,
        w_conf: ListConfig,
        w_pos: ListConfig,
        w_yaw: ListConfig,  # cos
        w_spd: ListConfig,  # huber
        w_vel: ListConfig,  # huber
    ) -> None:
        super().__init__(dist_sync_on_step=False)
        self.prefix = prefix
        self.winner_takes_all = winner_takes_all
        self.p_rand_train_agent = p_rand_train_agent
        self.n_decoders = n_decoders
        self.n_pred = n_pred
        self.l_pos = l_pos
        self.n_step_add_train_agent = n_step_add_train_agent
        self.focal_gamma_conf = list(focal_gamma_conf)
        self.w_conf = list(w_conf)
        self.w_pos = list(w_pos)
        self.w_yaw = list(w_yaw)
        self.w_spd = list(w_spd)
        self.w_vel = list(w_vel)

        self.add_state("counter_traj", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counter_conf", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_pos", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_conf", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_yaw", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_spd", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_vel", default=tensor(0.0), dist_reduce_fx="sum")

        for i in range(self.n_decoders):
            for j in range(self.n_pred):
                self.add_state(f"counter_d{i}_p{j}", default=tensor(0.0), dist_reduce_fx="sum")
                self.add_state(f"conf_d{i}_p{j}", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_valid: Tensor,
        pred_conf: Tensor,
        pred_pos: Tensor,
        pred_spd: Optional[Tensor],
        pred_vel: Optional[Tensor],
        pred_yaw_bbox: Optional[Tensor],
        pred_cov: Optional[Tensor],
        ref_role: Tensor,
        ref_type: Tensor,
        gt_valid: Tensor,
        gt_pos: Tensor,
        gt_spd: Tensor,
        gt_vel: Tensor,
        gt_yaw_bbox: Tensor,
        gt_cmd: Tensor,
        **kwargs,
    ) -> None:
        """
        Args:
            pred_valid: [n_scene, n_agent], bool
            pred_conf: [n_decoder, n_scene, n_agent, n_pred], not normalized!
            pred_pos: [n_decoder, n_scene, n_agent, n_pred, n_step_future, 2]
            pred_spd: [n_decoder, n_scene, n_agent, n_pred, n_step_future, 1]
            pred_vel: [n_decoder, n_scene, n_agent, n_pred, n_step_future, 2]
            pred_yaw_bbox: [n_decoder, n_scene, n_agent, n_pred, n_step_future, 1]
            pred_cov: [n_decoder, n_scene, n_agent, n_pred, n_step_future, 2, 2]
            gt_valid: [n_scene, n_agent, n_step_future], bool
            gt_pos: [n_scene, n_agent, n_step_future, 2]
            gt_spd: [n_scene, n_agent, n_step_future, 1]
            gt_vel: [n_scene, n_agent, n_step_future, 2]
            gt_yaw_bbox: [n_scene, n_agent, n_step_future, 1]
            ref_role: [n_scene, n_agent, 3], one hot bool [sdc=0, interest=1, predict=2]
            ref_type: [n_scene, n_agent, 3], one hot bool [veh=0, ped=1, cyc=2]
            agent_cmd: [n_scene, n_agent, 8], one hot bool
        """
        n_agent_type = ref_type.shape[-1]
        n_decoder, n_scene, n_agent, n_pred = pred_conf.shape
        assert (ref_role.any(-1) & pred_valid == ref_role.any(-1)).all(), "All relevat agents shall be predicted!"

        # ! prepare avails
        avails = ref_role.any(-1)  # [n_scene, n_agent]
        # add rand agents for training
        if self.p_rand_train_agent > 0:
            avails = avails | (torch.bernoulli(self.p_rand_train_agent * torch.ones_like(avails)).bool())
        # add long tracked agents for training
        _track_len = gt_valid.sum(-1)  # [n_scene, n_agent]
        for i in range(n_agent_type):
            if self.n_step_add_train_agent[i] > 0:
                avails = avails | (ref_type[:, :, i] & (_track_len > self.n_step_add_train_agent[i]))

        avails = gt_valid & avails.unsqueeze(-1)  # [n_scene, n_agent, n_step_future]
        avails = avails.unsqueeze(0).expand(n_decoder, -1, -1, -1)  # [n_decoder, n_scene, n_agent, n_step_future]
        if n_decoder > 1:
            # [n_decoder], randomly train ensembles with 50% of chance
            mask_ensemble = torch.bernoulli(0.5 * torch.ones_like(pred_conf[:, 0, 0, 0])).bool()
            # make sure at least one ensemble is trained
            if not mask_ensemble.any():
                mask_ensemble[torch.randint(0, n_decoder, (1,))] |= True
            avails = avails & mask_ensemble[:, None, None, None]
        # [n_decoder, n_scene, n_agent, n_pred, n_step_future]
        avails = avails.unsqueeze(3).expand(-1, -1, -1, n_pred, -1)

        # ! normalize pred_conf
        # [n_decoder, n_scene, n_agent, n_pred], per ensemble
        pred_conf = torch.softmax(pred_conf, dim=-1)

        # ! save conf histogram
        _prob = pred_conf.masked_fill(~(pred_valid[None, :, :, None]), 0.0)
        for i in range(self.n_decoders):
            for j in range(self.n_pred):
                x = getattr(self, f"conf_d{i}_p{j}")
                x += (_prob[i, :, :, j] * (avails[i, :, :, j].any(-1))).sum()

        # ! winnter takes all
        with torch.no_grad():
            decoder_idx = torch.arange(n_decoder)[:, None, None, None]  # [n_decoder, 1, 1, 1]
            scene_idx = torch.arange(n_scene)[None, :, None, None]  # [1, n_scene, 1, 1]
            agent_idx = torch.arange(n_agent)[None, None, :, None]  # [1, 1, n_agent, 1]

            if "hard" in self.winner_takes_all:
                # [n_decoder, n_scene, n_agent, n_pred, n_step_future]
                dist = torch.norm(pred_pos - gt_pos[None, :, :, None, :, :], dim=-1)
                dist = dist.masked_fill(~avails, 0.0).sum(-1)  # [n_decoder, n_scene, n_agent, n_pred]
                if "joint" in self.winner_takes_all:
                    dist = dist.sum(2, keepdim=True)  # [n_decoder, n_scene, 1, n_pred]
                k_top = int(self.winner_takes_all[-1])
                i = torch.randint(high=k_top, size=())
                # [n_decoder, n_scene, n_agent, 1]
                mode_idx = dist.topk(k_top, dim=-1, largest=False, sorted=False)[1][..., [i]]
            elif self.winner_takes_all == "cmd":
                assert n_pred == gt_cmd.shape[-1]
                mode_idx = (gt_cmd + 0.0).argmax(-1, keepdim=True)  # [n_scene, n_agent, 1]
                mode_idx = mode_idx.unsqueeze(0).expand(n_decoder, -1, -1, -1)  # [n_decoder, n_scene, n_agent, 1]

            # ! save hard assignment histogram: [n_decoder, n_scene, n_agent, n_pred]
            counter_modes = torch.nn.functional.one_hot(mode_idx.squeeze(-1), self.n_pred)
            for i in range(self.n_decoders):
                for j in range(self.n_pred):
                    x = getattr(self, f"counter_d{i}_p{j}")
                    x += (counter_modes[i, :, :, j] * (avails[i, :, :, j].any(-1))).sum()

        # ! avails and counter
        # avails: [n_decoder, n_scene, n_agent, n_pred, n_step_future]
        avails = avails[decoder_idx, scene_idx, agent_idx, mode_idx]
        self.counter_traj += avails.sum()
        self.counter_conf += avails[:, :, :, 0, :].any(-1).sum()

        # ! prepare agent dependent loss weights
        focal_gamma_conf, w_conf, w_pos, w_yaw, w_spd, w_vel = 0, 0, 0, 0, 0, 0
        for i in range(n_agent_type):  # [n_scene, n_agent]
            focal_gamma_conf += ref_type[:, :, i] * self.focal_gamma_conf[i]
            w_conf += ref_type[:, :, i] * self.w_conf[i]
            w_pos += ref_type[:, :, i] * self.w_pos[i]
            w_yaw += ref_type[:, :, i] * self.w_yaw[i]
            w_spd += ref_type[:, :, i] * self.w_spd[i]
            w_vel += ref_type[:, :, i] * self.w_vel[i]

        # ! error_conf
        # pred_conf: [n_decoder, n_scene, n_agent, n_pred], not normalized!
        pred_conf = pred_conf[decoder_idx, scene_idx, agent_idx, mode_idx]
        focal_gamma_conf = torch.pow(1 - pred_conf, focal_gamma_conf[None, :, :, None])
        w_conf = w_conf[None, :, :, None]
        self.error_conf += (-torch.log(pred_conf) * w_conf * focal_gamma_conf).masked_fill(~(avails.any(-1)), 0.0).sum()

        # ! error_pos
        pred_pos = pred_pos[decoder_idx, scene_idx, agent_idx, mode_idx]
        if self.l_pos == "huber":
            errors_pos = F.huber_loss(pred_pos, gt_pos[None, :, :, None, :, :], reduction="none").sum(-1)
        elif self.l_pos == "l2":
            errors_pos = torch.norm(pred_pos - gt_pos[None, :, :, None, :, :], p=2, dim=-1)
        elif self.l_pos == "nll_mtr":
            pred_cov = pred_cov[decoder_idx, scene_idx, agent_idx, mode_idx]
            errors_pos = compute_nll_mtr(pred_pos - gt_pos[None, :, :, None, :, :], pred_cov)
        elif self.l_pos == "nll_torch":
            gmm = MultivariateNormal(pred_pos, scale_tril=pred_cov[decoder_idx, scene_idx, agent_idx, mode_idx])
            errors_pos = -gmm.log_prob(gt_pos[None, :, :, None, :, :])
        self.error_pos += (errors_pos * w_pos[None, :, :, None, None]).masked_fill(~avails, 0.0).sum()

        # ! error_spd
        if sum(self.w_spd) > 0 and pred_spd is not None:
            pred_spd = pred_spd[decoder_idx, scene_idx, agent_idx, mode_idx]
            errors_spd = F.huber_loss(pred_spd, gt_spd[None, :, :, None, :, :], reduction="none").squeeze(-1)
            self.error_spd += (errors_spd * w_spd[None, :, :, None, None]).masked_fill(~avails, 0.0).sum()

        # ! error_vel
        if sum(self.w_vel) > 0 and pred_vel is not None:
            pred_vel = pred_vel[decoder_idx, scene_idx, agent_idx, mode_idx]
            errors_vel = F.huber_loss(pred_vel, gt_vel[None, :, :, None, :, :], reduction="none").sum(-1)
            self.error_vel += (errors_vel * w_vel[None, :, :, None, None]).masked_fill(~avails, 0.0).sum()

        # ! error_yaw
        if sum(self.w_yaw) > 0 and pred_yaw_bbox is not None:
            pred_yaw_bbox = pred_yaw_bbox[decoder_idx, scene_idx, agent_idx, mode_idx]
            errors_yaw = -torch.cos(pred_yaw_bbox - gt_yaw_bbox[None, :, :, None, :, :]).squeeze(-1)
            self.error_yaw += (errors_yaw * w_yaw[None, :, :, None, None]).masked_fill(~avails, 0.0).sum()

    def compute(self) -> Dict[str, Tensor]:

        out_dict = {
            f"{self.prefix}/counter_traj": self.counter_traj,
            f"{self.prefix}/counter_conf": self.counter_conf,
            f"{self.prefix}/error_pos": self.error_pos,
            f"{self.prefix}/error_conf": self.error_conf,
            f"{self.prefix}/error_yaw": self.error_yaw,
            f"{self.prefix}/error_spd": self.error_spd,
            f"{self.prefix}/error_vel": self.error_vel,
        }
        out_dict[f"{self.prefix}/loss"] = (
            self.error_pos + self.error_yaw + self.error_spd + self.error_vel
        ) / self.counter_traj + self.error_conf / self.counter_conf

        for i in range(self.n_decoders):
            for j in range(self.n_pred):
                out_dict[f"{self.prefix}/counter_d{i}_p{j}"] = getattr(self, f"counter_d{i}_p{j}")
                out_dict[f"{self.prefix}/conf_d{i}_p{j}"] = getattr(self, f"conf_d{i}_p{j}")

        return out_dict
