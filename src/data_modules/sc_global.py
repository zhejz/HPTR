# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict
from omegaconf import DictConfig
import torch
from torch import nn, Tensor
from utils.pose_pe import PosePE


class SceneCentricGlobal(nn.Module):
    def __init__(
        self,
        time_step_current: int,
        data_size: DictConfig,
        dropout_p_history: float,
        use_current_tl: bool,
        add_ohe: bool,
        pl_aggr: bool,
        pose_pe: DictConfig,
    ) -> None:
        super().__init__()
        self.dropout_p_history = dropout_p_history  # [0, 1], turn off if set to negative
        self.step_current = time_step_current
        self.n_step_hist = time_step_current + 1
        self.use_current_tl = use_current_tl
        self.add_ohe = add_ohe
        self.pl_aggr = pl_aggr
        self.n_pl_node = data_size["map/valid"][-1]

        self.pose_pe_agent = PosePE(pose_pe["agent"])
        self.pose_pe_map = PosePE(pose_pe["map"])
        self.pose_pe_tl = PosePE(pose_pe["tl"])

        tl_attr_dim = self.pose_pe_tl.out_dim + data_size["tl_stop/state"][-1]
        if self.pl_aggr:
            agent_attr_dim = (
                self.pose_pe_agent.out_dim * self.n_step_hist
                + data_size["agent/spd"][-1] * self.n_step_hist  # 1
                + data_size["agent/vel"][-1] * self.n_step_hist  # 2
                + data_size["agent/yaw_rate"][-1] * self.n_step_hist  # 1
                + data_size["agent/acc"][-1] * self.n_step_hist  # 1
                + data_size["agent/size"][-1]  # 3
                + data_size["agent/type"][-1]  # 3
                + self.n_step_hist  # valid
            )
            map_attr_dim = self.pose_pe_map.out_dim * self.n_pl_node + data_size["map/type"][-1] + self.n_pl_node
        else:
            agent_attr_dim = (
                self.pose_pe_agent.out_dim
                + data_size["agent/spd"][-1]  # 1
                + data_size["agent/vel"][-1]  # 2
                + data_size["agent/yaw_rate"][-1]  # 1
                + data_size["agent/acc"][-1]  # 1
                + data_size["agent/size"][-1]  # 3
                + data_size["agent/type"][-1]  # 3
            )
            map_attr_dim = self.pose_pe_map.out_dim + data_size["map/type"][-1]

        if self.add_ohe:
            self.register_buffer("history_step_ohe", torch.eye(self.n_step_hist))
            self.register_buffer("pl_node_ohe", torch.eye(self.n_pl_node))
            if not self.pl_aggr:
                map_attr_dim += self.n_pl_node
                agent_attr_dim += self.n_step_hist
            if not self.use_current_tl:
                tl_attr_dim += self.n_step_hist

        self.model_kwargs = {
            "agent_attr_dim": agent_attr_dim,
            "map_attr_dim": map_attr_dim,
            "tl_attr_dim": tl_attr_dim,
            "n_step_hist": self.n_step_hist,
            "n_pl_node": self.n_pl_node,
            "use_current_tl": self.use_current_tl,
            "pl_aggr": self.pl_aggr,
        }

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args: scene-centric Dict
            # (ref) reference information for transform back to global coordinate and submission to waymo
                "ref/pos": [n_scene, n_agent, 1, 2]
                "ref/yaw": [n_scene, n_agent, 1]
                "ref/rot": [n_scene, n_agent, 2, 2]
                "ref/role": [n_scene, n_agent, 3]
                "ref/type": [n_scene, n_agent, 3]
            # (gt) ground-truth agent future for training, not available for testing
                "gt/valid": [n_scene, n_agent, n_step_future], bool
                "gt/pos": [n_scene, n_agent, n_step_future, 2]
                "gt/spd": [n_scene, n_agent, n_step_future, 1]
                "gt/vel": [n_scene, n_agent, n_step_future, 2]
                "gt/yaw_bbox": [n_scene, n_agent, n_step_future, 1]
                "gt/cmd": [n_scene, n_agent, 8]
            # (sc) scene-centric agents states
                "sc/agent_valid": [n_scene, n_agent, n_step_hist]
                "sc/agent_pos": [n_scene, n_agent, n_step_hist, 2]
                "sc/agent_vel": [n_scene, n_agent, n_step_hist, 2]
                "sc/agent_spd": [n_scene, n_agent, n_step_hist, 1]
                "sc/agent_acc": [n_scene, n_agent, n_step_hist, 1]
                "sc/agent_yaw_bbox": [n_scene, n_agent, n_step_hist, 1]
                "sc/agent_yaw_rate": [n_scene, n_agent, n_step_hist, 1]
            # agent attributes
                "sc/agent_type": [n_scene, n_agent, 3]
                "sc/agent_role": [n_scene, n_agent, 3]
                "sc/agent_size": [n_scene, n_agent, 3]
            # map polylines
                "sc/map_valid": [n_scene, n_pl, n_pl_node], bool
                "sc/map_type": [n_scene, n_pl, 11], bool one_hot
                "sc/map_pos": [n_scene, n_pl, n_pl_node, 2], float32
                "sc/map_dir": [n_scene, n_pl, n_pl_node, 2], float32
            # traffic lights
                "sc/tl_valid": [n_scene, n_step_hist, n_tl], bool
                "sc/tl_state": [n_scene, n_step_hist, n_tl, 5], bool one_hot
                "sc/tl_pos": [n_scene, n_step_hist, n_tl, 2], x,y
                "sc/tl_dir": [n_scene, n_step_hist, n_tl, 2], x,y

        Returns: add following keys to batch Dict
            # agent type: no need to be aggregated.
                "input/agent_type": [n_scene, n_agent, 3]
            # agent history
                if pl_aggr: # processed by mlp encoder, c.f. our method.
                    "input/agent_valid": [n_scene, n_agent], bool
                    "input/agent_attr": [n_scene, n_agent, agent_attr_dim]
                    "input/agent_pos": [n_scene, n_agent, 2], (x,y), pos of the current step.
                    "input/map_valid": [n_scene, n_pl], bool
                    "input/map_attr": [n_scene, n_pl, map_attr_dim]
                    "input/map_pos": [n_scene, n_pl, 2], (x,y), polyline starting node in global coordinate.
                else: # processed by pointnet encoder, c.f. vectornet.
                    "input/agent_valid": [n_scene, n_agent, n_step_hist], bool
                    "input/agent_attr": [n_scene, n_agent, n_step_hist, agent_attr_dim]
                    "input/agent_pos": [n_scene, n_agent, 2]
                    "input/map_valid": [n_scene, n_pl, n_pl_node], bool
                    "input/map_attr": [n_scene, n_pl, n_pl_node, map_attr_dim]
                    "input/"map_pos"": [n_scene, n_pl, 2]
            # traffic lights: stop point, cannot be aggregated, detections are not tracked, singular node polyline.
                if use_current_tl:
                    "input/tl_valid": [n_scene, 1, n_tl], bool
                    "input/tl_attr": [n_scene, 1, n_tl, tl_attr_dim]
                    "input/tl_pos": [n_scene, 1, n_tl, 2] (x,y)
                else:
                    "input/tl_valid": [n_scene, n_step_hist, n_tl], bool
                    "input/tl_attr": [n_scene, n_step_hist, n_tl, tl_attr_dim]
                    "input/tl_pos": [n_scene, n_step_hist, n_tl, 2]
        """
        batch["input/agent_type"] = batch["sc/agent_type"]
        batch["input/agent_valid"] = batch["sc/agent_valid"]
        batch["input/tl_valid"] = batch["sc/tl_valid"]
        batch["input/map_valid"] = batch["sc/map_valid"]

        # ! randomly mask history agent/tl/map
        if self.training and (0 < self.dropout_p_history <= 1.0):
            prob_mask = torch.ones_like(batch["input/agent_valid"][..., :-1]) * (1 - self.dropout_p_history)
            batch["input/agent_valid"][..., :-1] &= torch.bernoulli(prob_mask).bool()
            prob_mask = torch.ones_like(batch["input/tl_valid"]) * (1 - self.dropout_p_history)
            batch["input/tl_valid"] &= torch.bernoulli(prob_mask).bool()
            prob_mask = torch.ones_like(batch["input/map_valid"]) * (1 - self.dropout_p_history)
            batch["input/map_valid"] &= torch.bernoulli(prob_mask).bool()

        # ! prepare "input/agent"
        batch["input/agent_pos"] = batch["ref/pos"].squeeze(2)
        if self.pl_aggr:  # [n_scene, n_agent, agent_attr_dim]
            agent_invalid = ~batch["input/agent_valid"].unsqueeze(-1)  # [n_scene, n_agent, n_step_hist, 1]
            agent_invalid_reduced = agent_invalid.all(-2)  # [n_scene, n_agent, 1]
            batch["input/agent_attr"] = torch.cat(
                [
                    self.pose_pe_agent(batch["sc/agent_pos"], batch["sc/agent_yaw_bbox"])
                    .masked_fill(agent_invalid, 0)
                    .flatten(-2, -1),
                    batch["sc/agent_vel"].masked_fill(agent_invalid, 0).flatten(-2, -1),  # n_step_hist*2
                    batch["sc/agent_spd"].masked_fill(agent_invalid, 0).squeeze(-1),  # n_step_hist
                    batch["sc/agent_yaw_rate"].masked_fill(agent_invalid, 0).squeeze(-1),  # n_step_hist
                    batch["sc/agent_acc"].masked_fill(agent_invalid, 0).squeeze(-1),  # n_step_hist
                    batch["sc/agent_size"].masked_fill(agent_invalid_reduced, 0),  # 3
                    batch["sc/agent_type"].masked_fill(agent_invalid_reduced, 0),  # 3
                    batch["input/agent_valid"],  # n_step_hist
                ],
                dim=-1,
            )
            batch["input/agent_valid"] = batch["input/agent_valid"].any(-1)  # [n_scene, n_agent]
        else:  # [n_scene, n_agent, n_step_hist, agent_attr_dim]
            batch["input/agent_attr"] = torch.cat(
                [
                    self.pose_pe_agent(batch["sc/agent_pos"], batch["sc/agent_yaw_bbox"]),
                    batch["sc/agent_vel"],  # vel xy, 2
                    batch["sc/agent_spd"],  # speed, 1
                    batch["sc/agent_yaw_rate"],  # yaw rate, 1
                    batch["sc/agent_acc"],  # acc, 1
                    batch["sc/agent_size"].unsqueeze(-2).expand(-1, -1, self.n_step_hist, -1),  # 3
                    batch["sc/agent_type"].unsqueeze(-2).expand(-1, -1, self.n_step_hist, -1),  # 3
                ],
                dim=-1,
            )

        # ! prepare "input/map_attr": [n_scene, n_pl, n_pl_node, map_attr_dim]
        batch["input/map_pos"] = batch["sc/map_pos"][:, :, 0]
        if self.pl_aggr:  # [n_scene, n_pl, map_attr_dim]
            map_invalid = ~batch["input/map_valid"].unsqueeze(-1)  # [n_scene, n_pl, n_pl_node, 1]
            map_invalid_reduced = map_invalid.all(-2)  # [n_scene, n_pl, 1]
            batch["input/map_attr"] = torch.cat(
                [
                    self.pose_pe_map(batch["sc/map_pos"], batch["sc/map_dir"])
                    .masked_fill(map_invalid, 0)
                    .flatten(-2, -1),
                    batch["sc/map_type"].masked_fill(map_invalid_reduced, 0),  # n_pl_type
                    batch["input/map_valid"],  # n_pl_node
                ],
                dim=-1,
            )
            batch["input/map_valid"] = batch["input/map_valid"].any(-1)  # [n_scene, n_pl]
        else:  # [n_scene, n_pl, n_pl_node, map_attr_dim]
            batch["input/map_attr"] = torch.cat(
                [
                    self.pose_pe_map(batch["sc/map_pos"], batch["sc/map_dir"]),  # pl_dim
                    batch["sc/map_type"].unsqueeze(-2).expand(-1, -1, self.n_pl_node, -1),  # n_pl_type
                ],
                dim=-1,
            )

        # ! prepare "input/tl_attr": [n_scene, n_step_hist/1, n_tl, tl_attr_dim]
        # [n_scene, n_step_hist, n_tl, 2]
        tl_pos = batch["sc/tl_pos"]
        tl_dir = batch["sc/tl_dir"]
        tl_state = batch["sc/tl_state"]
        if self.use_current_tl:
            tl_pos = tl_pos[:, [-1]]  # [n_scene, 1, n_tl, 2]
            tl_dir = tl_dir[:, [-1]]  # [n_scene, 1, n_tl, 2]
            tl_state = tl_state[:, [-1]]  # [n_scene, 1, n_tl, 5]
            batch["input/tl_valid"] = batch["input/tl_valid"][:, [-1]]  # [n_scene, 1, n_tl]
        batch["input/tl_attr"] = torch.cat([self.pose_pe_tl(tl_pos, tl_dir), tl_state], dim=-1)
        batch["input/tl_pos"] = tl_pos
        # ! add one-hot encoding for sequence (temporal, order of polyline nodes)
        if self.add_ohe:
            n_scene, n_agent, _ = batch["sc/agent_valid"].shape
            n_pl = batch["sc/map_valid"].shape[1]
            if not self.pl_aggr:  # there is no need to add ohe if self.pl_aggr
                batch["input/agent_attr"] = torch.cat(
                    [
                        batch["input/agent_attr"],
                        self.history_step_ohe[None, None, :, :].expand(n_scene, n_agent, -1, -1),
                    ],
                    dim=-1,
                )
                batch["input/map_attr"] = torch.cat(
                    [batch["input/map_attr"], self.pl_node_ohe[None, None, :, :].expand(n_scene, n_pl, -1, -1),],
                    dim=-1,
                )

            if not self.use_current_tl:  # there is no need to add ohe if use_current_tl
                n_tl = batch["input/tl_valid"].shape[-1]
                batch["input/tl_attr"] = torch.cat(
                    [batch["input/tl_attr"], self.history_step_ohe[None, :, None, :].expand(n_scene, -1, n_tl, -1)],
                    dim=-1,
                )

        return batch
