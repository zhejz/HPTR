# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict
from omegaconf import DictConfig
import torch
from torch import nn, Tensor
from utils.transform_utils import torch_rad2rot, torch_pos2local, torch_dir2local, torch_rad2local


class AgentCentricPreProcessing(nn.Module):
    def __init__(
        self,
        time_step_current: int,
        data_size: DictConfig,
        n_target: int,
        n_other: int,
        n_map: int,
        n_tl: int,
        mask_invalid: bool,
    ) -> None:
        super().__init__()
        self.step_current = time_step_current
        self.n_step_hist = time_step_current + 1
        self.n_target = n_target
        self.n_other = n_other
        self.n_map = n_map
        self.n_tl = n_tl
        self.mask_invalid = mask_invalid
        self.model_kwargs = {"gt_in_local": True, "agent_centric": True}

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args: scene-centric Dict
            # agent states
                "agent/valid": [n_scene, n_step, n_agent], bool,
                "agent/pos": [n_scene, n_step, n_agent, 2], float32
                "agent/vel": [n_scene, n_step, n_agent, 2], float32, v_x, v_y
                "agent/spd": [n_scene, n_step, n_agent, 1], norm of vel, signed using yaw_bbox and vel_xy
                "agent/acc": [n_scene, n_step, n_agent, 1], m/s2, acc[t] = (spd[t]-spd[t-1])/dt
                "agent/yaw_bbox": [n_scene, n_step, n_agent, 1], float32, yaw of the bbox heading
                "agent/yaw_rate": [n_scene, n_step, n_agent, 1], rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
            # agent attributes
                "agent/type": [n_scene, n_agent, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
                "agent/role": [n_scene, n_agent, 3], bool [sdc=0, interest=1, predict=2]
                "agent/size": [n_scene, n_agent, 3], float32: [length, width, height]
            # map polylines
                "map/valid": [n_scene, n_pl, n_pl_node], bool
                "map/type": [n_scene, n_pl, 11], bool one_hot
                "map/pos": [n_scene, n_pl, n_pl_node, 2], float32
                "map/dir": [n_scene, n_pl, n_pl_node, 2], float32
            # traffic lights
                "tl_stop/valid": [n_scene, n_step, n_tl_stop], bool
                "tl_stop/state": [n_scene, n_step, n_tl_stop, 5], bool one_hot
                "tl_stop/pos": [n_scene, n_step, n_tl_stop, 2], x,y
                "tl_stop/dir": [n_scene, n_step, n_tl_stop, 2], x,y

        Returns: agent-centric Dict, masked according to valid
            # (ref) reference information for transform back to global coordinate and submission to waymo
                "ref/pos": [n_scene, n_target, 1, 2]
                "ref/rot": [n_scene, n_target, 2, 2]
                "ref/idx": [n_scene, n_target]
                "ref/idx_n": int, original number of agents
                "ref/role": [n_scene, n_target, 3]
                "ref/type": [n_scene, n_target, 3]
            # (gt) ground-truth target future for training, not available for testing
                "gt/valid": [n_scene, n_target, n_step_future], bool
                "gt/pos": [n_scene, n_target, n_step_future, 2]
                "gt/spd": [n_scene, n_target, n_step_future, 1]
                "gt/vel": [n_scene, n_target, n_step_future, 2]
                "gt/yaw_bbox": [n_scene, n_target, n_step_future, 1]
                "gt/cmd": [n_scene, n_target, 8]
            # (ac) agent-centric target agents states
                "ac/target_valid": [n_scene, n_target, n_step_hist]
                "ac/target_pos": [n_scene, n_target, n_step_hist, 2]
                "ac/target_vel": [n_scene, n_target, n_step_hist, 2]
                "ac/target_spd": [n_scene, n_target, n_step_hist, 1]
                "ac/target_acc": [n_scene, n_target, n_step_hist, 1]
                "ac/target_yaw_bbox": [n_scene, n_target, n_step_hist, 1]
                "ac/target_yaw_rate": [n_scene, n_target, n_step_hist, 1]
            # target agents attributes
                "ac/target_type": [n_scene, n_target, 3]
                "ac/target_role": [n_scene, n_target, 3]
                "ac/target_size": [n_scene, n_target, 3]
            # other agents states
                "ac/other_valid": [n_scene, n_target, n_other, n_step_hist]
                "ac/other_pos": [n_scene, n_target, n_other, n_step_hist, 2]
                "ac/other_vel": [n_scene, n_target, n_other, n_step_hist, 2]
                "ac/other_spd": [n_scene, n_target, n_other, n_step_hist, 1]
                "ac/other_acc": [n_scene, n_target, n_other, n_step_hist, 1]
                "ac/other_yaw_bbox": [n_scene, n_target, n_other, n_step_hist, 1]
                "ac/other_yaw_rate": [n_scene, n_target, n_other, n_step_hist, 1]
            # other agents attributes
                "ac/other_type": [n_scene, n_target, n_other, 3]
                "ac/other_role": [n_scene, n_target, n_other, 3]
                "ac/other_size": [n_scene, n_target, n_other, 3]
            # map polylines
                "ac/map_valid": [n_scene, n_target, n_map, n_pl_node], bool
                "ac/map_type": [n_scene, n_target, n_map, 11], bool one_hot
                "ac/map_pos": [n_scene, n_target, n_map, n_pl_node, 2], float32
                "ac/map_dir": [n_scene, n_target, n_map, n_pl_node, 2], float32
            # traffic lights
                "ac/tl_valid": [n_scene, n_target, n_step_hist, n_tl], bool
                "ac/tl_state": [n_scene, n_target, n_step_hist, n_tl, 5], bool one_hot
                "ac/tl_pos": [n_scene, n_target, n_step_hist, n_tl, 2], x,y
                "ac/tl_dir": [n_scene, n_target, n_step_hist, n_tl, 2], x,y
        """
        prefix = "" if self.training else "history/"
        n_scene = batch[prefix + "agent/valid"].shape[0]

        # ! find target agents
        if self.training:
            target_weights = batch[prefix + "agent/role"].sum(-1) + batch[prefix + "agent/valid"][:, self.step_current]
        else:
            target_weights = (
                batch[prefix + "agent/role"][..., -1] * 10
                + batch[prefix + "agent/role"].sum(-1)
                + batch[prefix + "agent/valid"][:, self.step_current]
            )
        target_indices = torch.topk(target_weights, self.n_target, largest=True, dim=-1)[1]
        scene_indices = torch.arange(n_scene).unsqueeze(1)
        batch["ref/idx"] = target_indices  # [n_scene, n_target]
        batch["ref/idx_n"] = batch[prefix + "agent/valid"].shape[-1]

        # ! get ref pos/yaw/rot for global to local transform
        ref_pos = batch[prefix + "agent/pos"][:, self.step_current, :][scene_indices, target_indices].unsqueeze(2)
        ref_yaw = batch[prefix + "agent/yaw_bbox"][:, self.step_current, :, 0][scene_indices, target_indices]
        ref_rot = torch_rad2rot(ref_yaw)
        batch["ref/pos"] = ref_pos  # [n_scene, n_target, 1, 2]
        batch["ref/rot"] = ref_rot  # [n_scene, n_target, 2, 2]

        # [n_scene, n_agent, :] -> [n_scene, n_target, :]
        batch["ref/type"] = batch[prefix + "agent/type"][scene_indices, target_indices]
        batch["ref/role"] = batch[prefix + "agent/role"][scene_indices, target_indices]

        # ! prepare target agents states
        # [n_scene, n_step, n_agent, ...] -> [n_scene, n_agent, n_step, ...] -> [n_scene, n_target, n_step_hist, ...]
        for k in ("valid", "pos", "vel", "spd", "acc", "yaw_bbox", "yaw_rate"):
            batch[f"ac/target_{k}"] = batch[f"{prefix}agent/{k}"][:, : self.n_step_hist].transpose(1, 2)[
                scene_indices, target_indices
            ]
        # [n_scene, n_target, n_step_hist, 2]
        batch["ac/target_pos"] = torch_pos2local(batch["ac/target_pos"], ref_pos, ref_rot)
        batch["ac/target_vel"] = torch_dir2local(batch["ac/target_vel"], ref_rot)
        # [n_scene, n_target, n_step_hist, 1]
        batch["ac/target_yaw_bbox"] = torch_rad2local(batch["ac/target_yaw_bbox"], ref_yaw.unsqueeze(-1), cast=False)

        # ! prepare target agents attributes
        # [n_scene, n_agent, :] -> [n_scene, n_target, :]
        for k in ("type", "role", "size"):
            batch[f"ac/target_{k}"] = batch[f"{prefix}agent/{k}"][scene_indices, target_indices]

        # ! training/validation time, prepare "gt/" for losses
        # [n_scene, n_step, n_agent, ...] -> [n_scene, n_agent, n_step, ...] -> [n_scene, n_target, n_step_future, ...]
        if "agent/valid" in batch.keys():
            for k in ("valid", "spd", "pos", "vel", "yaw_bbox"):
                batch[f"gt/{k}"] = batch[f"agent/{k}"][:, self.n_step_hist :].transpose(1, 2)[
                    scene_indices, target_indices
                ]
            # [n_scene, n_target, n_step_hist, 2]
            batch["gt/pos"] = torch_pos2local(batch["gt/pos"], ref_pos, ref_rot)
            batch["gt/vel"] = torch_dir2local(batch["gt/vel"], ref_rot)
            # [n_scene, n_target, n_step_hist, 1]
            batch["gt/yaw_bbox"] = torch_rad2local(batch["gt/yaw_bbox"], ref_yaw.unsqueeze(-1), cast=False)
            # [n_scene, n_agent, :] -> [n_scene, n_target, :]
            batch["gt/cmd"] = batch["agent/cmd"][scene_indices, target_indices]

        # ! prepare scene_indices and target_indices for other agents
        # [n_scene, n_step, n_agent] -> [n_scene, 1, n_agent, n_step_hist]
        other_valid = batch[prefix + "agent/valid"][:, : self.n_step_hist].transpose(1, 2).unsqueeze(1)
        other_valid = other_valid.repeat(1, self.n_target, 1, 1)  # [n_scene, n_target, n_agent, n_step_hist]
        # remove ego: target_indices [n_scene, n_target]
        for _s in range(n_scene):
            for _t in range(self.n_target):
                other_valid[_s, _t, target_indices[_s, _t]] = False
        # target_pos: [n_scene, n_target, 1, 2], batch["agent/pos"]: [n_scene, n_step, n_agent, 2]
        other_dist = torch.norm(batch[prefix + "agent/pos"][:, [self.step_current]] - ref_pos, dim=-1)
        other_dist.masked_fill_(~other_valid[..., self.step_current], float("inf"))  # [n_scene, n_target, n_agent]
        # [n_scene, n_target, n_other]
        other_dist, other_indices = torch.topk(other_dist, self.n_other, largest=False, dim=-1)
        other_scene_indices = torch.arange(n_scene)[:, None, None]  # [n_scene, 1, 1]
        other_target_indices = torch.arange(self.n_target)[None, :, None]  # [1, n_target, 1]

        # ! prepare other agents states
        # [n_scene, n_target, n_agent, n_step_hist] -> [n_scene, n_target, n_other, n_step_hist]
        batch["ac/other_valid"] = other_valid[other_scene_indices, other_target_indices, other_indices]
        batch["ac/other_valid"] = batch["ac/other_valid"] & (other_dist.unsqueeze(-1) < 2e3)
        # [n_scene, n_step, n_agent, :] -> [n_scene, n_target, n_other, n_step_hist, :]
        for k in ("spd", "acc", "yaw_rate", "pos", "vel", "yaw_bbox"):
            batch[f"ac/other_{k}"] = (
                batch[f"{prefix}agent/{k}"][:, : self.n_step_hist]
                .transpose(1, 2)
                .unsqueeze(1)
                .repeat(1, self.n_target, 1, 1, 1)[other_scene_indices, other_target_indices, other_indices]
            )
        # target_pos: [n_scene, n_target, 1, 2], target_rot: [n_scene, n_target, 2, 2], target_yaw: [n_scene, n_target]
        # [n_scene, n_target, n_other, n_step_hist, 2]
        batch["ac/other_pos"] = torch_pos2local(batch["ac/other_pos"], ref_pos.unsqueeze(2), ref_rot.unsqueeze(2))
        batch["ac/other_vel"] = torch_dir2local(batch["ac/other_vel"], ref_rot.unsqueeze(2))
        # [n_scene, n_target, n_other, n_step_hist, 1]
        batch["ac/other_yaw_bbox"] = torch_rad2local(batch["ac/other_yaw_bbox"], ref_yaw[:, :, None, None], cast=False)

        # ! prepare other agents attributes
        # [n_scene, n_agent, :] -> [n_scene, n_target, n_other, :]
        for k in ("type", "role", "size"):
            batch[f"ac/other_{k}"] = (
                batch[f"{prefix}agent/{k}"]
                .unsqueeze(1)
                .repeat(1, self.n_target, 1, 1)[other_scene_indices, other_target_indices, other_indices]
            )

        # ! prepare agent-centric map polylines
        # [n_scene, n_pl, n_pl_node, 2], [n_scene, n_target, 1, 2]
        map_dist = torch.norm(batch["map/pos"][:, :, 0].unsqueeze(1) - ref_pos, dim=-1)
        map_dist.masked_fill_(~batch["map/valid"][..., 0].unsqueeze(1), float("inf"))  # [n_scene, n_target, n_pl]
        map_dist, map_indices = torch.topk(map_dist, self.n_map, largest=False, dim=-1)  # [n_scene, n_target, n_map]

        # [n_scene, n_pl, n_pl_node(20) / n_pl_type(11)] -> [n_scene, n_target, n_map, n_pl_node(20) / n_pl_type(11)]
        for k in ("valid", "type"):
            batch[f"ac/map_{k}"] = (
                batch[f"map/{k}"]
                .unsqueeze(1)
                .repeat(1, self.n_target, 1, 1)[other_scene_indices, other_target_indices, map_indices]
            )
        batch["ac/map_valid"] = batch["ac/map_valid"] & (map_dist.unsqueeze(-1) < 3e3)

        # [n_scene, n_pl, n_pl_node, 2] -> [n_scene, n_target, n_map, n_pl_node, 2]
        for k in ("pos", "dir"):
            batch[f"ac/map_{k}"] = (
                batch[f"map/{k}"]
                .unsqueeze(1)
                .repeat(1, self.n_target, 1, 1, 1)[other_scene_indices, other_target_indices, map_indices]
            )
        # target_pos: [n_scene, n_target, 1, 2], target_rot: [n_scene, n_target, 2, 2]
        # [n_scene, n_target, n_map, n_pl_node, 2]
        batch["ac/map_pos"] = torch_pos2local(batch["ac/map_pos"], ref_pos.unsqueeze(2), ref_rot.unsqueeze(2))
        batch["ac/map_dir"] = torch_dir2local(batch["ac/map_dir"], ref_rot.unsqueeze(2))

        # ! prepare agent-centric traffic lights
        # [n_scene, n_step_hist, n_tl_stop, 2], [n_scene, n_target, 1, 2]
        tl_dist = torch.norm(
            batch[prefix + "tl_stop/pos"][:, : self.n_step_hist].unsqueeze(1) - ref_pos.unsqueeze(2), dim=-1
        )
        # [n_scene, n_target, n_step_hist, n_tl_stop]
        tl_dist.masked_fill_(~batch[prefix + "tl_stop/valid"][:, : self.n_step_hist].unsqueeze(1), float("inf"))
        # [n_scene, n_target, n_step_hist, n_tl]
        tl_dist, tl_indices = torch.topk(tl_dist, self.n_tl, largest=False, dim=-1)
        tl_scene_indices = torch.arange(n_scene)[:, None, None, None]  # [n_scene, 1, 1, 1]
        tl_target_indices = torch.arange(self.n_target)[None, :, None, None]  # [1, n_target, 1, 1]
        tl_step_indices = torch.arange(tl_indices.shape[2])[None, None, :, None]  # [1, 1, n_target, 1]

        # [n_scene, n_step, n_tl_stop] -> [n_scene, n_target, n_step_hist, n_tl]
        batch["ac/tl_valid"] = (
            batch[prefix + "tl_stop/valid"][:, : self.n_step_hist]
            .unsqueeze(1)
            .repeat(1, self.n_target, 1, 1)[tl_scene_indices, tl_target_indices, tl_step_indices, tl_indices]
        )
        batch["ac/tl_valid"] = batch["ac/tl_valid"] & (tl_dist < 1e3)
        # [n_scene, n_step, n_tl_stop, :] -> [n_scene, n_target, n_step_hist, n_tl, :]
        for k in ("pos", "dir", "state"):
            batch[f"ac/tl_{k}"] = (
                batch[f"{prefix}tl_stop/{k}"][:, : self.n_step_hist]
                .unsqueeze(1)
                .repeat(1, self.n_target, 1, 1, 1)[tl_scene_indices, tl_target_indices, tl_step_indices, tl_indices]
            )
        # target_pos: [n_scene, n_target, 1, 2], target_rot: [n_scene, n_target, 2, 2]
        # [n_scene, n_target, n_step_hist, n_tl, 2]
        batch["ac/tl_pos"] = torch_pos2local(batch["ac/tl_pos"], ref_pos.unsqueeze(2), ref_rot.unsqueeze(2))
        batch["ac/tl_dir"] = torch_dir2local(batch["ac/tl_dir"], ref_rot.unsqueeze(2))

        if self.mask_invalid:
            self.zero_mask_invalid(batch)
        return batch

    @staticmethod
    def zero_mask_invalid(batch: Dict[str, Tensor]):
        for k_agent in ["target", "other"]:

            agent_invalid = ~batch[f"ac/{k_agent}_valid"].unsqueeze(-1)
            for k in ["pos", "vel", "spd", "acc", "yaw_bbox", "yaw_rate"]:
                _key = f"ac/{k_agent}_{k}"
                batch[_key] = batch[_key].masked_fill(agent_invalid, 0)

            agent_invalid = ~(batch[f"ac/{k_agent}_valid"].any(-1, keepdim=True))
            for k in ["type", "role", "size"]:
                _key = f"ac/{k_agent}_{k}"
                batch[_key] = batch[_key].masked_fill(agent_invalid, 0)

        map_invalid = ~batch["ac/map_valid"].unsqueeze(-1)
        batch["ac/map_pos"] = batch["ac/map_pos"].masked_fill(map_invalid, 0)
        batch["ac/map_dir"] = batch["ac/map_dir"].masked_fill(map_invalid, 0)
        map_invalid = ~(batch["ac/map_valid"].any(-1, keepdim=True))
        batch["ac/map_type"] = batch["ac/map_type"].masked_fill(map_invalid, 0)

        tl_invalid = ~batch["ac/tl_valid"].unsqueeze(-1)
        for k in ["state", "pos", "dir"]:
            _key = f"ac/tl_{k}"
            batch[_key] = batch[_key].masked_fill(tl_invalid, 0)

        gt_invalid = ~batch["gt/valid"].unsqueeze(-1)
        for k in ["pos", "spd", "vel", "yaw_bbox"]:
            _key = f"gt/{k}"
            batch[_key] = batch[_key].masked_fill(gt_invalid, 0)
