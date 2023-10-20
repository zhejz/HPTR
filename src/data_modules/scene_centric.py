# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict
from omegaconf import DictConfig
import torch
from torch import nn, Tensor
from utils.transform_utils import torch_rad2rot, torch_pos2local, torch_dir2local, torch_rad2local


class SceneCentricPreProcessing(nn.Module):
    def __init__(self, time_step_current: int, data_size: DictConfig, gt_in_local: bool, mask_invalid: bool) -> None:
        super().__init__()
        self.step_current = time_step_current
        self.n_step_hist = time_step_current + 1
        self.gt_in_local = gt_in_local
        self.mask_invalid = mask_invalid
        self.model_kwargs = {"gt_in_local": gt_in_local, "agent_centric": False}

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

        Returns: scene-centric Dict, masked according to valid
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
            # agents attributes
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
        """
        prefix = "" if self.training else "history/"

        # ! prepare "ref/"
        batch["ref/type"] = batch[prefix + "agent/type"]
        batch["ref/role"] = batch[prefix + "agent/role"]

        last_valid_step = (
            self.step_current - batch[prefix + "agent/valid"][:, : self.step_current + 1].flip(1).max(1)[1]
        )  # [n_scene, n_agent]
        i_scene = torch.arange(batch["ref/type"].shape[0]).unsqueeze(1)  # [n_scene, 1]
        i_agent = torch.arange(batch["ref/type"].shape[1]).unsqueeze(0)  # [1, n_agent]
        ref_pos = batch[prefix + "agent/pos"][i_scene, last_valid_step, i_agent].unsqueeze(-2).contiguous()
        ref_yaw = batch[prefix + "agent/yaw_bbox"][i_scene, last_valid_step, i_agent]
        ref_rot = torch_rad2rot(ref_yaw.squeeze(-1))
        batch["ref/pos"] = ref_pos
        batch["ref/yaw"] = ref_yaw
        batch["ref/rot"] = ref_rot

        # ! prepare agents states
        # [n_scene, n_step, n_agent, ...] -> [n_scene, n_agent, n_step_hist, ...]
        for k in ("valid", "pos", "vel", "spd", "acc", "yaw_bbox", "yaw_rate"):
            batch[f"sc/agent_{k}"] = batch[f"{prefix}agent/{k}"][:, : self.n_step_hist].transpose(1, 2)

        # ! prepare agents attributes
        for k in ("type", "role", "size"):
            batch[f"sc/agent_{k}"] = batch[f"{prefix}agent/{k}"]

        # ! training/validation time, prepare "gt/" for losses
        if "agent/valid" in batch.keys():
            batch["gt/cmd"] = batch["agent/cmd"]
            for k in ("valid", "spd", "pos", "vel", "yaw_bbox"):
                batch[f"gt/{k}"] = batch[f"agent/{k}"][:, self.n_step_hist :].transpose(1, 2).contiguous()

            if self.gt_in_local:
                # [n_scene, n_agent, n_step_hist, 2]
                batch["gt/pos"] = torch_pos2local(batch["gt/pos"], ref_pos, ref_rot)
                batch["gt/vel"] = torch_dir2local(batch["gt/vel"], ref_rot)
                # [n_scene, n_agent, n_step_hist, 1]
                batch["gt/yaw_bbox"] = torch_rad2local(batch["gt/yaw_bbox"], ref_yaw, cast=False)

        # ! prepare map polylines
        for k in ("valid", "type", "pos", "dir"):
            batch[f"sc/map_{k}"] = batch[f"map/{k}"]

        # ! prepare traffic lights
        for k in ("valid", "state", "pos", "dir"):
            batch[f"sc/tl_{k}"] = batch[f"{prefix}tl_stop/{k}"][:, : self.n_step_hist]

        if self.mask_invalid:
            self.zero_mask_invalid(batch)
        return batch

    @staticmethod
    def zero_mask_invalid(batch: Dict[str, Tensor]):

        agent_invalid = ~batch["sc/agent_valid"].unsqueeze(-1)
        for k in ["pos", "vel", "spd", "acc", "yaw_bbox", "yaw_rate"]:
            _key = f"sc/agent_{k}"
            batch[_key] = batch[_key].masked_fill(agent_invalid, 0)

        agent_invalid = ~(batch["sc/agent_valid"].any(-1, keepdim=True))
        for k in ["type", "role", "size"]:
            _key = f"sc/agent_{k}"
            batch[_key] = batch[_key].masked_fill(agent_invalid, 0)

        map_invalid = ~batch["sc/map_valid"].unsqueeze(-1)
        batch["sc/map_pos"] = batch["sc/map_pos"].masked_fill(map_invalid, 0)
        batch["sc/map_dir"] = batch["sc/map_dir"].masked_fill(map_invalid, 0)
        map_invalid = ~(batch["sc/map_valid"].any(-1, keepdim=True))
        batch["sc/map_type"] = batch["sc/map_type"].masked_fill(map_invalid, 0)

        tl_invalid = ~batch["sc/tl_valid"].unsqueeze(-1)
        for k in ["state", "pos", "dir"]:
            _key = f"sc/tl_{k}"
            batch[_key] = batch[_key].masked_fill(tl_invalid, 0)

        gt_invalid = ~batch["gt/valid"].unsqueeze(-1)
        for k in ["pos", "spd", "vel", "yaw_bbox"]:
            _key = f"gt/{k}"
            batch[_key] = batch[_key].masked_fill(gt_invalid, 0)
