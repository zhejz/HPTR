# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import tensorflow
from torch import Tensor
from torchmetrics.metric import Metric
from google.protobuf import text_format
from waymo_open_dataset.protos import motion_metrics_pb2
from waymo_open_dataset.metrics.python.config_util_py import get_breakdown_names_from_motion_config
from waymo_open_dataset.metrics.ops import py_metrics_ops


class WaymoMetrics(Metric):
    """
    validation metrics based on ground truth trajectory, using waymo_open_dataset api
    """

    def __init__(
        self,
        prefix: str,
        step_gt: int,
        step_current: int,
        n_max_pred_agent: int,
        n_agent: int,
        interactive_challenge: bool,
    ) -> None:
        """
        submission_type: MOTION_PREDICTION = 1; INTERACTION_PREDICTION = 2; ARGOVERSE2 = 3
        """
        super().__init__(dist_sync_on_step=True)

        self.prefix = prefix
        self.step_gt = step_gt
        self.step_current = step_current
        self.interactive_challenge = interactive_challenge
        self.track_future_samples = step_gt - step_current
        self.metrics_config, self.metrics_names = self._waymo_metrics_config_names(
            step_current, self.track_future_samples
        )
        self.metrics_config = self.metrics_config.SerializeToString()
        self.metrics_type = ["min_ade", "min_fde", "miss_rate", "overlap_rate", "mean_average_precision"]

        self.n_agent = n_agent

        if self.interactive_challenge:
            self.m_joint = 1
            self.n_pred = 2
        else:
            self.m_joint = n_max_pred_agent
            self.n_pred = 1

        self.add_state("prediction_trajectory_gpu", default=[], dist_reduce_fx="cat")
        self.add_state("prediction_score_gpu", default=[], dist_reduce_fx="cat")
        self.add_state("ground_truth_trajectory_gpu", default=[], dist_reduce_fx="cat")
        self.add_state("ground_truth_is_valid_gpu", default=[], dist_reduce_fx="cat")
        self.add_state("prediction_ground_truth_indices_mask_gpu", default=[], dist_reduce_fx="cat")
        self.add_state("object_type_gpu", default=[], dist_reduce_fx="cat")
        self.add_state("brier_minADE_gpu", default=[], dist_reduce_fx="cat")
        self.add_state("brier_minFDE_gpu", default=[], dist_reduce_fx="cat")

        self.ops_inputs_cpu = {
            "prediction_trajectory": [],
            "prediction_score": [],
            "ground_truth_trajectory": [],
            "ground_truth_is_valid": [],
            "prediction_ground_truth_indices_mask": [],
            "object_type": [],
            "brier_minADE": [],
            "brier_minFDE": [],
        }

    def update(self, batch: Dict[str, Tensor], pred_traj: Tensor, pred_score: Optional[Tensor] = None) -> None:
        """
        pytorch tensors on gpu/cpu
        pred_traj: [n_batch, step_start+1...step_end, n_agent, K, 2]
        pred_score: [n_batch, n_agent, K] normalized prob or None
        """
        # [n_batch, n_agent]
        mask_pred = batch["agent/role"][..., 2]
        # mask_other = (~mask_pred) & (batch["agent/valid"][:, self.step_current, :].cpu())
        mask_other = (~mask_pred) & batch["agent/valid"][:, : self.step_current + 1, :].all(1)

        # ! for argoverse 2 evaluation
        batch_idx = torch.arange(pred_score.shape[0])[:, None]
        agent_idx = torch.arange(pred_score.shape[1])[None, :]
        dist = torch.norm(batch["agent/pos"][:, self.step_current + 1 :].unsqueeze(3) - pred_traj, dim=-1)
        dist = dist * (batch["agent/valid"][:, self.step_current + 1 :].unsqueeze(-1))
        # brier_minADE
        min_ade, min_idx = dist.mean(1).min(-1)
        brier_minADE = min_ade + (1 - pred_score[batch_idx, agent_idx, min_idx]) ** 2
        self.brier_minADE_gpu.append(brier_minADE[mask_pred])
        # brier_minFDE
        min_fde, min_idx = dist[:, -1].min(-1)
        brier_minFDE = min_fde + (1 - pred_score[batch_idx, agent_idx, min_idx]) ** 2
        # self.brier_minFDE_gpu.append(brier_minFDE[mask_pred])
        self.brier_minFDE_gpu.append(brier_minFDE[mask_pred & batch["agent/valid"][:, -1]])

        # gt_traj: [n_batch, n_agent, step_gt, 7]
        gt_traj = torch.cat(
            [
                batch["agent/pos"],  # [n_batch, n_step, n_agent, 2]
                batch["agent/size"][..., :2].unsqueeze(1).expand(-1, batch["agent/pos"].shape[1], -1, -1),
                batch["agent/yaw_bbox"],  # [n_batch, n_step, n_agent, 1]
                batch["agent/vel"],  # [n_batch, n_step, n_agent, 2]
            ],
            axis=-1,
        ).transpose(1, 2)
        gt_traj = gt_traj[:, :, : self.step_gt + 1, :]

        # gt_valid: [n_batch, n_agent, step_gt]
        gt_valid = batch["agent/valid"].transpose(1, 2)
        gt_valid = gt_valid[:, :, : self.step_gt + 1]

        # agent_type: [n_batch, n_agent, 3] one_hot -> [n_batch, n_agent] [Vehicle=1, Pedestrian=2, Cyclist=3]
        agent_type = batch["agent/type"].float().argmax(dim=-1) + 1.0

        # [n_batch, step_start+1...step_end, n_agent, K, 2] -> downsample -> [n_batch, steps, n_agent, K, 2]
        pred_traj = pred_traj[:, 4 : self.track_future_samples : 5]
        if self.interactive_challenge:
            # [n_batch, 1, K, n_agent, steps, 2]
            pred_traj = pred_traj.permute((0, 3, 2, 1, 4)).unsqueeze(1)
            if pred_score is None:
                pred_score = torch.ones_like(pred_traj[:, :, :, 0, 0, 0]).softmax(-1)
            else:
                # [n_batch, 1, K]
                pred_score = pred_score.sum(dim=1, keepdim=True)
        else:
            # [n_batch, n_agent, K, 1, steps, 2]
            pred_traj = pred_traj.permute((0, 2, 3, 1, 4)).unsqueeze(3)
            if pred_score is None:
                pred_score = torch.ones_like(pred_traj[:, :, :, 0, 0, 0]).softmax(-1)

        n_batch = gt_traj.shape[0]
        n_gt_step = gt_traj.shape[2]
        n_pred_step = pred_traj.shape[-2]
        n_K = pred_traj.shape[2]
        device = pred_traj.device

        prediction_trajectory = torch.zeros(
            [n_batch, self.m_joint, n_K, self.n_pred, n_pred_step, 2], dtype=torch.float32, device=device
        )
        prediction_score = torch.zeros([n_batch, self.m_joint, n_K], dtype=torch.float32, device=device)
        ground_truth_trajectory = torch.zeros([n_batch, self.n_agent, n_gt_step, 7], dtype=torch.float32, device=device)
        ground_truth_is_valid = torch.zeros([n_batch, self.n_agent, n_gt_step], dtype=torch.bool, device=device)
        prediction_ground_truth_indices_mask = torch.zeros(
            [n_batch, self.m_joint, self.n_pred], dtype=torch.bool, device=device
        )
        object_type = torch.zeros([n_batch, self.n_agent], dtype=torch.float32, device=device)

        for i in range(n_batch):
            # reorder and reduce ground_truth_trajectory and ground_truth_is_valid, first pred_agent then other_agent
            n_pred_agent = mask_pred[i].sum()
            n_other_agent = mask_other[i].sum()

            if self.interactive_challenge:
                # pred_traj: [n_batch, 1, K, n_agent, steps, 2]
                prediction_trajectory[i, :, :, :n_pred_agent] = pred_traj[i, :, :, mask_pred[i]]
                prediction_score[i] = pred_score[i]
                prediction_ground_truth_indices_mask[i, :, :n_pred_agent] = True
            else:
                # pred_traj: [n_batch, n_agent, K, 1, steps, 2]
                prediction_trajectory[i, :n_pred_agent] = pred_traj[i, mask_pred[i]]
                prediction_score[i, :n_pred_agent] = pred_score[i][mask_pred[i]]
                prediction_ground_truth_indices_mask[i, :n_pred_agent] = True

            ground_truth_trajectory[i, :n_pred_agent] = gt_traj[i][mask_pred[i]]
            ground_truth_is_valid[i, :n_pred_agent] = gt_valid[i][mask_pred[i]]
            ground_truth_trajectory[i, n_pred_agent : n_pred_agent + n_other_agent] = gt_traj[i][mask_other[i]]
            ground_truth_is_valid[i, n_pred_agent : n_pred_agent + n_other_agent] = gt_valid[i][mask_other[i]]
            object_type[i, :n_pred_agent] = agent_type[i][mask_pred[i]]
            object_type[i, n_pred_agent : n_pred_agent + n_other_agent] = agent_type[i][mask_other[i]]

        self.prediction_trajectory_gpu.append(prediction_trajectory)
        self.prediction_score_gpu.append(prediction_score)
        self.ground_truth_trajectory_gpu.append(ground_truth_trajectory)
        self.ground_truth_is_valid_gpu.append(ground_truth_is_valid)
        self.prediction_ground_truth_indices_mask_gpu.append(prediction_ground_truth_indices_mask)
        self.object_type_gpu.append(object_type)

    def compute(self) -> Dict[str, Tensor]:
        out_dict = {
            "prediction_trajectory": self.prediction_trajectory_gpu,
            "prediction_score": self.prediction_score_gpu,
            "ground_truth_trajectory": self.ground_truth_trajectory_gpu,
            "ground_truth_is_valid": self.ground_truth_is_valid_gpu,
            "prediction_ground_truth_indices_mask": self.prediction_ground_truth_indices_mask_gpu,
            "object_type": self.object_type_gpu,
            "brier_minADE": self.brier_minADE_gpu,
            "brier_minFDE": self.brier_minFDE_gpu,
        }
        return out_dict

    def aggregate_on_cpu(self, gpu_dict_sync: Dict[str, Tensor]) -> None:
        for k, v in gpu_dict_sync.items():
            if type(v) is list:
                assert len(v) == 1
                v = v[0]
            if v.numel() == 1:
                v = v.unsqueeze(0)
                if k == "prediction_ground_truth_indices_mask":
                    v = v[:, None, None]
            self.ops_inputs_cpu[k].append(v.cpu())

    def compute_waymo_motion_metrics(self) -> Dict[str, Tensor]:
        tensorflow.config.set_visible_devices([], "GPU")
        ops_inputs = {}
        for k in self.ops_inputs_cpu.keys():
            ops_inputs[k] = torch.cat(self.ops_inputs_cpu[k], dim=0)
            self.ops_inputs_cpu[k] = []

        if self.interactive_challenge:
            # [1, 1, 2]
            indices = torch.arange(self.n_pred, dtype=torch.int64)[None, None, :]
        else:
            # [1, 8, 1]
            indices = torch.arange(self.m_joint, dtype=torch.int64)[None, :, None]
        # [n_batch, self.m_joint, self.n_pred]
        ops_inputs["prediction_ground_truth_indices"] = indices.expand(ops_inputs["object_type"].shape[0], -1, -1)

        out_dict = {}
        metric_values = py_metrics_ops.motion_metrics(
            config=self.metrics_config,
            prediction_trajectory=ops_inputs["prediction_trajectory"],
            prediction_score=ops_inputs["prediction_score"],
            ground_truth_trajectory=ops_inputs["ground_truth_trajectory"],
            ground_truth_is_valid=ops_inputs["ground_truth_is_valid"],
            prediction_ground_truth_indices_mask=ops_inputs["prediction_ground_truth_indices_mask"],
            object_type=ops_inputs["object_type"],
            prediction_ground_truth_indices=ops_inputs["prediction_ground_truth_indices"],
        )

        for m_type in self.metrics_type:  # e.g. min_ade
            values = np.array(getattr(metric_values, m_type))
            sum_VEHICLE = 0.0
            sum_PEDESTRIAN = 0.0
            sum_CYCLIST = 0.0
            counter_VEHICLE = 0.0
            counter_PEDESTRIAN = 0.0
            counter_CYCLIST = 0.0
            for i, m_name in enumerate(self.metrics_names):  # e.g. TYPE_CYCLIST_15
                out_dict[f"waymo_metrics/{self.prefix}_{m_type}_{m_name}"] = values[i]
                if "VEHICLE" in m_name:
                    sum_VEHICLE += values[i]
                    counter_VEHICLE += 1
                elif "PEDESTRIAN" in m_name:
                    sum_PEDESTRIAN += values[i]
                    counter_PEDESTRIAN += 1
                elif "CYCLIST" in m_name:
                    sum_CYCLIST += values[i]
                    counter_CYCLIST += 1
            out_dict[f"{self.prefix}/{m_type}"] = values.mean()
            out_dict[f"{self.prefix}/veh/{m_type}"] = sum_VEHICLE / counter_VEHICLE
            out_dict[f"{self.prefix}/ped/{m_type}"] = sum_PEDESTRIAN / counter_PEDESTRIAN
            out_dict[f"{self.prefix}/cyc/{m_type}"] = sum_CYCLIST / counter_CYCLIST
        out_dict[f"{self.prefix}/brier_minADE"] = ops_inputs["brier_minADE"].mean()
        out_dict[f"{self.prefix}/brier_minFDE"] = ops_inputs["brier_minFDE"].mean()
        return out_dict

    @staticmethod
    def _waymo_metrics_config_names(
        track_history_samples: int, track_future_samples: int
    ) -> Tuple[motion_metrics_pb2.MotionMetricsConfig, List[str]]:
        config = motion_metrics_pb2.MotionMetricsConfig()
        config_text = f"""
            track_steps_per_second: 10
            prediction_steps_per_second: 2
            track_history_samples: {track_history_samples}
            track_future_samples: {track_future_samples}
            speed_lower_bound: 1.4
            speed_upper_bound: 11.0
            speed_scale_lower: 0.5
            speed_scale_upper: 1.0
            max_predictions: 6
            """
        if track_future_samples == 80:
            config_text += """
                step_configurations {
                measurement_step: 5
                lateral_miss_threshold: 1.0
                longitudinal_miss_threshold: 2.0
                }"""
            config_text += """
                step_configurations {
                measurement_step: 9
                lateral_miss_threshold: 1.8
                longitudinal_miss_threshold: 3.6
                }"""
            config_text += """
                step_configurations {
                measurement_step: 15
                lateral_miss_threshold: 3.0
                longitudinal_miss_threshold: 6.0
                }"""
        elif track_future_samples == 60:
            config_text += """
                step_configurations {
                measurement_step: 11
                lateral_miss_threshold: 2.2
                longitudinal_miss_threshold: 4.4
                }"""
        else:
            raise ValueError
        text_format.Parse(config_text, config)
        metric_names = get_breakdown_names_from_motion_config(config)
        return config, metric_names
