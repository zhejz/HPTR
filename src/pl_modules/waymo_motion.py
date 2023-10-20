# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, List, Tuple, Optional
import hydra
from pytorch_lightning import LightningModule
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
import numpy as np
import wandb
from pathlib import Path
from utils.vis_waymo import VisWaymo


class WaymoMotion(LightningModule):
    def __init__(
        self,
        time_step_current: int,
        time_step_end: int,
        data_size: DictConfig,
        train_metric: DictConfig,
        waymo_metric: DictConfig,
        model: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        pre_processing: DictConfig,
        post_processing: DictConfig,
        n_video_batch: int,
        inference_repeat_n: int,
        inference_cache_map: bool,
        sub_womd: DictConfig,
        sub_av2: DictConfig,
        interactive_challenge: bool = False,
        wb_artifact: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # pre_processing
        self.pre_processing = []
        pre_proc_kwargs = {}
        for _, v in pre_processing.items():
            _pre_proc = hydra.utils.instantiate(v, time_step_current=time_step_current, data_size=data_size)
            self.pre_processing.append(_pre_proc)
            pre_proc_kwargs |= _pre_proc.model_kwargs
        self.pre_processing = nn.Sequential(*self.pre_processing)
        # model
        self.model = hydra.utils.instantiate(model, **pre_proc_kwargs, _recursive_=False)
        # post_processing
        self.post_processing = nn.Sequential(*[hydra.utils.instantiate(v) for _, v in post_processing.items()])
        # save submission files
        self.sub_womd = hydra.utils.instantiate(
            sub_womd,
            k_futures=post_processing.waymo.k_pred,
            wb_artifact=wb_artifact,
            interactive_challenge=interactive_challenge,
        )
        self.sub_av2 = hydra.utils.instantiate(sub_av2, k_futures=post_processing.waymo.k_pred)
        # metrics
        self.train_metric = hydra.utils.instantiate(
            train_metric, prefix="train", n_decoders=self.model.n_decoders, n_pred=self.model.n_pred
        )
        self.waymo_metric = hydra.utils.instantiate(
            waymo_metric,
            prefix="waymo_pred",
            step_gt=time_step_end,
            step_current=time_step_current,
            interactive_challenge=interactive_challenge,
            n_agent=data_size["agent/valid"][-1],
        )

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        with torch.no_grad():
            batch = self.pre_processing(batch)
            input_dict = {k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k}
            gt_dict = {k.replace("/", "_"): v for k, v in batch.items() if "gt/" in k}
            pred_dict = {k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k}

        pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = self.model(**input_dict)
        pred_dict = self.post_processing(pred_dict)

        metrics_dict = self.train_metric(**pred_dict, **gt_dict)

        for k in metrics_dict.keys():
            if ("error_" in k) or ("loss" in k) or ("counter_traj" in k) or ("counter_conf" in k):
                self.log(k, metrics_dict[k], on_step=True)

        if self.global_rank == 0:
            n_d = self.train_metric.n_decoders
            n_p = self.train_metric.n_pred
            for k in ["conf", "counter"]:
                for i in range(n_d):
                    w = []
                    for j in range(n_p):
                        k_str = f"{self.train_metric.prefix}/{k}_d{i}_p{j}"
                        w.append(metrics_dict[k_str].item())
                    h = np.histogram(range(n_p), weights=w, density=True, bins=n_p, range=(0, n_p - 1))
                    self.logger[0].experiment.log(
                        {f"{self.train_metric.prefix}/{k}_d{i}": wandb.Histogram(np_histogram=h)}
                    )

        return metrics_dict[f"{self.train_metric.prefix}/loss"]

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        # ! pre-processing
        for _ in range(self.hparams.inference_repeat_n):
            batch = self.pre_processing(batch)
            input_dict = {k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k}
            pred_dict = {k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k}

        # ! model inference
        pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = self.model(
            inference_repeat_n=self.hparams.inference_repeat_n,
            inference_cache_map=self.hparams.inference_cache_map,
            **input_dict,
        )
        # print(pred_dict["pred"].shape)

        # ! post-processing
        # for _ in range(self.hparams.inference_repeat_n):
        pred_dict = self.post_processing(pred_dict)

        if self.hparams.inference_repeat_n > 1:
            return  # measuring FPS for online inference.

        # ! waymo metrics
        waymo_ops_inputs = self.waymo_metric(batch, pred_dict["waymo_trajs"], pred_dict["waymo_scores"])
        self.waymo_metric.aggregate_on_cpu(waymo_ops_inputs)
        self.waymo_metric.reset()

        self._save_to_submission_files(pred_dict, batch)

        # ! visualization
        if self.global_rank == 0 and (batch_idx < self.hparams.n_video_batch):
            vis_eps_idx = list(range(min(pred_dict["waymo_valid"].shape[0], 3)))
            # log all preds only for the first batch
            k_to_log = min(pred_dict["waymo_trajs"].shape[-2], 6) if batch_idx == 0 else 1
            videos, images = self.save_visualizations(
                prefix="scene",
                step_current=self.hparams.time_step_current,
                step_gt=self.hparams.time_step_end,
                step_end=self.hparams.time_step_end,
                batch_idx=batch_idx,
                batch=batch,
                pred_valid=pred_dict["waymo_valid"],
                pred_trajs=pred_dict["waymo_trajs"],
                pred_scores=pred_dict["waymo_scores"],
                pred_yaw_bbox=pred_dict["waymo_yaw_bbox"],
                k_to_log=k_to_log,
                vis_eps_idx=vis_eps_idx,
                save_im=True,
            )
            for p, im in images.items():
                self.logger[0].experiment.log({p: wandb.Image(p)}, commit=False)
            for p in videos.keys():
                self.logger[0].experiment.log({p: wandb.Video(p)}, commit=False)

    def validation_epoch_end(self, outputs):
        epoch_waymo_metrics = self.waymo_metric.compute_waymo_motion_metrics()
        epoch_waymo_metrics["epoch"] = self.current_epoch
        for k, v in epoch_waymo_metrics.items():
            self.log(k, v, on_epoch=True)
        self.log("val/loss", -epoch_waymo_metrics[f"{self.waymo_metric.prefix}/mean_average_precision"])

        if self.global_rank == 0:
            self.sub_womd.save_sub_files(self.logger[0])
            self.sub_av2.save_sub_files(self.logger[0])

    @staticmethod
    def save_visualizations(
        prefix: str,
        step_current: int,
        step_gt: int,
        step_end: int,
        batch_idx: int,
        batch: Dict[str, Tensor],
        pred_valid: Tensor,
        pred_trajs: Tensor,
        pred_scores: Tensor,
        vis_eps_idx: List[int] = [],
        k_to_log: int = 1,
        save_im: bool = True,
        pred_yaw_bbox: Optional[Tensor] = None,
    ) -> Dict[str, Tuple]:
        """
        Args:
            pred_valid: [n_scene, n_step, n_agent]
            pred_trajs: [n_scene, n_step, n_agent, n_pred=6, 2]
            pred_scores: [n_batch, n_agent, n_pred], normalized prob.
            pred_yaw: [n_scene, n_step, n_agent, n_pred=6, 1]
        """
        episode_keys = [
            "agent/valid",
            "agent/pos",
            "agent/yaw_bbox",
            "agent/spd",
            "agent/role",
            "agent/type",
            "agent/cmd",
            "agent/size",
            "agent/goal",
            "agent/dest",
            "map/valid",
            "map/type",
            "map/pos",
            "map/boundary",
            "tl_lane/valid",
            "tl_lane/state",
            "tl_lane/idx",
            "tl_stop/valid",
            "tl_stop/state",
            "tl_stop/pos",
            "tl_stop/dir",
            "episode_idx",
        ]
        batch_np = {k: batch[k].cpu().numpy() for k in episode_keys if k in batch}
        batch_np["agent/cmd"] = batch_np["agent/cmd"].argmax(-1)  # [n_scene, n_agent]
        # compute yaw_bbox: [n_scene, n_step-1, n_agent, n_pred, 2]
        if pred_yaw_bbox is None:
            pred_yaw_bbox = torch.diff(pred_trajs, dim=1)
            pred_yaw_bbox = torch.atan2(pred_yaw_bbox[..., 1], pred_yaw_bbox[..., 0]).unsqueeze(-1)
            pred_yaw_bbox = torch.cat([pred_yaw_bbox, pred_yaw_bbox[:, [-1]]], dim=1)
        # to cpu numpy array
        pred_valid = pred_valid.cpu().numpy()
        pred_trajs = pred_trajs.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        pred_yaw_bbox = pred_yaw_bbox.cpu().numpy()

        videos, images = {}, {}
        vis_eps_idx = range(batch["agent/valid"].shape[0]) if len(vis_eps_idx) == 0 else vis_eps_idx
        for idx in vis_eps_idx:
            _path = Path(f"video_{batch_idx}-{idx}")
            _path.mkdir(exist_ok=True, parents=True)
            episode = {k: batch_np[k][idx] for k in batch_np.keys()}
            vis_waymo = VisWaymo(episode["map/valid"], episode["map/type"], episode["map/pos"], episode["map/boundary"])
            for kf in range(k_to_log):
                prediction = {
                    "step_current": step_current,
                    "step_gt": step_gt,
                    "step_end": step_end,
                    "agent/pos": pred_trajs[idx, :, :, kf, :2],  # [n_step, n_agent, 2]
                    "score": pred_scores[idx, :, kf],  # [n_agent]
                    "agent/yaw_bbox": pred_yaw_bbox[idx, :, :, kf],  # [n_step,n_agent,1]
                    "agent/valid": pred_valid[idx],  # [n_step,n_agent]
                }
                buffer_video = vis_waymo.save_prediction_videos(
                    f"{_path.name}/{prefix}_K{kf}", episode=episode, prediction=prediction
                )
                videos.update(buffer_video)

            if save_im:
                _path = Path(f"image_{batch_idx}-{idx}")
                _path.mkdir(exist_ok=True, parents=True)
                list_type_str = np.array(["veh", "ped", "cyc"])
                for k_agent in range(pred_valid.shape[-1]):
                    if pred_valid[idx, :, k_agent].any() and batch_np["agent/role"][idx, k_agent].any():
                        type_str = list_type_str[batch_np["agent/type"][idx, k_agent]][0]
                        im_path = f"{_path.name}/{prefix}_{type_str}_A{k_agent}.jpg"
                        images[im_path] = vis_waymo.save_prediction_images(
                            im_path=im_path,
                            step_current=step_current,
                            gt_valid=batch_np["agent/valid"][idx, :, k_agent],  # [n_step]
                            gt_pos=batch_np["agent/pos"][idx, :, k_agent],  # [n_step, 2]
                            gt_yaw=batch_np["agent/yaw_bbox"][idx, :, k_agent],  # [n_step, 1]
                            gt_size=batch_np["agent/size"][idx, k_agent],  # [3], lwh
                            pred_xy=pred_trajs[idx, :, k_agent],  # [n_step_future, n_pred, 2]
                            pred_scores=pred_scores[idx, k_agent],  # [n_pred]
                            cmd=batch_np["agent/cmd"][idx, k_agent],
                            episode=episode,
                        )

        return videos, images

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        # ! map can be empty for some scenes, check batch["map/valid"]
        batch = self.pre_processing(batch)
        input_dict = {k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k}
        pred_dict = {k.replace("/", "_"): v for k, v in batch.items() if "ref/" in k}
        pred_dict["pred_valid"], pred_dict["pred_conf"], pred_dict["pred"] = self.model(**input_dict)
        pred_dict = self.post_processing(pred_dict)
        self._save_to_submission_files(pred_dict, batch)

    def test_epoch_end(self, outputs):
        if self.global_rank == 0:
            self.sub_womd.save_sub_files(self.logger[0])
            self.sub_av2.save_sub_files(self.logger[0])

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        scheduler = {
            "scheduler": hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=optimizer),
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }
        return [optimizer], [scheduler]

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        self.log_dict(grad_norm_dict, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def _save_to_submission_files(self, pred_dict: Dict, batch: Dict) -> None:
        submission_kargs_dict = {
            "waymo_trajs": pred_dict["waymo_trajs"],  # after nms
            "waymo_scores": pred_dict["waymo_scores"],  # after nms
            "mask_pred": batch["history/agent/role"][..., 2],
            "object_id": batch["history/agent/object_id"],
            "scenario_center": batch["scenario_center"],
            "scenario_yaw": batch["scenario_yaw"],
            "scenario_id": batch["scenario_id"],
        }
        self.sub_av2.add_to_submissions(**submission_kargs_dict)
        self.sub_womd.add_to_submissions(**submission_kargs_dict)