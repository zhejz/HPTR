# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import List
from omegaconf import ListConfig
from pathlib import Path
import tarfile
import os
import pandas as pd
from torch import Tensor
from waymo_open_dataset.protos import motion_submission_pb2
from pytorch_lightning.loggers import WandbLogger
from .transform_utils import torch_pos2global, torch_rad2rot

# ! single GPU only


class SubWOMD:
    def __init__(
        self,
        k_futures: int,
        wb_artifact: str,
        interactive_challenge: bool,
        activate: bool,
        method_name: str,
        authors: ListConfig[str],
        affiliation: str,
        description: str,
        method_link: str,
        account_name: str,
    ) -> None:
        self.activate = activate
        if activate:
            self.submissions = {}
            for _k in range(1, k_futures + 1):
                self.submissions[_k] = motion_submission_pb2.MotionChallengeSubmission()
                self.submissions[_k].account_name = account_name
                self.submissions[_k].unique_method_name = f"{method_name}_K{_k}"
                self.submissions[_k].authors.extend(list(authors))
                self.submissions[_k].affiliation = affiliation
                self.submissions[_k].description = description
                self.submissions[_k].method_link = method_link
                if interactive_challenge:
                    self.submissions[_k].submission_type = 2
                else:
                    self.submissions[_k].submission_type = 1

    def add_to_submissions(
        self,
        waymo_trajs: Tensor,
        waymo_scores: Tensor,
        mask_pred: Tensor,
        object_id: Tensor,
        scenario_center: Tensor,
        scenario_yaw: Tensor,
        scenario_id: List[str],
    ) -> None:
        """
        Args:
            waymo_trajs: [n_batch, step_start+1...step_end, n_agent, K, 2]
            waymo_scores: [n_batch, n_agent, K]
            mask_pred: [n_batch, n_agent] bool
            object_id: [n_batch, n_agent] int
            scenario_center: [n_batch, 2]
            scenario_yaw: [n_batch]
            scenario_id: list of str
        """
        if not self.activate:
            return

        waymo_trajs = waymo_trajs[:, 4::5].permute(0, 2, 3, 1, 4)  # [n_batch, n_agent, K, n_step, 2]
        waymo_trajs = torch_pos2global(
            waymo_trajs.flatten(1, 3), scenario_center.unsqueeze(1), torch_rad2rot(scenario_yaw)
        ).view(waymo_trajs.shape)

        waymo_trajs = waymo_trajs.cpu().numpy()
        waymo_scores = waymo_scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()
        object_id = object_id.cpu().numpy()

        for i_batch in range(waymo_trajs.shape[0]):
            agent_pos = waymo_trajs[i_batch, mask_pred[i_batch]]  # [n_agent_pred, K, n_step, 2]
            agent_id = object_id[i_batch, mask_pred[i_batch]]  # [n_agent_pred]
            agent_score = waymo_scores[i_batch, mask_pred[i_batch]]  # [n_agent_pred, K]

            for n_K, submission in self.submissions.items():
                scenario_prediction = motion_submission_pb2.ChallengeScenarioPredictions()
                scenario_prediction.scenario_id = scenario_id[i_batch]

                if submission.submission_type == 1:
                    # single prediction
                    for i_track in range(agent_pos.shape[0]):
                        prediction = motion_submission_pb2.SingleObjectPrediction()
                        prediction.object_id = agent_id[i_track]
                        for _k in range(n_K):
                            scored_trajectory = motion_submission_pb2.ScoredTrajectory()
                            scored_trajectory.confidence = agent_score[i_track, _k]
                            scored_trajectory.trajectory.center_x.extend(agent_pos[i_track, _k, :, 0])
                            scored_trajectory.trajectory.center_y.extend(agent_pos[i_track, _k, :, 1])
                            prediction.trajectories.append(scored_trajectory)
                        scenario_prediction.single_predictions.predictions.append(prediction)
                else:
                    # joint prediction
                    for _k in range(n_K):
                        scored_joint_trajectory = motion_submission_pb2.ScoredJointTrajectory()
                        scored_joint_trajectory.confidence = agent_score[:, _k].sum(0)
                        for i_track in range(agent_pos.shape[0]):
                            object_trajectory = motion_submission_pb2.ObjectTrajectory()
                            object_trajectory.object_id = agent_id[i_track]
                            object_trajectory.trajectory.center_x.extend(agent_pos[i_track, _k, :, 0])
                            object_trajectory.trajectory.center_y.extend(agent_pos[i_track, _k, :, 1])
                            scored_joint_trajectory.trajectories.append(object_trajectory)
                        scenario_prediction.joint_prediction.joint_trajectories.append(scored_joint_trajectory)

                submission.scenario_predictions.append(scenario_prediction)

    def save_sub_files(self, logger: WandbLogger) -> List[str]:
        if not self.activate:
            return []

        print(f"saving womd submission files to {os.getcwd()}")
        file_paths = []
        for k, submission in self.submissions.items():
            submission_dir = Path(f"womd_K{k}")
            submission_dir.mkdir(exist_ok=True)
            f = open(submission_dir / f"womd_K{k}.bin", "wb")
            f.write(submission.SerializeToString())
            f.close()
            tar_file_name = submission_dir.as_posix() + ".tar.gz"
            with tarfile.open(tar_file_name, "w:gz") as tar:
                tar.add(submission_dir, arcname=submission_dir.name)
            if isinstance(logger, WandbLogger):
                logger.experiment.save(tar_file_name)
            else:
                file_paths.append(tar_file_name)
        return file_paths


class SubAV2:
    def __init__(self, k_futures: int, activate: bool) -> None:
        self._SUBMISSION_COL_NAMES = [
            "scenario_id",
            "track_id",
            "probability",
            "predicted_trajectory_x",
            "predicted_trajectory_y",
        ]
        self.activate = activate
        if activate:
            self.submissions = {}
            for _k in range(1, k_futures + 1):
                self.submissions[_k] = []

    def add_to_submissions(
        self,
        waymo_trajs: Tensor,
        waymo_scores: Tensor,
        mask_pred: Tensor,
        object_id: Tensor,
        scenario_center: Tensor,
        scenario_yaw: Tensor,
        scenario_id: List[str],
    ) -> None:
        """
        Args:
            waymo_trajs: [n_batch, step_start+1...step_end, n_agent, K, 2]
            waymo_scores: [n_batch, n_agent, K]
            mask_pred: [n_batch, n_agent] bool
            object_id: [n_batch, n_agent] int
            scenario_center: [n_batch, 2]
            scenario_yaw: [n_batch]
            scenario_id: list of str
        """
        if not self.activate:
            return

        waymo_trajs = waymo_trajs.permute(0, 2, 3, 1, 4)  # [n_batch, n_agent, K, n_step, 2]
        waymo_trajs = torch_pos2global(
            waymo_trajs.flatten(1, 3), scenario_center.unsqueeze(1), torch_rad2rot(scenario_yaw)
        ).view(waymo_trajs.shape)

        waymo_trajs = waymo_trajs.cpu().numpy()
        waymo_scores = waymo_scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()
        object_id = object_id.cpu().numpy()

        for i_batch in range(waymo_trajs.shape[0]):
            agent_pos = waymo_trajs[i_batch, mask_pred[i_batch]]  # [n_agent_pred, K, n_step, 2]
            agent_id = object_id[i_batch, mask_pred[i_batch]]  # [n_agent_pred]
            agent_score = waymo_scores[i_batch, mask_pred[i_batch]]  # [n_agent_pred, K]

            for i_agent in range(agent_id.shape[0]):
                for n_K in self.submissions.keys():
                    confidence = agent_score[i_agent, :n_K] / agent_score[i_agent, :n_K].sum()
                    for _k in range(n_K):
                        self.submissions[n_K].append(
                            (
                                scenario_id[i_batch],
                                str(agent_id[i_agent]),
                                confidence[_k],
                                agent_pos[i_agent, _k, :, 0],
                                agent_pos[i_agent, _k, :, 1],
                            )
                        )

    def save_sub_files(self, logger: WandbLogger) -> List[str]:
        if not self.activate:
            return []

        print(f"saving av2 submission files to {os.getcwd()}")
        file_paths = []
        for k, prediction_rows in self.submissions.items():
            parquet_file_path = Path(f"av2_K{k}.parquet").as_posix()
            submission_df = pd.DataFrame(prediction_rows, columns=self._SUBMISSION_COL_NAMES)
            submission_df.to_parquet(parquet_file_path)
            if isinstance(logger, WandbLogger):
                logger.experiment.save(parquet_file_path)
            else:
                file_paths.append(parquet_file_path)
        return file_paths