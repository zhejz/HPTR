# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from utils.transform_utils import torch_rad2rot, torch_pos2local, torch_rad2local


@torch.no_grad()
def get_rel_pose(pose: Tensor, invalid: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Args:
        pose: [n_scene, n_emb, 3], (x,y,yaw), in global coordinate
        invalid: [n_scene, n_emb]

    Returns:
        rel_pose: [n_scene, n_emb, n_emb, 3] (x,y,yaw)
        rel_dist: [n_scene, n_emb, n_emb]
    """
    xy = pose[:, :, :2]  # [n_scene, n_emb, 2]
    yaw = pose[:, :, -1]  # [n_scene, n_emb]
    rel_pose = torch.cat(
        [
            torch_pos2local(xy.unsqueeze(1), xy.unsqueeze(2), torch_rad2rot(yaw)),
            torch_rad2local(yaw.unsqueeze(1), yaw, cast=False).unsqueeze(-1),
        ],
        dim=-1,
    )  # [n_scene, n_emb, n_emb, 3]
    rel_dist = torch.norm(rel_pose[..., :2], dim=-1)  # [n_scene, n_emb, n_emb]
    rel_dist.masked_fill_(invalid.unsqueeze(1) | invalid.unsqueeze(2), float("inf"))
    return rel_pose, rel_dist


@torch.no_grad()
def get_rel_dist(xy: Tensor, invalid: Tensor) -> Tensor:
    """
    Args:
        xy: [n_scene, n_emb, 2], in global coordinate
        invalid: [n_scene, n_emb]

    Returns:
        rel_dist: [n_scene, n_emb, n_emb]
    """
    rel_dist = torch.norm(xy.unsqueeze(1) - xy.unsqueeze(2), dim=-1)  # [n_scene, n_emb, n_emb]
    rel_dist.masked_fill_(invalid.unsqueeze(1) | invalid.unsqueeze(2), float("inf"))
    return rel_dist


@torch.no_grad()
def get_tgt_knn_idx(
    tgt_invalid: Tensor, rel_pose: Optional[Tensor], rel_dist: Tensor, n_tgt_knn: int, dist_limit: Union[float, Tensor],
) -> Tuple[Optional[Tensor], Tensor, Optional[Tensor]]:
    """
    Args:
        tgt_invalid: [n_scene, n_tgt]
        rel_pose: [n_scene, n_src, n_tgt, 3]
        rel_dist: [n_scene, n_src, n_tgt]
        knn: int, set to <=0 to skip knn, i.e. n_tgt_knn=n_tgt
        dist_limit: float, or Tensor [n_scene, n_tgt, 1]

    Returns:
        idx_tgt: [n_scene, n_src, n_tgt_knn], or None
        tgt_invalid_knn: [n_scene, n_src, n_tgt_knn]
        rpe: [n_scene, n_src, n_tgt_knn, 3]
    """
    n_scene, n_src, _ = rel_dist.shape
    idx_scene = torch.arange(n_scene)[:, None, None]  # [n_scene, 1, 1]
    idx_src = torch.arange(n_src)[None, :, None]  # [1, n_src, 1]

    if 0 < n_tgt_knn < tgt_invalid.shape[1]:
        # [n_scene, n_src, n_tgt_knn]
        dist_knn, idx_tgt = torch.topk(rel_dist, n_tgt_knn, dim=-1, largest=False, sorted=False)
        # [n_scene, n_src, n_tgt_knn]
        tgt_invalid_knn = tgt_invalid.unsqueeze(1).expand(-1, n_src, -1)[idx_scene, idx_src, idx_tgt]
        # [n_batch, n_src, n_tgt_knn, 3]
        if rel_pose is None:
            rpe = None
        else:
            rpe = rel_pose[idx_scene, idx_src, idx_tgt]
    else:
        dist_knn = rel_dist
        tgt_invalid_knn = tgt_invalid.unsqueeze(1).expand(-1, n_src, -1)  # [n_scene, n_src, n_tgt]
        rpe = rel_pose
        idx_tgt = None

    tgt_invalid_knn = tgt_invalid_knn | (dist_knn > dist_limit)
    if rpe is not None:
        rpe = rpe.masked_fill(tgt_invalid_knn.unsqueeze(-1), 0)

    return idx_tgt, tgt_invalid_knn, rpe
