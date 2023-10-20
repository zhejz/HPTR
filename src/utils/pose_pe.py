# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from torch import Tensor, nn
from models.modules.pos_emb import PositionalEmbedding, PositionalEmbeddingRad


class PosePE(nn.Module):
    def __init__(self, mode: str, pe_dim: int = 256, theta_xy: float = 1e3, theta_cs: float = 1e1):
        super().__init__()
        self.mode = mode
        if self.mode == "xy_dir":
            self.out_dim = 4
        elif self.mode == "mpa_pl":
            self.out_dim = 7
        elif self.mode == "pe_xy_dir":
            self.out_dim = pe_dim
            self.pe_xy = PositionalEmbedding(dim=pe_dim // 4, theta=theta_xy)
            self.pe_dir = PositionalEmbedding(dim=pe_dim // 4, theta=theta_cs)
        elif self.mode == "pe_xy_yaw":
            self.out_dim = pe_dim
            self.pe_xy = PositionalEmbedding(dim=pe_dim // 4, theta=theta_xy)
            self.pe_yaw = PositionalEmbeddingRad(dim=pe_dim // 2)
        else:
            raise NotImplementedError

    def forward(self, xy: Tensor, dir: Tensor):
        """
        Args: input either dir or yaw.
            xy: [..., 2]
            dir: cos/sin [..., 2] or yaw [..., 1]

        Returns:
            pos_out: [..., self.out_dim]
        """
        if self.mode == "xy_dir":
            if dir.shape[-1] == 1:
                dir = torch.cat([dir.cos(), dir.sin()], dim=-1)
            pos_out = torch.cat([xy, dir], dim=-1)
        elif self.mode == "mpa_pl":
            if dir.shape[-1] == 1:
                dir = torch.cat([dir.cos(), dir.sin()], dim=-1)
            pos_out = self.encode_polyline(xy, dir)
        elif self.mode == "pe_xy_dir":
            if dir.shape[-1] == 1:
                dir = torch.cat([dir.cos(), dir.sin()], dim=-1)
            pos_out = torch.cat(
                [self.pe_xy(xy[..., 0]), self.pe_xy(xy[..., 1]), self.pe_dir(dir[..., 0]), self.pe_dir(dir[..., 1])],
                dim=-1,
            )
        elif self.mode == "pe_xy_yaw":
            if dir.shape[-1] == 1:
                dir = dir.squeeze(-1)
            else:
                dir = torch.atan2(dir[..., 1], dir[..., 0])
            pos_out = torch.cat([self.pe_xy(xy[..., 0]), self.pe_xy(xy[..., 1]), self.pe_yaw(dir)], dim=-1)
        return pos_out

    @staticmethod
    def encode_polyline(pos: Tensor, dir: Tensor) -> Tensor:
        """
        Args: pos and dir with respect to the agent
            pos: [..., 2]
            dir: [..., 2]

        Returns:
            pl_feature: [..., 7]
        """
        eps = torch.finfo(pos.dtype).eps
        # [n_scene, n_target, n_map, n_pl_node, 2]
        segments_start = pos
        segment_vec = dir
        # [n_scene, n_target, n_map, n_pl_node]
        segment_proj = (-segments_start * segment_vec).sum(-1) / ((segment_vec * segment_vec).sum(-1) + eps)
        # [n_scene, n_target, n_map, n_pl_node, 2]
        closest_points = segments_start + torch.clamp(segment_proj, min=0, max=1).unsqueeze(-1) * segment_vec
        # [n_scene, n_target, n_map, n_pl_node, 1]
        r_norm = torch.norm(closest_points, dim=-1, keepdim=True)
        segment_vec_norm = torch.norm(segment_vec, dim=-1, keepdim=True)
        pl_feature = torch.cat(
            [
                r_norm,  # 1
                closest_points / (r_norm + eps),  # 2
                segment_vec / (segment_vec_norm + eps),  # 2
                segment_vec_norm,  # 1
                torch.norm(segments_start + segment_vec - closest_points, dim=-1, keepdim=True),  # 1
            ],
            dim=-1,
        )
        return pl_feature