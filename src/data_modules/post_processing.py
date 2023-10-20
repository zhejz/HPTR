# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import List, Dict
import torch
from torch import nn, Tensor


class ToDict(nn.Module):
    def __init__(self, predictions: List[str]) -> None:
        super().__init__()
        self.dims = {"pred_pos": None, "pred_spd": None, "pred_vel": None, "pred_yaw_bbox": None, "pred_cov": None}

        pred_dim = 0
        if "pos" in predictions:
            self.dims["pred_pos"] = (pred_dim, pred_dim + 2)
            pred_dim += 2
        if "spd" in predictions:
            self.dims["pred_spd"] = (pred_dim, pred_dim + 1)
            pred_dim += 1
        if "vel" in predictions:
            self.dims["pred_vel"] = (pred_dim, pred_dim + 2)
            pred_dim += 2
        if "yaw_bbox" in predictions:
            self.dims["pred_yaw_bbox"] = (pred_dim, pred_dim + 1)
            pred_dim += 1
        if "cov1" in predictions:
            self.dims["pred_cov"] = (pred_dim, pred_dim + 1)
            pred_dim += 1
        elif "cov2" in predictions:
            self.dims["pred_cov"] = (pred_dim, pred_dim + 2)
            pred_dim += 2
        elif "cov3" in predictions:
            self.dims["pred_cov"] = (pred_dim, pred_dim + 3)
            pred_dim += 3

    def forward(self, pred_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Inputs:
            valid: [n_scene, n_target]
            conf: [n_decoders, n_scene, n_target, n_pred], not normalized!
            pred: [n_decoders, n_scene, n_target, n_pred, n_step_future, pred_dim]
        """
        for k, v in self.dims.items():
            if v is None:
                pred_dict[k] = None
            else:
                pred_dict[k] = pred_dict["pred"][..., v[0] : v[1]]
        # del pred_dict["pred"]
        return pred_dict


class GetCovMat(nn.Module):
    def __init__(self, rho_clamp: float, std_min: float, std_max: float) -> None:
        super().__init__()
        self.rho_clamp = rho_clamp
        self.std_min = std_min
        self.std_max = std_max

    def forward(self, pred_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Inputs:
            pred_cov: [n_decoders, n_scene, n_target, n_pred, n_step_future, 1/2/3]

        Outputs:
            pred_cov: [n_decoders, n_scene, n_target, n_pred, n_step_future, 2, 2], in tril form.
        """
        if pred_dict["pred_cov"] is not None:
            cov_shape = pred_dict["pred_cov"].shape
            cov_dof = cov_shape[-1]
            if cov_dof == 3:
                a = torch.clamp(pred_dict["pred_cov"][..., 0], min=self.std_min, max=self.std_max).exp()
                b = torch.clamp(pred_dict["pred_cov"][..., 1], min=self.std_min, max=self.std_max).exp()
                c = torch.clamp(pred_dict["pred_cov"][..., 2], min=-self.rho_clamp, max=self.rho_clamp)
            elif cov_dof == 2:
                a = torch.clamp(pred_dict["pred_cov"][..., 0], min=self.std_min, max=self.std_max).exp()
                b = torch.clamp(pred_dict["pred_cov"][..., 1], min=self.std_min, max=self.std_max).exp()
                c = torch.zeros_like(a)
            elif cov_dof == 1:
                a = torch.clamp(pred_dict["pred_cov"][..., 0], min=self.std_min, max=self.std_max).exp()
                b = a
                c = torch.zeros_like(a)

            pred_dict["pred_cov"] = torch.stack([a, torch.zeros_like(a), c, b], dim=-1).view(*cov_shape[:-1], 2, 2)

        return pred_dict


class OffsetToKmeans(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, pred_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # not used, for back compatibility
        return pred_dict
