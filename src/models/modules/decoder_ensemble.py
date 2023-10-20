# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Tuple, List
import hydra
import torch
from torch import nn, Tensor
from omegaconf import DictConfig
from functorch import combine_state_for_ensemble, vmap
from .mlp import MLP


class DecoderEnsemble(nn.Module):
    def __init__(self, n_decoders: int, decoder_cfg: DictConfig) -> None:
        super().__init__()
        self.use_vmap = decoder_cfg["use_vmap"]
        self.n_decoders = n_decoders
        if self.use_vmap and self.n_decoders > 1:
            _decoders = [hydra.utils.instantiate(decoder_cfg) for _ in range(n_decoders)]
            fmodel_decoders, params_decoders, buffers_decoders = combine_state_for_ensemble(_decoders)
            assert buffers_decoders == ()
            self.v_model = vmap(fmodel_decoders, randomness="different")
            [p.requires_grad_() for p in params_decoders]
            self.params_decoders = nn.ParameterList(params_decoders)
        else:
            self._decoders = nn.ModuleList([hydra.utils.instantiate(decoder_cfg) for _ in range(n_decoders)])

    def forward(self, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            conf: [n_decoders, n_scene, n_agent, n_pred]
            pred: [n_decoders, n_scene, n_agent, n_pred, n_step_future, pred_dim]
        """
        if self.use_vmap and self.n_decoders > 1:
            conf, pred = self.v_model(tuple(self.params_decoders), (), **kwargs)
        else:
            conf, pred = [], []
            for decoder in self._decoders:
                c, p = decoder(**kwargs)
                conf.append(c)
                pred.append(p)
            conf = torch.stack(conf, dim=0)
            pred = torch.stack(pred, dim=0)
        return conf, pred


class MLPHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        use_vmap: bool,
        n_step_future: int,
        out_mlp_layernorm: bool,
        out_mlp_batchnorm: bool,
        use_agent_type: bool,
        predictions: List[str],
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_agent_type = use_agent_type
        self.n_step_future = n_step_future

        self.pred_dim = 0
        if "pos" in predictions:
            self.pred_dim += 2
        if "spd" in predictions:
            self.pred_dim += 1
        if "vel" in predictions:
            self.pred_dim += 2
        if "yaw_bbox" in predictions:
            self.pred_dim += 1
        if "cov1" in predictions:
            self.pred_dim += 1
        elif "cov2" in predictions:
            self.pred_dim += 2
        elif "cov3" in predictions:
            self.pred_dim += 3

        _d = hidden_dim * 2
        cfg_mlp_pred = {
            "fc_dims": [hidden_dim, _d, _d, self.n_step_future * self.pred_dim],
            "end_layer_activation": False,
            "use_layernorm": out_mlp_layernorm,
            "use_batchnorm": out_mlp_batchnorm,
        }
        cfg_mlp_conf = {
            "end_layer_activation": False,
            "use_layernorm": out_mlp_layernorm,
            "use_batchnorm": out_mlp_batchnorm,
        }
        n_mlp_head = 3 if use_agent_type else 1
        self.mlp_pred = MLPEnsemble(n_decoders=n_mlp_head, decoder_cfg=cfg_mlp_pred, use_vmap=use_vmap)

        cfg_mlp_conf["fc_dims"] = [hidden_dim, _d, _d, 1]
        self.mlp_conf = MLPEnsemble(n_decoders=n_mlp_head, decoder_cfg=cfg_mlp_conf, use_vmap=use_vmap)

    def forward(self, valid: Tensor, emb: Tensor, agent_type: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            valid: [n_scene, n_agent]
            emb: [n_scene, n_agent, n_pred, hidden_dim]
            agent_type: [n_scene, n_agent, 3]

        Returns:
            conf: [n_scene, n_agent, n_pred]
            pred: [n_scene, n_agent, n_pred, n_step_future, pred_dim]
        """
        pred = self.mlp_pred(x=emb, valid_mask=valid.unsqueeze(-1))  # [1/3, n_scene, n_agent, n_pred, 400]
        conf = self.mlp_conf(x=emb, valid_mask=valid.unsqueeze(-1)).squeeze(-1)  # [1/3, n_scene, n_agent, n_pred]

        if self.use_agent_type:
            _type = agent_type.movedim(-1, 0).unsqueeze(-1)  # [3, n_scene, n_agent, 1]
            pred = (pred * _type.unsqueeze(-1)).sum(0)
            conf = (conf * _type).sum(0)
        else:
            pred = pred.squeeze(0)
            conf = conf.squeeze(0)

        n_scene, n_agent, n_pred = conf.shape
        return conf, pred.view(n_scene, n_agent, n_pred, self.n_step_future, self.pred_dim)


class MLPEnsemble(nn.Module):
    def __init__(self, n_decoders: int, decoder_cfg: DictConfig, use_vmap: bool) -> None:
        super().__init__()
        self.use_vmap = use_vmap
        self.n_decoders = n_decoders
        if self.use_vmap and self.n_decoders > 1:
            _decoders = [MLP(**decoder_cfg) for _ in range(n_decoders)]
            fmodel_decoders, params_decoders, buffers_decoders = combine_state_for_ensemble(_decoders)
            assert buffers_decoders == ()
            self.v_model = vmap(fmodel_decoders, randomness="different")
            [p.requires_grad_() for p in params_decoders]
            self.params_decoders = nn.ParameterList(params_decoders)
        else:
            self._decoders = nn.ModuleList([MLP(**decoder_cfg) for _ in range(n_decoders)])

    def forward(self, **kwargs) -> Tensor:
        if self.use_vmap and self.n_decoders > 1:
            out = self.v_model(tuple(self.params_decoders), (), **kwargs)
        else:
            out = []
            for decoder in self._decoders:
                x = decoder(**kwargs)
                out.append(x)
            out = torch.stack(out, dim=0)
        return out
