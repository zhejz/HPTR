# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Tuple
import numpy as np
import torch
from torch import nn, Tensor
from omegaconf import DictConfig
from .modules.mlp import MLP
from .modules.point_net import PointNet
from .modules.transformer import TransformerBlock
from .modules.decoder_ensemble import DecoderEnsemble, MLPHead
from .modules.multi_modal import MultiModalAnchors


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_pred: int,
        use_vmap: bool,
        mlp_head: DictConfig,
        multi_modal_anchors: DictConfig,
        tf_cfg: DictConfig,
        latent_query: DictConfig,
        n_latent_query: int,
        n_layer_tf_all2all: int,
        latent_query_use_tf_decoder: bool,
        n_layer_tf_anchor: int,
        anchor_self_attn: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_pred = n_pred
        self.n_layer_tf_all2all = n_layer_tf_all2all
        self.n_latent_query = n_latent_query
        self.latent_query_use_tf_decoder = latent_query_use_tf_decoder

        if self.n_layer_tf_all2all > 0:
            if self.n_latent_query > 0:
                self.latent_query = MultiModalAnchors(
                    hidden_dim=hidden_dim, emb_dim=hidden_dim, n_pred=self.n_latent_query, **latent_query
                )
                if self.latent_query_use_tf_decoder:
                    self.tf_latent_query = TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        n_layer=n_layer_tf_all2all,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                else:
                    self.tf_latent_cross = TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, n_layer=1, **tf_cfg
                    )
                    self.tf_latent_self = TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, n_layer=n_layer_tf_all2all, **tf_cfg
                    )
            else:
                self.tf_self_attn = TransformerBlock(
                    d_model=hidden_dim, d_feedforward=hidden_dim * 4, n_layer=n_layer_tf_all2all, **tf_cfg
                )

        self.anchors = MultiModalAnchors(
            hidden_dim=hidden_dim, emb_dim=hidden_dim, n_pred=n_pred, **multi_modal_anchors
        )
        self.tf_anchor = TransformerBlock(
            d_model=hidden_dim,
            d_feedforward=hidden_dim * 4,
            n_layer=n_layer_tf_anchor,
            decoder_self_attn=anchor_self_attn,
            **tf_cfg,
        )
        self.mlp_head = MLPHead(hidden_dim=hidden_dim, use_vmap=use_vmap, n_pred=n_pred, **mlp_head)

    def forward(self, valid: Tensor, target_type: Tensor, emb: Tensor, emb_invalid: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            valid: [n_scene, n_target]
            emb_invalid: [n_scene*n_target, :]
            emb: [n_scene*n_target, :, hidden_dim]
            target_type: [n_scene, n_target, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]

        Returns:
            conf: [n_scene, n_target, n_pred]
            pred: [n_scene, n_target, n_pred, n_step_future, pred_dim]
        """
        if self.n_layer_tf_all2all > 0:  # ! all2all attention
            if self.n_latent_query > 0:  # all2all attention with latent query
                # [n_scene*n_agent, n_latent_query, out_dim]
                lq_emb = self.latent_query(valid.flatten(0, 1), None, target_type.flatten(0, 1))
                if self.latent_query_use_tf_decoder:
                    emb, _ = self.tf_latent_query(src=lq_emb, tgt=emb, tgt_padding_mask=emb_invalid)
                else:
                    emb, _ = self.tf_latent_cross(src=lq_emb, tgt=emb, tgt_padding_mask=emb_invalid)
                    emb, _ = self.tf_latent_self(src=emb, tgt=emb)
                emb_invalid = (~valid).flatten(0, 1).unsqueeze(-1).expand(-1, lq_emb.shape[1])
            else:  # all2all attention without latent query
                emb, _ = self.tf_self_attn(src=emb, tgt=emb, tgt_padding_mask=emb_invalid)

        # [n_batch, n_pred, hidden_dim]
        anchors = self.anchors(valid.flatten(0, 1), emb, target_type.flatten(0, 1))
        # [n_batch, n_pred, hidden_dim]
        emb, _ = self.tf_anchor(src=anchors, tgt=emb, tgt_padding_mask=emb_invalid)
        # [n_scene, n_target, n_pred, hidden_dim]
        emb = emb.view(valid.shape[0], valid.shape[1], self.n_pred, self.hidden_dim)
        # generate output
        conf, pred = self.mlp_head(valid, emb, target_type)
        return conf, pred


class AgentCentricGlobal(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        n_pl_node: int,
        use_current_tl: bool,
        pl_aggr: bool,
        n_step_hist: int,
        n_decoders: int,
        decoder: DictConfig,
        tf_cfg: DictConfig,
        intra_class_encoder: DictConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_pred = decoder.n_pred
        self.n_decoders = n_decoders
        self.pl_aggr = pl_aggr

        self.intra_class_encoder = IntraClassEncoder(
            hidden_dim=hidden_dim,
            agent_attr_dim=agent_attr_dim,
            map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim,
            pl_aggr=pl_aggr,
            tf_cfg=tf_cfg,
            use_current_tl=use_current_tl,
            n_step_hist=n_step_hist,
            n_pl_node=n_pl_node,
            **intra_class_encoder,
        )

        decoder["tf_cfg"] = tf_cfg
        decoder["hidden_dim"] = hidden_dim
        self.decoder = DecoderEnsemble(n_decoders, decoder_cfg=decoder)

        model_parameters = filter(lambda p: p.requires_grad, self.intra_class_encoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Encoder parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Decoder parameters: {total_params/1000000:.2f}M")

    def forward(
        self,
        target_valid: Tensor,
        target_type: Tensor,
        target_attr: Tensor,
        other_valid: Tensor,
        other_attr: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        inference_repeat_n: int = 1,
        inference_cache_map: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
           target_type: [n_scene, n_target, 3]
            # target history, other history, map
                if pl_aggr:
                    target_valid: [n_scene, n_target], bool
                    target_attr: [n_scene, n_target, agent_attr_dim]
                    other_valid: [n_scene, n_target, n_other], bool
                    other_attr: [n_scene, n_target, n_other, agent_attr_dim]
                    map_valid: [n_scene, n_target, n_map], bool
                    map_attr: [n_scene, n_target, n_map, map_attr_dim]
                else:
                    target_valid: [n_scene, n_target, n_step_hist], bool
                    target_attr: [n_scene, n_target, n_step_hist, agent_attr_dim]
                    other_valid: [n_scene, n_target, n_other, n_step_hist], bool
                    other_attr: [n_scene, n_target, n_other, n_step_hist, agent_attr_dim]
                    map_valid: [n_scene, n_target, n_map, n_pl_node], bool
                    map_attr: [n_scene, n_target, n_map, n_pl_node, map_attr_dim]
            # traffic lights: cannot be aggregated, detections are not tracked.
                if use_current_tl:
                    tl_valid: [n_scene, n_target, 1, n_tl], bool
                    tl_attr: [n_scene, n_target, 1, n_tl, tl_attr_dim]
                else:
                    tl_valid: [n_scene, n_target, n_step_hist, n_tl], bool
                    tl_attr: [n_scene, n_target, n_step_hist, n_tl, tl_attr_dim]

        Returns: will be compared to "output/gt_pos": [n_scene, n_agent, n_step_future, 2]
            valid: [n_scene, n_target]
            conf: [n_decoder, n_scene, n_target, n_pred], not normalized!
            pred: [n_decoder, n_scene, n_target, n_pred, n_step_future, pred_dim]
        """
        for _ in range(inference_repeat_n):
            valid = target_valid if self.pl_aggr else target_valid.any(-1)  # [n_scene, n_target]
            emb, emb_invalid = self.intra_class_encoder(
                target_valid=target_valid,
                target_attr=target_attr,
                other_valid=other_valid,
                other_attr=other_attr,
                map_valid=map_valid,
                map_attr=map_attr,
                tl_valid=tl_valid,
                tl_attr=tl_attr,
            )

            conf, pred = self.decoder(valid=valid, target_type=target_type, emb=emb, emb_invalid=emb_invalid)

        assert torch.isfinite(conf).all()
        assert torch.isfinite(pred).all()
        return valid, conf, pred


class IntraClassEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        pl_aggr: bool,
        n_step_hist: int,
        n_pl_node: int,
        tf_cfg: DictConfig,
        use_current_tl: bool,
        add_learned_pe: bool,
        use_point_net: bool,
        n_layer_mlp: int,
        mlp_cfg: DictConfig,
        n_layer_tf: int,
    ) -> None:
        super().__init__()
        self.pl_aggr = pl_aggr
        self.use_current_tl = use_current_tl
        self.add_learned_pe = add_learned_pe
        self.use_point_net = use_point_net

        self.fc_tl = MLP([tl_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
        if self.use_point_net:
            assert not self.pl_aggr
            self.point_net_target = PointNet(agent_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg)
            self.point_net_other = PointNet(agent_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg)
            self.point_net_map = PointNet(map_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg)
        else:
            self.fc_target = MLP([agent_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
            self.fc_other = MLP([agent_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
            self.fc_map = MLP([map_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)

        if self.add_learned_pe:
            if self.pl_aggr or self.use_point_net:
                self.pe_target = nn.Parameter(torch.zeros([1, hidden_dim]), requires_grad=True)
                self.pe_other = nn.Parameter(torch.zeros([1, 1, hidden_dim]), requires_grad=True)
                self.pe_map = nn.Parameter(torch.zeros([1, 1, hidden_dim]), requires_grad=True)
            else:
                self.pe_target = nn.Parameter(torch.zeros([1, n_step_hist, hidden_dim]), requires_grad=True)
                self.pe_other = nn.Parameter(torch.zeros([1, 1, n_step_hist, hidden_dim]), requires_grad=True)
                self.pe_map = nn.Parameter(torch.zeros([1, 1, n_pl_node, hidden_dim]), requires_grad=True)
            if self.use_current_tl:
                self.pe_tl = nn.Parameter(torch.zeros([1, 1, 1, hidden_dim]), requires_grad=True)
            else:
                self.pe_tl = nn.Parameter(torch.zeros([1, n_step_hist, 1, hidden_dim]), requires_grad=True)

        self.tf_map = None
        self.tf_tl = None
        self.tf_other = None
        self.tf_target = None
        if n_layer_tf > 0:
            self.tf_tl = nn.ModuleList(
                [
                    TransformerBlock(d_model=hidden_dim, d_feedforward=hidden_dim * 4, **tf_cfg)
                    for _ in range(n_layer_tf)
                ]
            )
            self.tf_map = nn.ModuleList(
                [
                    TransformerBlock(d_model=hidden_dim, d_feedforward=hidden_dim * 4, **tf_cfg)
                    for _ in range(n_layer_tf)
                ]
            )
            self.tf_other = nn.ModuleList(
                [
                    TransformerBlock(d_model=hidden_dim, d_feedforward=hidden_dim * 4, **tf_cfg)
                    for _ in range(n_layer_tf)
                ]
            )
            if not (self.pl_aggr or self.use_point_net):  # singular token in this case
                self.tf_target = nn.ModuleList(
                    [
                        TransformerBlock(d_model=hidden_dim, d_feedforward=hidden_dim * 4, **tf_cfg)
                        for _ in range(n_layer_tf)
                    ]
                )

    def forward(
        self,
        target_valid: Tensor,
        target_attr: Tensor,
        other_valid: Tensor,
        other_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            # target history, other history, map
                if pl_aggr:
                    target_valid: [n_scene, n_target], bool
                    target_attr: [n_scene, n_target, agent_attr_dim]
                    other_valid: [n_scene, n_target, n_other], bool
                    other_attr: [n_scene, n_target, n_other, agent_attr_dim]
                    map_valid: [n_scene, n_target, n_map], bool
                    map_attr: [n_scene, n_target, n_map, map_attr_dim]
                else:
                    target_valid: [n_scene, n_target, n_step_hist], bool
                    target_attr: [n_scene, n_target, n_step_hist, agent_attr_dim]
                    other_valid: [n_scene, n_target, n_other, n_step_hist], bool
                    other_attr: [n_scene, n_target, n_other, n_step_hist, agent_attr_dim]
                    map_valid: [n_scene, n_target, n_map, n_pl_node], bool
                    map_attr: [n_scene, n_target, n_map, n_pl_node, map_attr_dim]
            # traffic lights: cannot be aggregated, detections are not tracked.
                if use_current_tl:
                    tl_valid: [n_scene, n_target, 1, n_tl], bool
                    tl_attr: [n_scene, n_target, 1, n_tl, tl_attr_dim]
                else:
                    tl_valid: [n_scene, n_target, n_step_hist, n_tl], bool
                    tl_attr: [n_scene, n_target, n_step_hist, n_tl, tl_attr_dim]

        Returns:
            emb: [n_batch, n_emb, hidden_dim], n_batch = n_scene * n_target
            emb_invalid: [n_batch, n_emb]
        """
        # ! MLP and polyline subnet
        # [n_batch, n_step_hist/1, n_tl, tl_attr_dim]
        tl_valid = tl_valid.flatten(0, 1)
        tl_emb = self.fc_tl(tl_attr.flatten(0, 1), tl_valid)

        if self.use_point_net:
            # [n_batch, n_map, map_attr_dim], [n_batch, n_map]
            map_emb, map_valid = self.point_net_map(map_attr.flatten(0, 1), map_valid.flatten(0, 1))
            # [n_batch, n_other, agent_attr_dim], [n_batch, n_other]
            other_emb, other_valid = self.point_net_other(other_attr.flatten(0, 1), other_valid.flatten(0, 1))
            # [n_scene, n_target, agent_attr_dim]
            target_emb, target_valid = self.point_net_target(target_attr, target_valid)
            target_emb = target_emb.flatten(0, 1)  # [n_batch, agent_attr_dim]
            target_valid = target_valid.flatten(0, 1)  # [n_batch]
        else:
            # [n_batch, n_map, (n_pl_node), map_attr_dim]
            map_valid = map_valid.flatten(0, 1)
            map_emb = self.fc_map(map_attr.flatten(0, 1), map_valid)
            # [n_batch, n_other, (n_step_hist), agent_attr_dim]
            other_valid = other_valid.flatten(0, 1)
            other_emb = self.fc_other(other_attr.flatten(0, 1), other_valid)
            # [n_batch, (n_step_hist), agent_attr_dim]
            target_valid = target_valid.flatten(0, 1)
            target_emb = self.fc_target(target_attr.flatten(0, 1), target_valid)

        # ! add learned PE
        if self.add_learned_pe:
            tl_emb = tl_emb + self.pe_tl
            map_emb = map_emb + self.pe_map
            other_emb = other_emb + self.pe_other
            target_emb = target_emb + self.pe_target

        # ! flatten tokens
        tl_emb = tl_emb.flatten(1, 2)  # [n_batch, (n_step_hist)*n_tl, :]
        tl_valid = tl_valid.flatten(1, 2)  # [n_batch, (n_step_hist)*n_tl]
        if self.pl_aggr or self.use_point_net:
            target_emb = target_emb.unsqueeze(1)  # [n_batch, 1, :]
            target_valid = target_valid.unsqueeze(1)  # [n_batch, 1]
        else:
            # target_emb: [n_batch, n_step_hist/1, :], target_valid: [n_batch, n_step_hist/1]
            map_emb = map_emb.flatten(1, 2)  # [n_batch, n_map*(n_pl_node), :]
            map_valid = map_valid.flatten(1, 2)  # [n_batch, n_map*(n_pl_node)]
            other_emb = other_emb.flatten(1, 2)  # [n_batch, n_other*(n_step_hist), :]
            other_valid = other_valid.flatten(1, 2)  # [n_batch, n_other*(n_step_hist)]

        # ! intra-class attention, c.f. Wayformer late fusion
        if self.tf_tl is not None:
            _tl_invalid = ~tl_valid
            for mod in self.tf_tl:
                tl_emb, _ = mod(
                    src=tl_emb, src_padding_mask=_tl_invalid, tgt=tl_emb, tgt_padding_mask=_tl_invalid
                )
        if self.tf_map is not None:
            _map_invalid = ~map_valid
            for mod in self.tf_map:
                map_emb, _ = mod(
                    src=map_emb, src_padding_mask=_map_invalid, tgt=map_emb, tgt_padding_mask=_map_invalid
                )
        if self.tf_other is not None:
            _other_invalid = ~other_valid
            for mod in self.tf_other:
                other_emb, _ = mod(
                    src=other_emb, src_padding_mask=_other_invalid, tgt=other_emb, tgt_padding_mask=_other_invalid
                )
        if self.tf_target is not None:
            _target_invalid = ~target_valid
            for mod in self.tf_target:
                target_emb, _ = mod(
                    src=target_emb, src_padding_mask=_target_invalid, tgt=target_emb, tgt_padding_mask=_target_invalid
                )

        emb = torch.cat([target_emb, other_emb, tl_emb, map_emb], dim=1)
        emb_invalid = ~torch.cat([target_valid, other_valid, tl_valid, map_valid], dim=1)

        return emb, emb_invalid
