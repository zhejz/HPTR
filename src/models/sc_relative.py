# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch import nn, Tensor
from omegaconf import DictConfig
from utils.pose_pe import PosePE
from .modules.mlp import MLP
from .modules.point_net import PointNet
from .modules.transformer import TransformerBlock
from .modules.decoder_ensemble import DecoderEnsemble, MLPHead
from .modules.rpe import get_rel_pose, get_tgt_knn_idx
from .modules.multi_modal import MultiModalAnchors


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        use_vmap: bool,
        d_rpe: int,
        n_pred: int,
        mlp_head: DictConfig,
        multi_modal_anchors: DictConfig,
        tf_n_layer: int,
        tf_cfg: DictConfig,
        agent_attr_dim: int,
        k_reinforce_tl: float,
        k_reinforce_agent: float,
        k_reinforce_anchor: float,
        k_reinforce_all: float,
        n_latent_query: float,
        latent_query_use_tf_decoder: bool,
        latent_query: DictConfig,
        use_attr_for_multi_modal: bool,
        anchor_self_attn: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_pred = n_pred
        self.k_reinforce_tl = k_reinforce_tl
        self.k_reinforce_agent = k_reinforce_agent
        self.k_reinforce_anchor = k_reinforce_anchor
        self.k_reinforce_all = k_reinforce_all
        self.n_latent_query = n_latent_query
        self.use_attr_for_multi_modal = use_attr_for_multi_modal
        self.anchor_self_attn = anchor_self_attn
        self.latent_query_use_tf_decoder = latent_query_use_tf_decoder

        if self.k_reinforce_tl > 0:
            self.tf_reinforce_tl = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                    for _ in range(tf_n_layer)
                ]
            )

        if self.k_reinforce_agent > 0:
            self.tf_reinforce_agent = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                    for _ in range(tf_n_layer)
                ]
            )

        if self.k_reinforce_all > 0:
            self.tf_reinforce_all = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        decoder_self_attn=False,
                        **tf_cfg,
                    )
                    for _ in range(tf_n_layer)
                ]
            )

        # anchor based approaches
        emb_dim = agent_attr_dim if self.use_attr_for_multi_modal else hidden_dim
        self.anchors = MultiModalAnchors(hidden_dim=hidden_dim, emb_dim=emb_dim, n_pred=n_pred, **multi_modal_anchors)
        self.mlp_head = MLPHead(hidden_dim=hidden_dim, use_vmap=use_vmap, n_pred=n_pred, **mlp_head)

        if self.k_reinforce_anchor > 0:
            if self.n_latent_query > 0:
                self.latent_query = MultiModalAnchors(
                    hidden_dim=hidden_dim, emb_dim=emb_dim, n_pred=self.n_latent_query, **latent_query
                )
                if self.latent_query_use_tf_decoder:
                    self.tf_latent_query = TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        n_layer=tf_n_layer,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                else:
                    self.tf_latent_cross = TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, d_rpe=d_rpe, n_layer=1, **tf_cfg
                    )
                    self.tf_latent_self = TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, n_layer=tf_n_layer, **tf_cfg
                    )

                self.tf_reinforce_anchor = TransformerBlock(
                    d_model=hidden_dim,
                    d_feedforward=hidden_dim * 4,
                    n_layer=tf_n_layer,
                    decoder_self_attn=anchor_self_attn,
                    **tf_cfg,
                )
            else:
                self.tf_reinforce_anchor = TransformerBlock(
                    d_model=hidden_dim,
                    d_feedforward=hidden_dim * 4,
                    d_rpe=d_rpe,
                    n_layer=tf_n_layer,
                    decoder_self_attn=anchor_self_attn,
                    **tf_cfg,
                )
        else:
            if self.anchor_self_attn:
                self.tf_anchor_self = TransformerBlock(
                    d_model=hidden_dim,
                    d_feedforward=hidden_dim * 4,
                    n_layer=tf_n_layer,
                    decoder_self_attn=False,
                    **tf_cfg,
                )

    def forward(
        self,
        agent_type: Tensor,
        agent_valid: Tensor,
        agent_attr: Tensor,
        agent_emb: Tensor,
        tl_valid: Tensor,
        tl_emb: Tensor,
        map_valid: Tensor,
        map_emb: Tensor,
        knn_idx_tl2self: Optional[Tensor],
        knn_invalid_tl2self: Optional[Tensor],
        knn_rpe_tl2self: Optional[Tensor],
        knn_idx_tl2map: Optional[Tensor],
        knn_invalid_tl2map: Optional[Tensor],
        knn_rpe_tl2map: Optional[Tensor],
        knn_idx_agent2self: Optional[Tensor],
        knn_invalid_agent2self: Optional[Tensor],
        knn_rpe_agent2self: Optional[Tensor],
        knn_idx_agent2maptl: Optional[Tensor],
        knn_invalid_agent2maptl: Optional[Tensor],
        knn_rpe_agent2maptl: Optional[Tensor],
        knn_idx_agent2all: Optional[Tensor],
        knn_invalid_agent2all: Optional[Tensor],
        knn_rpe_agent2all: Optional[Tensor],
        knn_idx_all2all: Optional[Tensor],
        knn_invalid_all2all: Optional[Tensor],
        knn_rpe_all2all: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            agent_type: [n_scene, n_agent, 3]
            agent_valid: [n_scene, n_agent]
            agent_attr: [n_scene, n_agent, agent_attr_dim], used if anchor_add_to_attr, assert pl_aggr
            agent_emb: [n_scene, n_agent, hidden_dim]
            tl_valid: [n_scene, n_tl]
            tl_emb: [n_scene, n_tl, hidden_dim]
            map_valid: [n_scene, n_map]
            map_emb: [n_scene, n_map, hidden_dim]

            if self.k_reinforce_tl > 0: # tl_emb
                knn_idx_tl2self: [n_scene, n_tl, n_knn_tl2self]
                knn_invalid_tl2self: [n_scene, n_tl, n_knn_tl2self]
                knn_rpe_tl2self: [n_scene, n_tl, n_knn_tl2self, d_rpe]
                knn_idx_tl2map: [n_scene, n_tl, n_knn_tl2map]
                knn_invalid_tl2map: [n_scene, n_tl, n_knn_tl2map]
                knn_rpe_tl2map: [n_scene, n_tl, n_knn_tl2map, d_rpe]

            if self.k_reinforce_agent > 0: # agent_emb
                knn_idx_agent2self: [n_scene, n_agent, n_knn_agent2self]
                knn_invalid_agent2self: [n_scene, n_agent, n_knn_agent2self]
                knn_rpe_agent2self: [n_scene, n_agent, n_knn_agent2self, d_rpe]
                knn_idx_agent2maptl: [n_scene, n_agent, n_knn_agent2maptl]
                knn_invalid_agent2maptl: [n_scene, n_agent, n_knn_agent2maptl]
                knn_rpe_agent2maptl: [n_scene, n_agent, n_knn_agent2maptl, d_rpe]

            if self.k_reinforce_anchor > 0: # anchor_emb
                knn_idx_agent2all: [n_scene, n_agent, n_knn_agent2all]
                knn_invalid_agent2all: [n_scene, n_agent, n_knn_agent2all]
                knn_rpe_agent2all: [n_scene, n_agent, n_knn_agent2all, d_rpe]

            if self.k_reinforce_all > 0:
                knn_idx_all2all: [n_scene, n_map+n_tl+n_agent, n_knn_all2all]
                knn_invalid_all2all: [n_scene, n_map+n_tl+n_agent, n_knn_all2all]
                knn_rpe_all2all: [n_scene, n_map+n_tl+n_agent, n_knn_all2all]

        Returns:
            conf: [n_decoders, n_scene, n_agent, n_pred]
            pred: [n_decoders, n_scene, n_agent, n_pred, n_step_future, pred_dim]
        """
        n_scene, n_agent = agent_valid.shape
        n_tl = tl_valid.shape[1]
        n_map = map_valid.shape[1]
        _idx_scene = torch.arange(n_scene)[:, None, None]  # [n_scene, 1, 1]
        _idx_agent = torch.arange(n_agent)[None, :, None]  # [1, n_agent, 1]

        # ! reinforce tl_emb by attending to map_emb
        if self.k_reinforce_tl > 0:
            _idx_tl = torch.arange(n_tl)[None, :, None]  # [1, n_tl, 1]
            tl_invalid = ~tl_valid
            _tgt = map_emb.unsqueeze(1).expand(-1, n_tl, -1, -1)
            if knn_idx_tl2map is not None:
                _tgt = _tgt[_idx_scene, _idx_tl, knn_idx_tl2map]
            for mod in self.tf_reinforce_tl:
                _decoder_tgt = tl_emb.unsqueeze(1).expand(-1, n_tl, -1, -1)
                if knn_idx_tl2self is not None:
                    _decoder_tgt = _decoder_tgt[_idx_scene, _idx_tl, knn_idx_tl2self]
                tl_emb, _ = mod(
                    src=tl_emb,  # [n_scene, n_tl, hidden_dim]
                    src_padding_mask=tl_invalid,  # [n_scene, n_tl]
                    tgt=_tgt,  # [n_scene, n_tl, n_knn_tl2map, hidden_dim]
                    tgt_padding_mask=knn_invalid_tl2map,  # [n_scene, n_tl, n_knn_tl2map]
                    rpe=knn_rpe_tl2map,  # [n_scene, n_tl, n_knn_tl2map, d_rpe]
                    decoder_tgt=_decoder_tgt,  # [n_scene, n_tl, n_knn_tl2self, hidden_dim]
                    decoder_tgt_padding_mask=knn_invalid_tl2self,  # [n_scene, n_tl, n_knn_tl2self]
                    decoder_rpe=knn_rpe_tl2self,  # [n_scene, n_tl, n_knn_tl2self, d_rpe]
                )

        # ! reinforce agent_emb by attending to map_emb and tl_emb
        if self.k_reinforce_agent > 0:
            agent_invalid = ~agent_valid
            _tgt = torch.cat([map_emb, tl_emb], dim=1).unsqueeze(1).expand(-1, n_agent, -1, -1)
            if knn_idx_agent2maptl is not None:
                _tgt = _tgt[_idx_scene, _idx_agent, knn_idx_agent2maptl]
            for mod in self.tf_reinforce_agent:
                _decoder_tgt = agent_emb.unsqueeze(1).expand(-1, n_agent, -1, -1)
                if knn_idx_agent2self is not None:
                    _decoder_tgt = _decoder_tgt[_idx_scene, _idx_agent, knn_idx_agent2self]
                agent_emb, _ = mod(
                    src=agent_emb,  # [n_scene, n_agent, hidden_dim]
                    src_padding_mask=agent_invalid,  # [n_scene, n_agent]
                    tgt=_tgt,  # [n_scene, n_agent, n_knn_agent2maptl, hidden_dim]
                    tgt_padding_mask=knn_invalid_agent2maptl,  # [n_scene, n_agent, n_knn_agent2maptl]
                    rpe=knn_rpe_agent2maptl,  # [n_scene, n_agent, n_knn_agent2maptl, d_rpe]
                    decoder_tgt=_decoder_tgt,  # [n_scene, n_agent, n_knn_agent2self, hidden_dim]
                    decoder_tgt_padding_mask=knn_invalid_agent2self,  # [n_scene, n_agent, n_knn_agent2self]
                    decoder_rpe=knn_rpe_agent2self,  # [n_scene, n_agent, n_knn_agent2self, d_rpe]
                )

        # ! all-to-all self attention
        if self.k_reinforce_all > 0:
            _emb = torch.cat([map_emb, tl_emb, agent_emb], dim=1)
            _emb_invalid = ~torch.cat([map_valid, tl_valid, agent_valid], dim=1)  # [n_scene, n_emb], bool
            n_emb = n_map + n_tl + n_agent
            _idx_all = torch.arange(n_emb)[None, :, None]  # [1, n_emb, 1]
            for mod in self.tf_reinforce_all:
                _tgt = _emb.unsqueeze(1).expand(-1, n_emb, -1, -1)
                if knn_idx_all2all is not None:
                    _tgt = _tgt[_idx_scene, _idx_all, knn_idx_all2all]
                _emb, _ = mod(
                    src=_emb,  # [n_scene, n_emb, hidden_dim]
                    src_padding_mask=_emb_invalid,  # [n_scene, n_emb]
                    tgt=_tgt,  # [n_scene, n_emb, n_knn_all2all, hidden_dim]
                    tgt_padding_mask=knn_invalid_all2all,  # [n_scene, n_emb, n_knn_all2all]
                    rpe=knn_rpe_all2all,  # [n_scene, n_emb, n_knn_all2all, d_rpe]
                )
            map_emb = _emb[:, :n_map]
            tl_emb = _emb[:, n_map : n_map + n_tl]
            agent_emb = _emb[:, -n_agent:]

        # ! decode to multiple futures
        # ! prepare multi-modal anchor_emb: [n_scene*n_pred, n_agent, hidden_dim]
        anchor_emb = agent_attr if self.use_attr_for_multi_modal else agent_emb
        # [n_scene*n_agent, n_pred, hidden_dim]
        anchor_emb = self.anchors(agent_valid.flatten(0, 1), anchor_emb.flatten(0, 1), agent_type.flatten(0, 1))
        # ! reinforce anchor_emb by attending to contexts, gather context for each agent
        if self.k_reinforce_anchor > 0:
            if self.n_latent_query > 0:  # latent query attention to reduce context dimension,
                ctx_emb = agent_attr if self.use_attr_for_multi_modal else agent_emb
                # [n_scene*n_agent, n_latent_query, hidden_dim]
                ctx_emb = self.latent_query(agent_valid.flatten(0, 1), ctx_emb.flatten(0, 1), agent_type.flatten(0, 1))
                # [n_scene, n_agent, n_emb, hidden_dim]
                _tgt = torch.cat([map_emb, tl_emb, agent_emb], dim=1).unsqueeze(1).expand(-1, n_agent, -1, -1)
                # [n_scene, n_agent, n_knn_agent2all, hidden_dim]
                if knn_idx_agent2all is not None:
                    _tgt = _tgt[_idx_scene, _idx_agent, knn_idx_agent2all]

                if self.latent_query_use_tf_decoder:
                    ctx_emb, _ = self.tf_latent_query(
                        src=ctx_emb,  # [n_scene*n_agent, n_latent_query, hidden_dim]
                        tgt=_tgt.flatten(0, 1).unsqueeze(1).expand(-1, self.n_latent_query, -1, -1),
                        tgt_padding_mask=knn_invalid_agent2all.flatten(0, 1)
                        .unsqueeze(1)
                        .expand(-1, self.n_latent_query, -1),
                        rpe=knn_rpe_agent2all.flatten(0, 1).unsqueeze(1).expand(-1, self.n_latent_query, -1, -1),
                    )
                else:
                    ctx_emb, _ = self.tf_latent_cross(
                        src=ctx_emb,  # [n_scene*n_agent, n_latent_query, hidden_dim]
                        tgt=_tgt.flatten(0, 1).unsqueeze(1).expand(-1, self.n_latent_query, -1, -1),
                        tgt_padding_mask=knn_invalid_agent2all.flatten(0, 1)
                        .unsqueeze(1)
                        .expand(-1, self.n_latent_query, -1),
                        rpe=knn_rpe_agent2all.flatten(0, 1).unsqueeze(1).expand(-1, self.n_latent_query, -1, -1),
                    )
                    ctx_emb, _ = self.tf_latent_self(src=ctx_emb, tgt=ctx_emb)

                anchor_emb, _ = self.tf_reinforce_anchor(src=anchor_emb, tgt=ctx_emb)
            else:  # no compression, ctx_emb: [n_scene*n_agent, n_knn_agent2all, hidden_dim]
                ctx_emb = torch.cat([map_emb, tl_emb, agent_emb], dim=1).unsqueeze(1).expand(-1, n_agent, -1, -1)
                # [n_scene, n_agent, n_knn_agent2all, hidden_dim]
                if knn_idx_agent2all is not None:
                    ctx_emb = ctx_emb[_idx_scene, _idx_agent, knn_idx_agent2all]
                anchor_emb, _ = self.tf_reinforce_anchor(
                    src=anchor_emb,  # [n_scene*n_agent, n_pred, hidden_dim]
                    tgt=ctx_emb.flatten(0, 1).unsqueeze(1).expand(-1, self.n_pred, -1, -1),
                    tgt_padding_mask=knn_invalid_agent2all.flatten(0, 1).unsqueeze(1).expand(-1, self.n_pred, -1),
                    rpe=knn_rpe_agent2all.flatten(0, 1).unsqueeze(1).expand(-1, self.n_pred, -1, -1),
                )
        else:  # ! no reinfoce by attending to context
            if self.anchor_self_attn:
                anchor_emb, _ = self.tf_anchor_self(src=anchor_emb, tgt=anchor_emb)

        #  generate output
        anchor_emb = anchor_emb.view(n_scene, n_agent, self.n_pred, self.hidden_dim)
        conf, pred = self.mlp_head(agent_valid, anchor_emb, agent_type)

        return conf, pred


class SceneCentricRelative(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        pl_aggr: bool,
        n_tgt_knn: int,
        tf_cfg: DictConfig,
        intra_class_encoder: DictConfig,
        decoder_remove_ego_agent: bool,
        n_decoders: int,
        decoder: DictConfig,
        rpe_mode: str,
        dist_limit_map: float = 2000,
        dist_limit_tl: float = 2000,
        dist_limit_agent: List[float] = [2000, 2000, 2000],
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_pred = decoder["n_pred"]
        self.n_decoders = n_decoders
        self.decoder_remove_ego_agent = decoder_remove_ego_agent
        self.decoder_k_reinforce_tl = decoder["k_reinforce_tl"]
        self.decoder_k_reinforce_agent = decoder["k_reinforce_agent"]
        self.decoder_k_reinforce_anchor = decoder["k_reinforce_anchor"]
        self.decoder_k_reinforce_all = decoder["k_reinforce_all"]
        self.n_tgt_knn = n_tgt_knn
        self.pl_aggr = pl_aggr
        self.dist_limit_map = dist_limit_map
        self.dist_limit_tl = dist_limit_tl
        self.dist_limit_agent = dist_limit_agent

        assert rpe_mode in ["xy_dir", "pe_xy_dir", "pe_xy_yaw"]
        self.pose_rpe = PosePE(rpe_mode, pe_dim=hidden_dim)

        self.intra_class_encoder = IntraClassEncoder(
            hidden_dim=hidden_dim,
            agent_attr_dim=agent_attr_dim,
            map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim,
            pl_aggr=pl_aggr,
            pose_rpe=self.pose_rpe,
            tf_cfg=tf_cfg,
            n_tgt_knn=n_tgt_knn,
            **intra_class_encoder,
        )

        decoder["hidden_dim"] = hidden_dim
        decoder["d_rpe"] = self.pose_rpe.out_dim
        decoder["agent_attr_dim"] = agent_attr_dim
        decoder["tf_cfg"] = tf_cfg
        self.decoder = DecoderEnsemble(n_decoders, decoder_cfg=decoder)

        model_parameters = filter(lambda p: p.requires_grad, self.intra_class_encoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Encoder parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Decoder parameters: {total_params/1000000:.2f}M")

    def forward(
        self,
        agent_valid: Tensor,
        agent_type: Tensor,
        agent_attr: Tensor,
        agent_pose: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        map_pose: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        tl_pose: Tensor,
        inference_repeat_n: int = 1,
        inference_cache_map: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            agent_valid: [n_scene, n_agent, (n_step_hist)]
            agent_type: [n_scene, n_agent, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            agent_attr: [n_scene, n_agent, (n_step_hist), agent_attr_dim], local wrt. the current step.
            agent_pose: [n_scene, n_agent, 3], x,y,theta, pos of the current step.
            map_valid: [n_scene, n_map, (n_pl_node)]
            map_attr: [n_scene, n_map, (n_pl_node), map_attr_dim], local wrt. the first node.
            map_pose: [n_scene, n_map, 3], x,y,theta, polyline starting node in global coordinate.
            tl_valid: [n_scene, n_step_hist, n_tl]
            tl_attr: [n_scene, n_step_hist, n_tl, tl_attr_dim]
            tl_pose: [n_scene, n_step_hist, n_tl, 3], x,y,theta

        Returns: will be compared to "output/gt_pos": [n_scene, n_agent, n_step_future, 2]
            valid: [n_scene, n_target]
            conf: [n_decoders, n_scene, n_target, n_pred], not normalized!
            pred: [n_decoders, n_scene, n_target, n_pred, n_step_future, pred_dim]
        """
        dist_limit_agent = 0
        for i in range(agent_type.shape[-1]):  # [n_scene, n_agent]
            dist_limit_agent += agent_type[:, :, i] * self.dist_limit_agent[i]
        dist_limit_agent = dist_limit_agent.unsqueeze(-1)  # [n_scene, n_agent, 1]

        tl_valid = tl_valid.flatten(1, 2)
        tl_attr = tl_attr.flatten(1, 2)
        tl_pose = tl_pose.flatten(1, 2)
        for _ in range(inference_repeat_n):
            if self.pl_aggr:
                emb_invalid = ~torch.cat([map_valid, tl_valid, agent_valid], dim=1)
            else:
                emb_invalid = ~torch.cat([map_valid.any(-1), tl_valid, agent_valid.any(-1)], dim=1)
            rel_pose, rel_dist = get_rel_pose(torch.cat([map_pose, tl_pose, agent_pose], dim=1), emb_invalid)

        map_emb, map_valid, tl_emb, tl_valid, agent_emb, agent_valid = self.intra_class_encoder(
            inference_repeat_n=inference_repeat_n,
            inference_cache_map=inference_cache_map,
            agent_valid=agent_valid,
            agent_attr=agent_attr,
            map_valid=map_valid,
            map_attr=map_attr,
            tl_valid=tl_valid,
            tl_attr=tl_attr,
            rel_pose=rel_pose,
            rel_dist=rel_dist,
            dist_limit_map=self.dist_limit_map,
            dist_limit_tl=self.dist_limit_tl,
            dist_limit_agent=dist_limit_agent,
        )

        # ! Decoder
        for _ in range(inference_repeat_n):
            n_map = map_valid.shape[1]
            n_tl = tl_valid.shape[1]
            n_agent = agent_valid.shape[1]

            if self.decoder_remove_ego_agent:
                rel_dist[:, range(-n_agent, 0), range(-n_agent, 0)] += float("inf")

            # ! reinforce traffic lights, cross attention
            if self.decoder_k_reinforce_tl > 0:
                knn_idx_tl2self, knn_invalid_tl2self, knn_rpe_tl2self = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid[:, n_map : n_map + n_tl],  # [n_scene, n_tl]
                    rel_pose=rel_pose[:, n_map : n_map + n_tl, n_map : n_map + n_tl],  # [n_scene, n_tl, n_tl, 3]
                    rel_dist=rel_dist[:, n_map : n_map + n_tl, n_map : n_map + n_tl],  # [n_scene, n_tl, n_tl]
                    n_tgt_knn=self.n_tgt_knn,
                    dist_limit=self.dist_limit_tl,
                )
                knn_idx_tl2map, knn_invalid_tl2map, knn_rpe_tl2map = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid[:, :n_map],  # [n_scene, n_map]
                    rel_pose=rel_pose[:, n_map : n_map + n_tl, :n_map],  # [n_scene, n_tl, n_map, 3]
                    rel_dist=rel_dist[:, n_map : n_map + n_tl, :n_map],  # [n_scene, n_tl, n_map]
                    n_tgt_knn=int(self.n_tgt_knn * self.decoder_k_reinforce_tl),
                    dist_limit=self.dist_limit_tl,
                )
                knn_rpe_tl2self = self.pose_rpe(xy=knn_rpe_tl2self[..., :2], dir=knn_rpe_tl2self[..., [2]])
                knn_rpe_tl2map = self.pose_rpe(xy=knn_rpe_tl2map[..., :2], dir=knn_rpe_tl2map[..., [2]])
            else:
                knn_idx_tl2self = None
                knn_invalid_tl2self = None
                knn_rpe_tl2self = None
                knn_idx_tl2map = None
                knn_invalid_tl2map = None
                knn_rpe_tl2map = None

            # ! reinforce agents, cross attention
            if self.decoder_k_reinforce_agent > 0:
                knn_idx_agent2self, knn_invalid_agent2self, knn_rpe_agent2self = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid[:, -n_agent:],  # [n_scene, n_agent]
                    rel_pose=rel_pose[:, -n_agent:, -n_agent:],  # [n_scene, n_agent, n_agent, 3]
                    rel_dist=rel_dist[:, -n_agent:, -n_agent:],  # [n_scene, n_agent, n_agent]
                    n_tgt_knn=self.n_tgt_knn,
                    dist_limit=dist_limit_agent,
                )
                knn_idx_agent2maptl, knn_invalid_agent2maptl, knn_rpe_agent2maptl = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid[:, : n_map + n_tl],  # [n_scene, n_map+n_tl]
                    rel_pose=rel_pose[:, -n_agent:, : n_map + n_tl],  # [n_scene, n_agent, n_map+n_tl, 3]
                    rel_dist=rel_dist[:, -n_agent:, : n_map + n_tl],  # [n_scene, n_agent, n_map+n_tl]
                    n_tgt_knn=int(self.n_tgt_knn * self.decoder_k_reinforce_agent),
                    dist_limit=dist_limit_agent,
                )
                knn_rpe_agent2self = self.pose_rpe(xy=knn_rpe_agent2self[..., :2], dir=knn_rpe_agent2self[..., [2]])
                knn_rpe_agent2maptl = self.pose_rpe(xy=knn_rpe_agent2maptl[..., :2], dir=knn_rpe_agent2maptl[..., [2]])
            else:
                knn_idx_agent2self = None
                knn_invalid_agent2self = None
                knn_rpe_agent2self = None
                knn_idx_agent2maptl = None
                knn_invalid_agent2maptl = None
                knn_rpe_agent2maptl = None

            # ! reinforce anchors, cross attention
            if self.decoder_k_reinforce_anchor:
                knn_idx_agent2all, knn_invalid_agent2all, knn_rpe_agent2all = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid,  # [n_scene, n_emb]
                    rel_pose=rel_pose[:, -n_agent:],  # [n_scene, n_agent, n_emb, 3]
                    rel_dist=rel_dist[:, -n_agent:],  # [n_scene, n_agent, n_emb]
                    n_tgt_knn=int(self.n_tgt_knn * self.decoder_k_reinforce_anchor),
                    dist_limit=dist_limit_agent,
                )
                knn_rpe_agent2all = self.pose_rpe(xy=knn_rpe_agent2all[..., :2], dir=knn_rpe_agent2all[..., [2]])
            else:
                knn_idx_agent2all = None
                knn_invalid_agent2all = None
                knn_rpe_agent2all = None

            # ! reinforce all tokens, self attention
            if self.decoder_k_reinforce_all > 0:
                knn_idx_all2all, knn_invalid_all2all, knn_rpe_all2all = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid,  # [n_scene, n_all]
                    rel_pose=rel_pose,  # [n_scene, n_all, n_all, 3]
                    rel_dist=rel_dist,  # [n_scene, n_all, n_all]
                    n_tgt_knn=int(self.n_tgt_knn * self.decoder_k_reinforce_all),
                    dist_limit=self.dist_limit_map,
                )
                knn_rpe_all2all = self.pose_rpe(xy=knn_rpe_all2all[..., :2], dir=knn_rpe_all2all[..., [2]])
            else:
                knn_idx_all2all = None
                knn_invalid_all2all = None
                knn_rpe_all2all = None

            conf, pred = self.decoder(
                agent_type=agent_type,
                agent_valid=agent_valid,
                agent_attr=agent_attr,
                agent_emb=agent_emb,
                tl_valid=tl_valid,
                tl_emb=tl_emb,
                map_valid=map_valid,
                map_emb=map_emb,
                # reinforce tl tokens, cross attention
                knn_idx_tl2self=knn_idx_tl2self,
                knn_invalid_tl2self=knn_invalid_tl2self,
                knn_rpe_tl2self=knn_rpe_tl2self,
                knn_idx_tl2map=knn_idx_tl2map,
                knn_invalid_tl2map=knn_invalid_tl2map,
                knn_rpe_tl2map=knn_rpe_tl2map,
                # reinforce agent tokens, cross attention
                knn_idx_agent2self=knn_idx_agent2self,
                knn_invalid_agent2self=knn_invalid_agent2self,
                knn_rpe_agent2self=knn_rpe_agent2self,
                knn_idx_agent2maptl=knn_idx_agent2maptl,
                knn_invalid_agent2maptl=knn_invalid_agent2maptl,
                knn_rpe_agent2maptl=knn_rpe_agent2maptl,
                # reinforce anchor tokens, cross attention
                knn_idx_agent2all=knn_idx_agent2all,
                knn_invalid_agent2all=knn_invalid_agent2all,
                knn_rpe_agent2all=knn_rpe_agent2all,
                # reinforce all tokens, self attention
                knn_idx_all2all=knn_idx_all2all,
                knn_invalid_all2all=knn_invalid_all2all,
                knn_rpe_all2all=knn_rpe_all2all,
            )

        assert torch.isfinite(conf).all()
        assert torch.isfinite(pred).all()
        return agent_valid, conf, pred


class IntraClassEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        pl_aggr: bool,
        pose_rpe: nn.Module,
        tf_cfg: DictConfig,
        n_tgt_knn: int,
        n_layer_mlp: int,
        mlp_cfg: DictConfig,
        n_layer_tf_map: int,
        n_layer_tf_tl: int,
        n_layer_tf_agent: int,
    ) -> None:
        super().__init__()
        self.pl_aggr = pl_aggr
        self.n_tgt_knn = n_tgt_knn
        self.pose_rpe = pose_rpe

        self.fc_tl = MLP([tl_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
        if self.pl_aggr:
            self.fc_map = MLP([map_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
            self.fc_agent = MLP([agent_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
        else:
            self.point_net_map = PointNet(map_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg)
            self.point_net_agent = PointNet(agent_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg)

        self.tf_map = None
        self.tf_tl = None
        self.tf_agent = None
        if n_layer_tf_map > 0:
            self.tf_map = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, d_rpe=self.pose_rpe.out_dim, **tf_cfg
                    )
                    for _ in range(n_layer_tf_map)
                ]
            )
        if n_layer_tf_tl > 0:
            self.tf_tl = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, d_rpe=self.pose_rpe.out_dim, **tf_cfg
                    )
                    for _ in range(n_layer_tf_tl)
                ]
            )
        if n_layer_tf_agent > 0:
            self.tf_agent = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, d_rpe=self.pose_rpe.out_dim, **tf_cfg
                    )
                    for _ in range(n_layer_tf_agent)
                ]
            )

    def forward(
        self,
        inference_repeat_n: int,
        inference_cache_map: bool,
        agent_valid: Tensor,
        agent_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        rel_pose: Tensor,
        rel_dist: Tensor,
        dist_limit_map: float,
        dist_limit_tl: float,
        dist_limit_agent: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            agent_valid: [n_scene, n_agent, (n_step_hist)]
            agent_attr: [n_scene, n_agent, (n_step_hist), agent_attr_dim], local wrt. the current step.
            map_valid: [n_scene, n_map, (n_pl_node)]
            map_attr: [n_scene, n_map, (n_pl_node), map_attr_dim], local wrt. the first node.
            tl_valid: [n_scene, n_tl]
            tl_attr: [n_scene, n_tl, tl_attr_dim]
            rel_pose: [n_scene, n_map+n_tl+n_agent, n_map+n_tl+n_agent, 3]
            rel_dist: [n_scene, n_map+n_tl+n_agent, n_map+n_tl+n_agent]

        Returns:
            map_emb: [n_scene, n_map, hidden_dim]
            map_valid: [n_scene, n_map]
            tl_emb: [n_scene, n_tl, hidden_dim] or tl_attr
            tl_valid: [n_scene, n_tl]
            agent_emb: [n_scene, n_agent, hidden_dim]
            agent_valid: [n_scene, n_agent]
        """
        for _ in range(inference_repeat_n):
            n_scene, n_tl = tl_valid.shape
            n_map = map_valid.shape[1]
            n_agent = agent_valid.shape[1]
            _idx_scene = torch.arange(n_scene)[:, None, None]  # [n_scene, 1, 1]

        # ! map
        _n_repeat_map = 1 if inference_cache_map else inference_repeat_n
        for _ in range(_n_repeat_map):
            map_emb, map_valid_reduced = self._mlp_map(map_attr, map_valid)  # [n_scene, n_map, hidden_dim]
            if self.tf_map is not None:
                _map_invalid = ~map_valid_reduced
                _map_idx_knn, _map_invalid_knn, _map_rpe_knn = get_tgt_knn_idx(
                    _map_invalid,
                    rel_pose[:, :n_map, :n_map],
                    rel_dist[:, :n_map, :n_map],
                    self.n_tgt_knn,
                    dist_limit=dist_limit_map,
                )
                _rpe = self.pose_rpe(xy=_map_rpe_knn[..., :2], dir=_map_rpe_knn[..., [2]])
                _idx_map = torch.arange(n_map)[None, :, None]  # [1, n_map, 1]
                for mod in self.tf_map:
                    _tgt = map_emb.unsqueeze(1).expand(-1, n_map, -1, -1)
                    if _map_idx_knn is not None:
                        _tgt = _tgt[_idx_scene, _idx_map, _map_idx_knn]
                    map_emb, _ = mod(
                        src=map_emb,  # [n_scene, n_map, hidden_dim]
                        src_padding_mask=_map_invalid,  # [n_scene, n_map]
                        tgt=_tgt,
                        tgt_padding_mask=_map_invalid_knn,  # [n_scene, n_map, n_tgt_knn]
                        rpe=_rpe,  # [n_scene, n_map, n_tgt_knn, d_rpe]
                    )

        # ! traffic lights
        for _ in range(inference_repeat_n):
            tl_emb = self.fc_tl(tl_attr, tl_valid)  # [n_scene, n_tl, hidden_dim]
            if self.tf_tl is not None:
                _tl_invalid = ~tl_valid
                _tl_idx_knn, _tl_invalid_knn, _tl_rpe_knn = get_tgt_knn_idx(
                    _tl_invalid,
                    rel_pose[:, n_map : n_map + n_tl, n_map : n_map + n_tl],
                    rel_dist[:, n_map : n_map + n_tl, n_map : n_map + n_tl],
                    self.n_tgt_knn,
                    dist_limit=dist_limit_tl,
                )
                _rpe = self.pose_rpe(xy=_tl_rpe_knn[..., :2], dir=_tl_rpe_knn[..., [2]])
                _idx_tl = torch.arange(n_tl)[None, :, None]  # [1, n_tl, 1]
                for mod in self.tf_tl:
                    _tgt = tl_emb.unsqueeze(1).expand(-1, n_tl, -1, -1)
                    if _tl_idx_knn is not None:
                        _tgt = _tgt[_idx_scene, _idx_tl, _tl_idx_knn]
                    tl_emb, _ = mod(
                        src=tl_emb,  # [n_scene, n_tl, hidden_dim]
                        src_padding_mask=_tl_invalid,  # [n_scene, n_tl]
                        tgt=_tgt,
                        tgt_padding_mask=_tl_invalid_knn,  # [n_scene, n_tl, n_tgt_knn]
                        rpe=_rpe,  # [n_scene, n_tl, n_tgt_knn, d_rpe]
                    )

        # ! agents
        for _ in range(inference_repeat_n):
            agent_emb, agent_valid_reduced = self._mlp_agent(agent_attr, agent_valid)  # [n_scene, n_agent, hidden_dim]
            if self.tf_agent is not None:
                _agent_invalid = ~agent_valid_reduced
                _agent_idx_knn, _agent_invalid_knn, _agent_rpe_knn = get_tgt_knn_idx(
                    _agent_invalid,
                    rel_pose[:, -n_agent:, -n_agent:],
                    rel_dist[:, -n_agent:, -n_agent:],
                    self.n_tgt_knn,
                    dist_limit=dist_limit_agent,
                )
                _rpe = self.pose_rpe(xy=_agent_rpe_knn[..., :2], dir=_agent_rpe_knn[..., [2]])
                _idx_agent = torch.arange(n_agent)[None, :, None]  # [1, n_agent, 1]
                for mod in self.tf_agent:
                    _tgt = agent_emb.unsqueeze(1).expand(-1, n_agent, -1, -1)
                    if _agent_idx_knn is not None:
                        _tgt = _tgt[_idx_scene, _idx_agent, _agent_idx_knn]
                    agent_emb, _ = mod(
                        src=agent_emb,  # [n_scene, n_agent, hidden_dim]
                        src_padding_mask=_agent_invalid,  # [n_scene, n_agent]
                        tgt=_tgt,
                        tgt_padding_mask=_agent_invalid_knn,  # [n_scene, n_agent, n_tgt_knn]
                        rpe=_rpe,  # [n_scene, n_agent, n_tgt_knn, d_rpe]
                    )

        return map_emb, map_valid_reduced, tl_emb, tl_valid, agent_emb, agent_valid_reduced

    def _mlp_agent(self, agent_attr: Tensor, agent_valid: Tensor) -> Tuple[Tensor, Tensor]:
        if self.pl_aggr:
            agent_emb = self.fc_agent(agent_attr, agent_valid)
        else:
            agent_emb, agent_valid = self.point_net_agent(agent_attr, agent_valid)
        return agent_emb, agent_valid

    def _mlp_map(self, map_attr: Tensor, map_valid: Tensor) -> Tuple[Tensor, Tensor]:
        if self.pl_aggr:
            map_emb = self.fc_map(map_attr, map_valid)  # [n_scene, n_map, hidden_dim]
        else:
            # map_attr: [n_scene, n_map, n_pl_node, map_attr_dim], map_valid: [n_scene, n_map, n_pl_node]
            map_emb, map_valid = self.point_net_map(map_attr, map_valid)  # [n_scene, n_map, hidden_dim]
        return map_emb, map_valid
