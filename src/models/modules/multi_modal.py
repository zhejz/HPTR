# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from torch import Tensor, nn
from .mlp import MLP


class MultiModalAnchors(nn.Module):
    def __init__(
        self,
        mode_emb: str,
        mode_init: str,
        hidden_dim: int,
        n_pred: int,
        emb_dim: int,
        use_agent_type: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_pred = n_pred
        self.use_agent_type = use_agent_type

        self.mode_init = mode_init
        n_anchors = 3 if use_agent_type else 1
        if self.mode_init == "xavier":
            self.anchors = torch.empty((n_anchors, n_pred, hidden_dim))
            nn.init.xavier_normal_(self.anchors)
            self.anchors = nn.Parameter(self.anchors * scale, requires_grad=True)
        elif self.mode_init == "uniform":
            self.anchors = torch.empty((n_anchors, n_pred, hidden_dim))
            self.anchors.uniform_(-scale, scale)
            self.anchors = nn.Parameter(self.anchors, requires_grad=True)
        elif self.mode_init == "randn":
            self.anchors = nn.Parameter(torch.randn([n_anchors, n_pred, hidden_dim]) * scale, requires_grad=True)
        else:
            raise NotImplementedError

        self.mode_emb = mode_emb
        if self.mode_emb == "linear":
            self.mlp_anchor = nn.Linear(self.anchors.shape[-1] + emb_dim, hidden_dim, bias=False)
        elif self.mode_emb == "mlp":
            self.mlp_anchor = MLP([self.anchors.shape[-1] + emb_dim] + [hidden_dim] * 2, end_layer_activation=False)
        elif self.mode_emb == "add" or self.mode_emb == "none":
            assert emb_dim == hidden_dim
            if self.anchors.shape[-1] != hidden_dim:
                self.mlp_anchor = nn.Linear(self.anchors.shape[-1], hidden_dim, bias=False)
            else:
                self.mlp_anchor = None
        else:
            raise NotImplementedError

    def forward(self, valid: Tensor, emb: Tensor, agent_type: Tensor) -> Tensor:
        """
        Args:
            valid: [n_scene*n_agent]
            emb: [n_scene*n_agent, in_dim]
            agent_type: [n_scene*n_agent, 3]

        Returns:
            mm_emb: [n_scene*n_agent, n_pred, out_dim]
        """
        # [n_scene*n_agent, n_pred, emb_dim]
        if self.use_agent_type:
            anchors = (self.anchors.unsqueeze(0) * agent_type[:, :, None, None]).sum(1)
        else:
            anchors = self.anchors.expand(valid.shape[0], -1, -1)

        if self.mode_emb == "linear" or self.mode_emb == "mlp":
            # [n_scene*n_agent, n_pred, hidden_dim + emb_dim]
            mm_emb = torch.cat([emb.unsqueeze(1).expand(-1, self.n_pred, -1), anchors], dim=-1)
            mm_emb = self.mlp_anchor(mm_emb)
        elif self.mode_emb == "add":
            if self.mlp_anchor is not None:
                anchors = self.mlp_anchor(anchors)  # [n_scene*n_agent, n_pred, hidden_dim]
            mm_emb = emb.unsqueeze(1) + anchors
        elif self.mode_emb == "none":
            if self.mlp_anchor is not None:
                anchors = self.mlp_anchor(anchors)  # [n_scene*n_agent, n_pred, hidden_dim]
            mm_emb = anchors
        return mm_emb.masked_fill(~valid[:, None, None], 0)
