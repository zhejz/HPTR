# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Tuple, List, Optional
import torch
from torch import Tensor, nn
from .mlp import MLP


class PointNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layer: int = 3,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        end_layer_activation: bool = True,
        dropout_p: Optional[float] = None,
        pool_mode: str = "max",  # max, mean, first
    ) -> None:
        super().__init__()
        self.pool_mode = pool_mode
        self.input_mlp = MLP(
            [input_dim, hidden_dim, hidden_dim],
            dropout_p=dropout_p,
            use_layernorm=use_layernorm,
            use_batchnorm=use_batchnorm,
        )

        mlp_layers: List[nn.Module] = []
        for _ in range(n_layer - 2):
            mlp_layers.append(
                MLP(
                    [hidden_dim, hidden_dim // 2],
                    dropout_p=dropout_p,
                    use_layernorm=use_layernorm,
                    use_batchnorm=use_batchnorm,
                )
            )
        mlp_layers.append(
            MLP(
                [hidden_dim, hidden_dim // 2],
                dropout_p=dropout_p,
                use_layernorm=use_layernorm,
                use_batchnorm=use_batchnorm,
                end_layer_activation=end_layer_activation,
            )
        )
        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self, x: Tensor, valid: Tensor) -> Tuple[Tensor, Tensor]:
        """c.f. VectorNet and SceneTransformer, Aggregate polyline/track level feature.

        Args:
            x: [n_batch, n_pl, n_pl_node, attr_dim]
            valid: [n_batch, n_pl, n_pl_node] bool

        Returns:
            emb: [n_batch, n_pl, hidden_dim]
            emb_valid: [n_batch, n_pl]
        """
        x = self.input_mlp(x, valid)  # [n_batch, n_pl, n_pl_node, hidden_dim]

        for mlp in self.mlp_layers:
            feature_encoded = mlp(x, valid, float("-inf"))  # [n_batch, n_pl, n_pl_node, hidden_dim//2]
            feature_pooled = feature_encoded.amax(dim=2, keepdim=True)
            x = torch.cat((feature_encoded, feature_pooled.expand(-1, -1, valid.shape[-1], -1)), dim=-1)

        if self.pool_mode == "max":
            x.masked_fill_(~valid.unsqueeze(-1), float("-inf"))  # [n_batch, n_pl, n_pl_node, hidden_dim]
            emb = x.amax(dim=2, keepdim=False)  # [n_batch, n_pl, hidden_dim]
        elif self.pool_mode == "first":
            emb = x[:, :, 0]
        elif self.pool_mode == "mean":
            x.masked_fill_(~valid.unsqueeze(-1), 0)  # [n_batch, n_pl, n_pl_node, hidden_dim]
            emb = x.sum(dim=2, keepdim=False)  # [batch_size, n_pl, hidden_dim]
            emb = emb / (valid.sum(dim=-1, keepdim=True) + torch.finfo(x.dtype).eps)

        emb_valid = valid.any(-1)  # [n_batch, n_pl]
        emb = emb.masked_fill(~emb_valid.unsqueeze(-1), 0)  # [n_batch, n_pl, hidden_dim]
        return emb, emb_valid
