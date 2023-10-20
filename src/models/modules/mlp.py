# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import List, Tuple, Union, Optional
from torch import Tensor, nn


def _get_activation(activation: str, inplace: bool) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation == "gelu":
        return nn.GELU()
    raise RuntimeError("activation {} not implemented".format(activation))


class MLP(nn.Module):
    def __init__(
        self,
        fc_dims: Union[List, Tuple],
        dropout_p: Optional[float] = None,
        use_layernorm: bool = False,
        activation: str = "relu",
        end_layer_activation: bool = True,
        init_weight_norm: bool = False,
        init_bias: Optional[float] = None,
        use_batchnorm: bool = False,
    ) -> None:
        super(MLP, self).__init__()
        assert len(fc_dims) >= 2
        assert not (use_layernorm and use_batchnorm)
        layers: List[nn.Module] = []
        for i in range(0, len(fc_dims) - 1):

            fc = nn.Linear(fc_dims[i], fc_dims[i + 1])

            if init_weight_norm:
                fc.weight.data *= 1.0 / fc.weight.norm(dim=1, p=2, keepdim=True)
            if init_bias is not None and i == len(fc_dims) - 2:
                fc.bias.data *= 0
                fc.bias.data += init_bias

            layers.append(fc)

            if i < len(fc_dims) - 2:
                if use_layernorm:
                    layers.append(nn.LayerNorm(fc_dims[i + 1]))
                elif use_batchnorm:
                    layers.append(nn.BatchNorm1d(fc_dims[i + 1]))
                if dropout_p is not None:
                    layers.append(nn.Dropout(p=dropout_p))
                layers.append(_get_activation(activation, inplace=True))
            if i == len(fc_dims) - 2:
                if end_layer_activation:
                    if use_layernorm:
                        layers.append(nn.LayerNorm(fc_dims[i + 1]))
                    elif use_batchnorm:
                        layers.append(nn.BatchNorm1d(fc_dims[i + 1]))
                    if dropout_p is not None:
                        layers.append(nn.Dropout(p=dropout_p))
                    self.end_layer_activation = _get_activation(activation, inplace=True)
                else:
                    self.end_layer_activation = None

        self.input_dim = fc_dims[0]
        self.output_dim = fc_dims[-1]
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None, fill_invalid: float = 0.0) -> Tensor:
        """
        Args:
            x: [..., input_dim]
            valid_mask: [...]
        Returns:
            x: [..., output_dim]
        """
        x = self.fc_layers(x.flatten(0, -2)).view(*x.shape[:-1], self.output_dim)
        if valid_mask is not None:
            x.masked_fill_(~valid_mask.unsqueeze(-1), fill_invalid)
        if self.end_layer_activation is not None:
            self.end_layer_activation(x)
        return x
