# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# KNARPE
class AttentionRPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout_p: float = 0.0,
        bias: bool = True,
        d_rpe: int = -1,
        apply_q_rpe: bool = False,
    ) -> None:
        """
        Always batch first. Always src and tgt have the same d_model.
        """
        super(AttentionRPE, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.apply_q_rpe = apply_q_rpe
        self.d_rpe = d_rpe

        assert self.d_head * n_head == d_model, "d_model must be divisible by n_head"

        if self.d_rpe > 0:
            n_project_rpe = 3 if apply_q_rpe else 2
            self.mlp_rpe = nn.Linear(d_rpe, n_project_rpe * d_model, bias=bias)

        self.in_proj_weight = nn.Parameter(torch.empty((3 * d_model, d_model)))
        self.out_proj_weight = nn.Parameter(torch.empty((d_model, d_model)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
            self.out_proj_bias = nn.Parameter(torch.empty(d_model))
        else:
            self.register_parameter("in_proj_bias", None)
            self.register_parameter("out_proj_bias", None)

        self.dropout = nn.Dropout(p=dropout_p, inplace=False) if dropout_p > 0 else None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
        if self.out_proj_bias is not None:
            nn.init.constant_(self.out_proj_bias, 0.0)

    def forward(
        self,
        src: Tensor,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rpe: Optional[Tensor] = None,
        need_weights=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            src: [n_batch, n_src, d_model]
            tgt: [n_batch, (n_src), n_tgt, d_model], None for self attention, (n_src) if using rpe.
            tgt_padding_mask: [n_batch, (n_src), n_tgt], bool, if True, tgt is invalid, (n_src) if using rpe.
            attn_mask: [n_batch, n_src, n_tgt], bool, if True, attn is disabled for that pair of src/tgt.
            rpe: [n_batch, n_src, n_tgt, d_rpe]

        Returns:
            out: [n_batch, n_src, d_model]
            attn_weights: [n_batch, n_src, n_tgt] if need_weights else None

        Remarks:
            absoulte_pe should be already added to src/tgt.
            if for a batch entry all tgt are invalid, then returns 0 for that batch entry.
        """
        n_batch, n_src, _ = src.shape
        if tgt is None:
            n_tgt = n_src
            # self-attention
            qkv = F.linear(src, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            n_tgt = tgt.shape[-2]
            # encoder-decoder attention
            w_src, w_tgt = self.in_proj_weight.split([self.d_model, self.d_model * 2])
            b_src, b_tgt = None, None
            if self.in_proj_bias is not None:
                b_src, b_tgt = self.in_proj_bias.split([self.d_model, self.d_model * 2])
            q = F.linear(src, w_src, b_src)
            kv = F.linear(tgt, w_tgt, b_tgt)
            k, v = kv.chunk(2, dim=-1)
        # q: [n_batch, n_src, d_model], k,v: [n_batch, (n_src), n_tgt, d_model]

        attn_invalid_mask = None  # [n_batch, n_src, n_tgt]
        if tgt_padding_mask is not None:  # [n_batch, n_tgt], bool
            attn_invalid_mask = tgt_padding_mask
            if attn_invalid_mask.dim() == 2:
                attn_invalid_mask = attn_invalid_mask.unsqueeze(1).expand(-1, n_src, -1)
        if attn_mask is not None:  # [n_batch, n_src, n_tgt], bool
            if attn_invalid_mask is None:
                attn_invalid_mask = attn_mask
            else:
                attn_invalid_mask = attn_invalid_mask | attn_mask

        mask_no_tgt_valid = None  # [n_batch, n_src]
        if attn_invalid_mask is not None:
            mask_no_tgt_valid = attn_invalid_mask.all(-1)
            if mask_no_tgt_valid.any():
                attn_invalid_mask = attn_invalid_mask & (~mask_no_tgt_valid.unsqueeze(-1))  # to avoid softmax nan
            else:
                mask_no_tgt_valid = None

        # get attn: [n_batch, n_head, n_src, n_tgt]
        if rpe is None:
            if k.dim() == 3:
                # ! normal attention; q: [n_batch, n_src, d_model], k,v: [n_batch, n_tgt, d_model]
                q = q.view(n_batch, n_src, self.n_head, self.d_head).transpose(1, 2).contiguous()
                k = k.view(n_batch, n_tgt, self.n_head, self.d_head).transpose(1, 2).contiguous()
                v = v.view(n_batch, n_tgt, self.n_head, self.d_head).transpose(1, 2).contiguous()
                attn = torch.matmul(q, k.transpose(-2, -1))  # [n_batch, n_head, n_src, n_tgt]
                # q: [n_batch, n_head, n_src, d_head], k,v: [n_batch, n_head, n_tgt, d_head]
            else:
                # ! KNN attention; q: [n_batch, n_src, d_model], k,v: [n_batch, n_src, n_tgt, d_model]
                # k,v: [n_batch, n_src, n_tgt, d_model] -> [n_batch, n_head, n_src, n_tgt_knn, d_head]
                k = k.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
                v = v.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
                # [n_batch, n_src, d_model] -> [n_batch, n_head, n_src, 1, d_head]
                q = q.view(n_batch, n_src, self.n_head, self.d_head).transpose(1, 2).unsqueeze(3)
                attn = torch.sum(q * k, dim=-1)  # [n_batch, n_head, n_src, n_tgt_knn]
        else:
            # ! rpe attention; q: [n_batch, n_src, d_model], k,v: [n_batch, n_tgt, d_model]
            assert self.d_rpe > 0
            # k,v: [n_batch, n_src, n_tgt, d_model] -> [n_batch, n_head, n_src, n_tgt_knn, d_head]
            k = k.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            v = v.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            # [n_batch, n_src, d_model] -> [n_batch, n_head, n_src, 1, d_head]
            q = q.view(n_batch, n_src, self.n_head, self.d_head).transpose(1, 2).unsqueeze(3)

            # project rpe to rpe_q, rpe_k, rpe_v: [n_batch, n_head, n_src, n_tgt, d_head]
            rpe = self.mlp_rpe(rpe)
            if self.apply_q_rpe:
                rpe_q, rpe_k, rpe_v = rpe.chunk(3, dim=-1)
                rpe_q = rpe_q.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            else:
                rpe_k, rpe_v = rpe.chunk(2, dim=-1)
            rpe_k = rpe_k.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            rpe_v = rpe_v.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)

            # get attn: [n_batch, n_head, n_src, n_tgt]
            if self.apply_q_rpe:
                attn = torch.sum((q + rpe_q) * (k + rpe_k), dim=-1)
                # attn = torch.sum((q + rpe_q) * (k + rpe_k) - rpe_q * rpe_k, dim=-1)
            else:
                attn = torch.sum(q * (k + rpe_k), dim=-1)
            # q: [n_batch, n_head, n_src, 1, d_head]
            # k,v: [n_batch, n_head, n_src, n_tgt, d_head]
            # rpe_q, rpe_k, rpe_v: [n_batch, n_head, n_src, n_tgt, d_head]

        if attn_invalid_mask is not None:
            # attn_invalid_mask: [n_batch, n_src, n_tgt], attn: [n_batch, n_head, n_src, n_tgt]
            attn = attn.masked_fill(attn_invalid_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(attn / math.sqrt(self.d_head), dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)

        # attn: [n_batch, n_head, n_src, n_tgt]
        if rpe is None:
            if v.dim() == 4:
                out = torch.matmul(attn, v)  # v, [n_batch, n_head, n_tgt, d_head]
            else:
                out = torch.sum(v * attn.unsqueeze(-1), dim=3)  # v: [n_batch, n_head, n_src, n_tgt, d_head]
        else:
            # v, rpe_v: [n_batch, n_head, n_src, n_tgt, d_head]
            out = torch.sum((v + rpe_v) * attn.unsqueeze(-1), dim=3)

        # out: [n_batch, n_head, n_src, d_head]
        out = out.transpose(1, 2).flatten(2, 3)  # [n_batch, n_src, d_model]
        out = F.linear(out, self.out_proj_weight, self.out_proj_bias)

        if mask_no_tgt_valid is not None:
            # mask_no_tgt_valid: [n_batch, n_src], out: [n_batch, n_src, d_model]
            out = out.masked_fill(mask_no_tgt_valid.unsqueeze(-1), 0)

        if need_weights:
            attn_weights = attn.mean(1)  # [n_batch, n_src, n_tgt]
            if mask_no_tgt_valid is not None:
                attn_weights = attn_weights.masked_fill(mask_no_tgt_valid.unsqueeze(-1), 0)
            return out, attn_weights
        else:
            return out, None
