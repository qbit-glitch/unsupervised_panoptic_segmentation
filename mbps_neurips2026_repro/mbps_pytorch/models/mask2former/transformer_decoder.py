"""Mask2Former Transformer Decoder with masked cross-attention.

Ported from: refs/dinov3/dinov3/eval/segmentation/models/heads/mask2former_transformer_decoder.py
Adapted: reduced dimensions (256 hidden, 8 heads, 1024 FFN) for M4 Pro memory.
All attention uses standard nn.MultiheadAttention — MPS-safe.

The key Mask2Former mechanism: each decoder layer predicts intermediate masks,
which are used as attention masks in the next layer's cross-attention, so queries
only attend to their predicted mask regions (masked cross-attention).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class SelfAttentionLayer(nn.Module):
    """Standard self-attention with optional pre/post LayerNorm."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0,
                 activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> Tensor:
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            return tgt + self.dropout(tgt2)
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            return self.norm(tgt)


class CrossAttentionLayer(nn.Module):
    """Cross-attention with optional pre/post LayerNorm."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0,
                 activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: Tensor, memory: Tensor,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> Tensor:
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            return tgt + self.dropout(tgt2)
        else:
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout(tgt2)
            return self.norm(tgt)


class FFNLayer(nn.Module):
    """Feed-forward network with optional pre/post LayerNorm."""

    def __init__(self, d_model: int, dim_feedforward: int = 2048,
                 dropout: float = 0.0, activation: str = "relu",
                 normalize_before: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt: Tensor) -> Tensor:
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            return tgt + self.dropout(tgt2)
        else:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout(tgt2)
            return self.norm(tgt)


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    """Mask2Former transformer decoder with masked cross-attention.

    Key mechanism: after each decoder layer, predicts intermediate masks.
    These masks become attention masks for the next cross-attention layer,
    so queries attend only to their predicted spatial regions.

    Args:
        in_channels: Input feature channels from pixel decoder.
        num_classes: Number of semantic classes.
        hidden_dim: Transformer hidden dimension.
        num_queries: Number of learnable queries.
        nheads: Number of attention heads.
        dim_feedforward: FFN intermediate dimension.
        dec_layers: Number of decoder layers.
        pre_norm: Use pre-LayerNorm (False = post-norm).
        mask_dim: Mask feature dimension for dot-product mask prediction.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        dec_layers: int = 9,
        pre_norm: bool = False,
        mask_dim: int = 256,
    ):
        super().__init__()

        # Positional encoding
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_feature_levels = 3  # Always use 3 scales

        # Decoder layers: cross-attn → self-attn → FFN
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(dec_layers):
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(hidden_dim, nheads, dropout=0.0,
                                    normalize_before=pre_norm)
            )
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(hidden_dim, nheads, dropout=0.0,
                                   normalize_before=pre_norm)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(hidden_dim, dim_feedforward, dropout=0.0,
                         normalize_before=pre_norm)
            )

        self.post_norm = nn.LayerNorm(hidden_dim)

        # Learnable queries
        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Level embedding for multi-scale features
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # Input projection (if in_channels != hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim:
                proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                nn.init.kaiming_uniform_(proj.weight, a=1)
                nn.init.constant_(proj.bias, 0)
                self.input_proj.append(proj)
            else:
                self.input_proj.append(nn.Identity())

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(
        self,
        multi_scale_features: list[torch.Tensor],
        mask_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with masked cross-attention.

        Args:
            multi_scale_features: List of 3 feature maps at [1/32, 1/16, 1/8].
            mask_features: (B, mask_dim, H/4, W/4) for mask dot product.

        Returns:
            Dict with:
                pred_logits: (B, Q, num_classes+1) from last layer.
                pred_masks: (B, Q, H/4, W/4) from last layer.
                aux_outputs: List of dicts from intermediate layers.
        """
        assert len(multi_scale_features) == self.num_feature_levels

        # Prepare multi-scale src + positional encodings
        src_list = []
        pos_list = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            pos_list.append(
                self.pe_layer(multi_scale_features[i], None).flatten(2)
            )
            src_list.append(
                self.input_proj[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )
            # Flatten NxCxHxW → HW×N×C (seq_len, batch, dim)
            pos_list[-1] = pos_list[-1].permute(2, 0, 1)
            src_list[-1] = src_list[-1].permute(2, 0, 1)

        _, bs, _ = src_list[0].shape

        # Initialize queries: Q×N×C
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # Initial prediction on learnable query features
        outputs_class, outputs_mask, attn_mask = self._forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # Decoder layers with round-robin multi-scale features
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            # Prevent all-masked rows (would cause NaN in softmax)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # Cross-attention (query attends to spatial features via mask)
            output = self.transformer_cross_attention_layers[i](
                output, src_list[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos_list[level_index],
                query_pos=query_embed,
            )

            # Self-attention (queries interact with each other)
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None,
                query_pos=query_embed,
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            # Predict masks for next layer's attention mask
            next_size = size_list[(i + 1) % self.num_feature_levels]
            outputs_class, outputs_mask, attn_mask = self._forward_prediction_heads(
                output, mask_features, attn_mask_target_size=next_size
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        return {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": [
                {"pred_logits": c, "pred_masks": m}
                for c, m in zip(predictions_class[:-1], predictions_mask[:-1])
            ],
        }

    def _forward_prediction_heads(
        self, output: Tensor, mask_features: Tensor,
        attn_mask_target_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict class logits, mask logits, and next-layer attention mask.

        Args:
            output: (Q, B, D) decoder output.
            mask_features: (B, D, H/4, W/4) pixel features.
            attn_mask_target_size: (H, W) for attention mask interpolation.

        Returns:
            outputs_class: (B, Q, num_classes+1)
            outputs_mask: (B, Q, H/4, W/4)
            attn_mask: (B*nheads, Q, H*W) bool mask for cross-attention.
        """
        decoder_output = self.post_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # (B, Q, D)

        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # Generate attention mask for next cross-attention layer
        # Interpolate mask predictions to target feature map size
        attn_mask = F.interpolate(
            outputs_mask, size=attn_mask_target_size,
            mode="bilinear", align_corners=False,
        )
        # True = masked (not allowed to attend), threshold at 0.5
        attn_mask = (
            attn_mask.sigmoid().flatten(2)
            .unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1) < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask
