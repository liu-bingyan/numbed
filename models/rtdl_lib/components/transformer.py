import time
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.rtdl_lib.utils import _all_or_none, _is_glu_activation, _make_nn_module

ModuleType = Union[str, Type[nn.Module]]

_INTERNAL_ERROR_MESSAGE = 'An internal error occurred. Please, open an issue via ' \
                         'https://github.com/Yura52/rtdl/issues/new/choose'

class MultiheadAttention(nn.Module):
    """Multihead Attention (MHA) module.

    This module is used in the `Transformer` class.
    """

    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Args:
            d_token: the size of one token
            n_heads: the number of attention heads
            dropout: the dropout probability
            bias: if True, then the linear layers will have bias
            initialization: initialization policy for parameters. Must be one of
                :code:`['kaiming', 'xavier']`. In [gorishniy2021revisiting], the 'kaiming'
                initialization was used.
        """
        super().__init__()
        assert d_token % n_heads == 0, 'd_token must be divisible by n_heads'
        d_head = d_token // n_heads
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_token = d_token
        self.initialization = initialization

        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.W_out = nn.Linear(d_token, d_token, bias)
        self.dropout = nn.Dropout(dropout)

        for parameter in [self.W_q.weight, self.W_k.weight, self.W_v.weight, self.W_out.weight]:
            if initialization == 'kaiming':
                nn.init.kaiming_uniform_(parameter, a=2 ** 0.5)
            elif initialization == 'xavier':
                nn.init.xavier_uniform_(parameter)
            else:
                raise ValueError(f'Unknown initialization: {initialization}')

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: Optional[nn.Linear] = None,
        value_compression: Optional[nn.Linear] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Perform the forward pass.

        Args:
            x_q: the query tensor of shape :code:`(batch_size, n_tokens, d_token)`
            x_kv: the key-value tensor of shape :code:`(batch_size, n_tokens, d_token)`
            key_compression: an optional linear layer for compressing keys
            value_compression: an optional linear layer for compressing values
        Returns:
            the output tensor of shape :code:`(batch_size, n_tokens, d_token)` and the attention
            weights of shape :code:`(batch_size, n_heads, n_tokens, n_tokens)`
        """
        q = self.W_q(x_q)
        k = self.W_k(x_kv)
        v = self.W_v(x_kv)

        if key_compression is not None:
            k = key_compression(k)
        if value_compression is not None:
            v = value_compression(v)

        batch_size, n_tokens, d_token = q.shape
        q = q.view(batch_size, n_tokens, self.n_heads, self.d_head)
        k = k.view(batch_size, n_tokens, self.n_heads, self.d_head)
        v = v.view(batch_size, n_tokens, self.n_heads, self.d_head)

        q = q.transpose(1, 2)  # (batch_size, n_heads, n_tokens, d_head)
        k = k.transpose(1, 2)  # (batch_size, n_heads, n_tokens, d_head)
        v = v.transpose(1, 2)  # (batch_size, n_heads, n_tokens, d_head)

        scale = self.d_head ** -0.5
        attention = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch_size, n_heads, n_tokens, n_tokens)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        x = torch.matmul(attention, v)  # (batch_size, n_heads, n_tokens, d_head)
        x = x.transpose(1, 2).contiguous()  # (batch_size, n_tokens, n_heads, d_head)
        x = x.view(batch_size, n_tokens, d_token)  # (batch_size, n_tokens, d_token)
        x = self.W_out(x)
        return x, attention

class Transformer(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS: ClassVar[Dict[str, bool]] = {
        'first_prenormalization': True,
        'prenormalization': True,
    }

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation(activation) else 1),
                bias_first,
            )
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x

    class Head(nn.Module):
        """The final module of the `Transformer` that performs BERT-like inference."""

        def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: ModuleType,
            normalization: ModuleType,
            d_out: int,
        ):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            x = x[:, -1]
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_token: int,
        n_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        last_layer_query_idx: Union[None, List[int], slice],
        n_tokens: Optional[int],
        kv_compression_ratio: Optional[float],
        kv_compression_sharing: Optional[str],
        head_activation: ModuleType,
        head_normalization: ModuleType,
        d_out: int,
    ) -> None:
        super().__init__()
        if isinstance(last_layer_query_idx, int):
            raise ValueError(
                'last_layer_query_idx must be None, list[int] or slice. '
                f'Do you mean last_layer_query_idx=[{last_layer_query_idx}] ?'
            )
        if not prenormalization:
            assert (
                not first_prenormalization
            ), 'If `prenormalization` is False, then `first_prenormalization` must be False'
        assert _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing]), (
            'If any of the following arguments is (not) None, then all of them must (not) be None: '
            'n_tokens, kv_compression_ratio, kv_compression_sharing'
        )
        assert kv_compression_sharing in [None, 'headwise', 'key-value', 'layerwise']
        if not prenormalization:
            if self.WARNINGS['prenormalization']:
                warnings.warn(
                    'prenormalization is set to False. Are you sure about this? '
                    'The training can become less stable. '
                    'You can turn off this warning by tweaking the '
                    'rtdl.Transformer.WARNINGS dictionary.',
                    UserWarning,
                )
            assert (
                not first_prenormalization
            ), 'If prenormalization is False, then first_prenormalization is ignored and must be set to False'
        if (
            prenormalization
            and first_prenormalization
            and self.WARNINGS['first_prenormalization']
        ):
            warnings.warn(
                'first_prenormalization is set to True. Are you sure about this? '
                'For example, the vanilla FTTransformer with '
                'first_prenormalization=True performs SIGNIFICANTLY worse. '
                'You can turn off this warning by tweaking the '
                'rtdl.Transformer.WARNINGS dictionary.',
                UserWarning,
            )
            time.sleep(3)

        def make_kv_compression():
            assert (
                n_tokens and kv_compression_ratio
            ), _INTERNAL_ERROR_MESSAGE  # for mypy
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False)

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression_ratio and kv_compression_sharing == 'layerwise'
            else None
        )

        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx

        self.blocks = nn.ModuleList([])
        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    'ffn': Transformer.FFN(
                        d_token=d_token,
                        d_hidden=ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=ffn_activation,
                    ),
                    'attention_residual_dropout': nn.Dropout(residual_dropout),
                    'ffn_residual_dropout': nn.Dropout(residual_dropout),
                    'output': nn.Identity(),  # for hooks-based introspection
                }
            )
            if layer_idx or not prenormalization or first_prenormalization:
                layer['attention_normalization'] = _make_nn_module(
                    attention_normalization, d_token
                )
            layer['ffn_normalization'] = _make_nn_module(ffn_normalization, d_token)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert (
                        kv_compression_sharing == 'key-value'
                    ), _INTERNAL_ERROR_MESSAGE
            self.blocks.append(layer)

        self.head = Transformer.Head(
            d_in=d_token,
            d_out=d_out,
            bias=True,
            activation=head_activation,  # type: ignore
            normalization=head_normalization if prenormalization else 'Identity',
        )

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, layer, stage, x):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f'{stage}_normalization'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual):
        assert stage in ['attention', 'ffn'], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f'{stage}_residual_dropout'](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'{stage}_normalization'](x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.ndim == 3
        ), 'The input must have 3 dimensions: (n_objects, n_tokens, d_token)'
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)

            query_idx = (
                self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            )
            x_residual = self._start_residual(layer, 'attention', x)
            x_residual, _ = layer['attention'](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if query_idx is not None:
                x = x[:, query_idx]
            x = self._end_residual(layer, 'attention', x, x_residual)

            x_residual = self._start_residual(layer, 'ffn', x)
            x_residual = layer['ffn'](x_residual)
            x = self._end_residual(layer, 'ffn', x, x_residual)
            x = layer['output'](x)

        x = self.head(x)
        return x 