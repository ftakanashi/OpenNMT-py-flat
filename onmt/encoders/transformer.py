"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask, sequence_mask_by_tag


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions,
                 # wei 20200730
                 flat_layers, flat_options,
                 nfr_tag_mode, d_tag
                 # end wei
                 ):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings

        # wei 20200730
        self.flat_layers = flat_layers
        self.flat_options = flat_options
        # end wei

        # wei 20200723
        self.nfr_tag_mode = nfr_tag_mode
        self.d_tag = d_tag
        if self.nfr_tag_mode in ('concat', 'add'):
            TAG_TYPES = 100   # wei 20200803 set it larger for possible expansion of nfr tag types.
            self.nfr_tag_embedding = nn.Embedding(TAG_TYPES, self.d_tag)
        # end wei

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            # wei 20200730
            opt.flat_layers, opt.flat_options,
            opt.nfr_tag_mode, opt.nfr_tag_vec_size
            # end wei
        )

    def forward(self, src, lengths=None, **kwargs):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)

        # wei 20200723
        # calculate for the NFR tag embeddings and incorporate them into tokens' representations
        if self.nfr_tag_mode != 'none':
            nfr_tags = kwargs.get('nfr_tag')
            tag_emb = self.nfr_tag_embedding(nfr_tags)
            if self.nfr_tag_mode == 'concat':
                out = torch.cat((out, tag_emb), dim=2)
            elif self.nfr_tag_mode == 'add':
                out += tag_emb
            else:
                raise Exception('DUMMY')
        # end wei

        # wei 20200731
        # feed forward procedure in flat setting
        if self.flat_layers > 0:
            flat_tags = kwargs.get('flat_tag')
            flat_options = kwargs.get('flat_options')
            assert flat_tags is not None and flat_options is not None, 'If flat setting is activated, please provide' \
                                                                       'flat tags and flat options.'

            max_full_attn_i = len(self.transformer) - self.flat_layers - 1
            for layer_n, layer in enumerate(self.transformer):
                if layer_n > max_full_attn_i:
                    mask = ~sequence_mask_by_tag(flat_tags, flat_options).unsqueeze(1)
                out = layer(out, mask)
            out = self.layer_norm(out)
            return emb, out.transpose(0, 1).contiguous(), lengths

        # end wei

        # wei 20200723
        # feed forward procedure in flat setting
        # if self.flat_layers > 0:
        #     max_full_attn_i = len(self.transformer) - self.flat_layers - 1
        #     append_len = lengths.max() - real_source_lengths.max()
        #     append_mask = torch.ones(mask.size(0), mask.size(1), append_len).type(torch.bool).to(mask.device)
        #
        #     # Run the forward pass of every layer of the tranformer.
        #     for layer_n, layer in enumerate(self.transformer):
        #         if layer_n > max_full_attn_i:
        #             mask = ~sequence_mask(real_source_lengths).unsqueeze(1)
        #             mask = torch.cat((mask, append_mask), dim=-1)
        #         out = layer(out, mask)
        #     out = self.layer_norm(out)
        #     return emb, out.transpose(0, 1).contiguous(), real_source_lengths
        # end wei

        else:
            # Run the forward pass of every layer of the tranformer.
            for layer in self.transformer:
                out = layer(out, mask)
            out = self.layer_norm(out)
            return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)


# class TransformerFlatEncoder(EncoderBase):
#
#     def __init__(self, num_layers, d_model, heads, d_ff, dropout,
#                  attention_dropout, embeddings, max_relative_positions,
#                  flat_layers):
#         super(TransformerFlatEncoder, self).__init__()
#
#         self.embeddings = embeddings
#         self.transformer = nn.ModuleList(
#             [TransformerEncoderLayer(
#                 d_model, heads, d_ff, dropout, attention_dropout,
#                 max_relative_positions=max_relative_positions)
#              for i in range(num_layers)])
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#
#         self.flat_layers = flat_layers
#         assert self.flat_layers > 0, 'number of flat layers must greater than 0'
#         assert self.flat_layers < num_layers, 'number of flat layers must less than total layers'
#
#     @classmethod
#     def from_opt(cls, opt, embeddings):
#         """Alternate constructor."""
#         return cls(
#             opt.enc_layers,
#             opt.enc_rnn_size,
#             opt.heads,
#             opt.transformer_ff,
#             opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
#             opt.attention_dropout[0] if type(opt.attention_dropout)
#             is list else opt.attention_dropout,
#             embeddings,
#             opt.max_relative_positions,
#             opt.flat_layers)
#
#     def forward(self, src, lengths=None):
#         """See :func:`EncoderBase.forward()`"""
#         self._check_args(src, lengths)
#
#         emb, real_source_lengths = self.embeddings(src, output_real_source_lengths=True)
#
#         out = emb.transpose(0, 1).contiguous()
#         mask = ~sequence_mask(lengths).unsqueeze(1)
#
#         max_full_attn_i = len(self.transformer) - self.flat_layers - 1
#         append_len = lengths.max() - real_source_lengths.max()
#         append_mask = torch.ones(mask.size(0), mask.size(1), append_len).type(torch.bool).to(mask.device)
#
#         # Run the forward pass of every layer of the tranformer.
#         for layer_n, layer in enumerate(self.transformer):
#             if layer_n > max_full_attn_i:
#                 mask = ~sequence_mask(real_source_lengths).unsqueeze(1)
#                 mask = torch.cat((mask, append_mask), dim=-1)
#             out = layer(out, mask)
#         out = self.layer_norm(out)
#
#         return emb, out.transpose(0, 1).contiguous(), real_source_lengths
#
#     def update_dropout(self, dropout, attention_dropout):
#         self.embeddings.update_dropout(dropout)
#         for layer in self.transformer:
#             layer.update_dropout(dropout, attention_dropout)