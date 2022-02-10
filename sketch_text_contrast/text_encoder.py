"""
Implementation adapted from:
https://github.com/openai/glide-text2im/blob/main/glide_text2im/text2im_model.py
"""

import torch as th
import torch.nn as nn
# from zmq import device

from .fp16_util import convert_module_to_f16
from .bpe import get_encoder
from .xf import LayerNorm, Transformer, convert_module_to_f16

class TextEncoder(nn.Module):
    """
    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param tokenizer: the text tokenizer for sampling/vocab size.
    """

    def __init__(self, model_channels, text_ctx, xf_width, xf_layers, xf_heads, xf_final_ln, xf_padding, device):
        super().__init__()

        self.model_channels = model_channels
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_padding = xf_padding
        self.dtype = th.float32                 # not sure where is is originally defined

        self.tokenizer = get_encoder()
        self.transformer = Transformer(
            text_ctx,
            xf_width,
            xf_layers,
            xf_heads,
        )

        for params in self.transformer.parameters():
            params.requires_grad = False

        if xf_final_ln:
            self.final_ln = LayerNorm(xf_width)
        else:
            self.final_ln = None
        
        self.token_embedding = nn.Embedding(self.tokenizer.n_vocab, xf_width, device=device)
        self.positional_embedding = nn.Parameter(th.empty(text_ctx, xf_width, dtype=th.float32, device=device))
        self.transformer_proj = nn.Linear(xf_width, self.model_channels * 4)

        for params in self.token_embedding.parameters():
            params.requires_grad = False

        for params in self.transformer_proj.parameters():
            params.requires_grad = False

        if self.xf_padding:
            self.padding_embedding = nn.Parameter(
                th.empty(text_ctx, xf_width, dtype=th.float32)
            )

    def convert_to_fp16(self):
        # super().convert_to_fp16()
        if self.xf_width:
            self.dtype = th.float16
            self.transformer.apply(convert_module_to_f16)
            self.transformer_proj.to(th.float16)
            self.token_embedding.to(th.float16)
            self.positional_embedding.to(th.float16)
            if self.xf_padding:
                self.padding_embedding.to(th.float16)
            if self.xf_ar:
                self.unemb.to(th.float16)

    def forward(self, tokens, mask):
        # assert tokens is not None

        # if self.cache_text_emb and self.cache is not None:
        #     assert (
        #         tokens == self.cache["tokens"]
        #     ).all(), f"Tokens {tokens.cpu().numpy().tolist()} do not match cache {self.cache['tokens'].cpu().numpy().tolist()}"
        #     return self.cache

        xf_in = self.token_embedding(tokens.long())
        xf_in = xf_in + self.positional_embedding[None]
        if self.xf_padding:
            assert mask is not None
            xf_in = th.where(mask[..., None], xf_in, self.padding_embedding[None])
        xf_out = self.transformer(xf_in.to(self.dtype))
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        # if self.cache_text_emb:
        #     self.cache = dict(
        #         tokens=tokens,
        #         xf_proj=xf_proj.detach(),
        #         xf_out=xf_out.detach() if xf_out is not None else None,
        #     )

        return outputs