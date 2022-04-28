import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 batch_size,
                 d_model,
                 nhead):
        super(LoFTREncoderLayer, self).__init__()

        self.bs = batch_size
        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention(self.nhead, self.dim)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model, eps=1e-7)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-7)

    def forward(self, x, source):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
        """
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(self.bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(self.bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(self.bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)   # [N, L, (H, D)]
        message = self.merge(message.view(self.bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, batch_size, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(batch_size, config['d_model'], config['nhead'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # masks for training
    # def forward(self, feat0, feat1, mask0=None, mask1=None):
    def forward(self, feat0, feat1):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        # assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for i, layer in enumerate(self.layers):
            name = self.layer_names[i]
            if name == 'self':
                feat0 = layer(feat0, feat0)
                feat1 = layer(feat1, feat1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1)
                feat1 = layer(feat1, feat0)
            else:
                raise KeyError

        return feat0, feat1
