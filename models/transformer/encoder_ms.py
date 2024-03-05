import mindspore
import mindspore.nn as nn
from mindspore import ops
import mindspore.ops.operations as P

from models.modules import MultiheadAttention


# TransformerEncoder
class TransformerEncoder(nn.Cell):
    def __init__(self, num_layers, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.encoder_layers = nn.CellList([
            TransformerEncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
 
    def construct(self, x, mask=None):
        non_padding_mask = None if mask is None else 1 - mask
        
        permute_idx = [i for i in range(len(x.shape))]
        permute_idx[0] = 1
        permute_idx[1] = 0

        x = x.transpose(permute_idx)
        for layer in self.encoder_layers:
            x = layer(x, non_padding_mask)
        x = x.transpose(permute_idx)
        return x


class TransformerEncoderLayer(nn.Cell):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        d_model = d_model
        num_heads = num_heads
        self.dropout = dropout
        # self.dropout = 1 - dropout
        self.attn_mask = None

        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)
        # self.self_attn_layer_norm = nn.LayerNorm(normalized_shape=d_model, epsilon=1e-05)
        self.fc1 = nn.Dense(in_channels=d_model, out_channels=d_model << 1)
        self.fc2 = nn.Dense(in_channels=d_model << 1, out_channels=d_model)
        self.final_layer_norm = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)

    def construct(self, x, mask):
        dim = P.Shape()(x)[0]

        attn_mask = None if self.attn_mask is None else self.attn_mask[:dim, :dim]
        res = x
        x, weight = self.self_attn(x, x, x, mask, attn_mask=attn_mask)
        x = ops.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.self_attn_layer_norm(x)

        res = x
        x = P.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = ops.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.final_layer_norm(x)
        return x
