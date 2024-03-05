import mindspore
import mindspore.nn as nn
from mindspore import ops
import mindspore.ops.operations as P


from models.modules import MultiheadAttention
import pdb

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    t = t.float().fill(float(-1e30)).type_as(t)
    return t


class TransformerDecoder(nn.Cell):
    def __init__(self, num_layers, d_model, num_heads, dropout=0.0, future_mask=True, attn='Vanilla', norm='BatchNorm'):
        super().__init__()
        self.future_mask = future_mask
        self.decoder_layers = nn.CellList([
            TransformerDecoderLayer(d_model, num_heads, dropout, attn, norm)
            for _ in range(num_layers)
        ])
        # self.decoder_layers = nn.ModuleList([
        #     TransformerDecoderLayer(d_model, num_heads, dropout, attn, norm)
        #     for _ in range(num_layers)
        # ])

    def buffered_future_mask(self, tensor):
        if not self.future_mask:
            return None
        dim = P.Shape()(tensor)[0]
        if not hasattr(self, '_future_mask') or self._future_mask is None :
        # if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = mindspore.numpy.triu(fill_with_neg_inf(tensor.new_ones(size=(dim, dim))), 1)
        if self._future_mask.shape[0] < dim:
            # self._future_mask = nn.Triu(fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
            self._future_mask = mindspore.numpy.triu(fill_with_neg_inf(self._future_mask.resize(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def construct(self, src, src_mask, tgt, tgt_mask, src_gauss_weight=None, tgt_gauss_weight=None):
        non_pad_src_mask = None if src_mask is None else 1 - src_mask
        non_pad_tgt_mask = None if tgt_mask is None else 1 - tgt_mask
        weight = None
        permute_idx = [i for i in range(len(tgt.shape))]
        permute_idx[0] = 1
        permute_idx[1] = 0
        if src is not None:
            src = src.transpose(permute_idx)

        x = tgt.transpose(permute_idx)
        for layer in self.decoder_layers:
            x, weight = layer(x, non_pad_tgt_mask,
                              src, non_pad_src_mask,
                              self.buffered_future_mask(x), 
                              src_gauss_weight, tgt_gauss_weight)
        return x.transpose(permute_idx), weight


class TransformerDecoderLayer(nn.Cell):
    def __init__(self, d_model, num_heads, dropout=0.0, attn='Vanilla', norm='LayerNorm'):
        super().__init__()
        d_model = d_model
        num_heads = num_heads
        self.dropout = dropout
        # mindspore里是相反的含义
        # self.dropout = 1 - dropout

        if attn == 'Vanilla':
            # self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
            # self.encoder_attn = MultiheadAttention(d_model, num_heads, dropout)
            self.self_attn = MultiheadAttention(d_model, num_heads)
            self.encoder_attn = MultiheadAttention(d_model, num_heads)
        else:
            self.self_attn = CosformerAttention(d_model, num_heads)
            self.encoder_attn = CosformerAttention(d_model, num_heads)
        self.fc1 = nn.Dense(in_channels=d_model, out_channels=d_model << 1)
        self.fc2 = nn.Dense(in_channels=d_model << 1, out_channels=d_model)
        self.norm = norm
        assert norm in ['LayerNorm', 'BatchNorm']
        if norm == 'LayerNorm':
            # print('LayerNorm')
            self.self_attn_layer_norm = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)
            self.encoder_attn_layer_norm = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)
            self.final_layer_norm = nn.LayerNorm(normalized_shape=[d_model], epsilon=1e-05)
        else:
            self.self_attn_layer_norm = nn.BatchNorm1d(num_features=d_model, momentum=0.9)
            self.encoder_attn_layer_norm = nn.BatchNorm1d(num_features=d_model, momentum=0.9)
            self.final_layer_norm = nn.BatchNorm1d(num_features=d_model, momentum=0.9)

    def construct(self, x, mask, encoder_out=None, encoder_mask=None, self_attn_mask=None, 
                src_gauss_weight=None, tgt_gauss_weight=None):
        res = x
        x, weight = self.self_attn(x, x, x, key_padding_mask = mask, attn_mask=self_attn_mask, gauss_weight=tgt_gauss_weight)
        x = ops.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        if self.norm == 'BatchNorm':
            x = self.self_attn_layer_norm(P.Transpose()(x, (1,2,0,))).permute(2,0,1)
        else:
            x = self.self_attn_layer_norm(x)

        if encoder_out is not None:
            res = x
            x, weight = self.encoder_attn(x, encoder_out, encoder_out, encoder_mask, gauss_weight=src_gauss_weight)
            x = ops.dropout(x, p=self.dropout, training=self.training)
            x = res + x
            if self.norm == 'BatchNorm':
                x = self.encoder_attn_layer_norm(P.Transpose()(x, (1,2,0,))).permute(2,0,1)
            else:
                x = self.encoder_attn_layer_norm(x)

        res = x
        x = P.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = ops.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        if self.norm == 'BatchNorm':
            x = self.final_layer_norm(P.Transpose()(x, (1,2,0,))).permute(2,0,1)
        else:
            x = self.final_layer_norm(x)
        return x, weight

