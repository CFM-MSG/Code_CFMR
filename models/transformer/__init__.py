import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore
from mindspore import ops
import pdb

# from models.transformer.decoder import TransformerDecoder
from models.transformer.encoder_ms import TransformerEncoder
from models.transformer.decoder_ms import TransformerDecoder


class Transformer(nn.Cell):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.0):
        super().__init__()
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, dropout)

    def construct(self, src, src_mask, tgt, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(enc_out, src_mask, tgt, tgt_mask)
        return out

class VideoEncoder(nn.Cell):
    def __init__(self, d_model, num_heads, num_decoder_layers1, num_decoder_layers2, concept_nums = 3, dropout=0.0):
        super().__init__()
        self.vid_encoder = TransformerDecoder(num_decoder_layers1, d_model, num_heads, dropout, attn='Vanilla', norm='LayerNorm')
        self.concept_nums = concept_nums

        self.vid_necks = nn.SequentialCell([
            nn.Dense(in_channels=d_model, out_channels=d_model, has_bias=True),
            nn.Dense(in_channels=d_model, out_channels=self.concept_nums * d_model, has_bias=True)
        ])

    def construct(self, vid = None, vid_mask  = None, gauss_weight=None):

        if vid != None:
            vid_enc_out, _ = self.vid_encoder(None, None, vid, vid_mask, tgt_gauss_weight = gauss_weight)
            pred = vid_enc_out[:,-1]
            vid_concept = self.vid_necks(pred)
            # vid_concepts = torch.stack(torch.split(vid_concept, self.concept_nums, dim=-1), dim=1).permute(0, 2, 1)
            vid_concepts = ops.stack(vid_concept.split(self.concept_nums, axis=-1), axis=1).permute(0, 2, 1)
        else:
            vid_concepts = None

        return vid_concepts



class TextEncoder(nn.Cell):
    def __init__(self, d_model, num_heads, num_decoder_layers1, num_decoder_layers2, concept_nums = 3, dropout=0.0):
        super().__init__()
        self.txt_encoder = TransformerDecoder(num_decoder_layers1, d_model, num_heads, dropout, attn='Vanilla', norm='LayerNorm')
        self.concept_nums = concept_nums

        self.txt_necks = nn.SequentialCell([
            nn.Dense(in_channels=d_model, out_channels=d_model, has_bias=True),
            nn.Dense(in_channels=d_model, out_channels=self.concept_nums * d_model, has_bias=True)
        ])

    def construct(self, txt = None, txt_mask = None):

        if txt != None and txt_mask != None:
            txt_enc_out, _ = self.txt_encoder(None, None, txt, txt_mask)
            pred = txt_enc_out[:,-1]
            txt_concept = self.txt_necks(pred)
            # txt_concepts = torch.stack(torch.split(txt_concept, self.concept_nums, dim=-1), dim=1).permute(0, 2, 1)
            txt_concepts = ops.stack(txt_concept.split(self.concept_nums, axis=-1), axis=1).permute(0, 2, 1)
        else:
            txt_concepts = None

        return txt_concepts


class CrossDecoder(nn.Cell):
    def __init__(self, d_model, num_heads, num_decoder_layers1, num_decoder_layers2, concept_nums = 3, dropout=0.0):
        super().__init__()

        self.decoder = TransformerDecoder(num_decoder_layers2, d_model, num_heads, dropout, attn='Vanilla', norm='LayerNorm')

    def construct(self, vid_concepts = None, vid_concept_mask  = None, txt_concepts = None, txt_concepts_mask = None, tgt = None, tgt_mask = None):
        assert tgt != None 
        if txt_concepts != None:
            out, _ = self.decoder(txt_concepts, txt_concepts_mask, tgt, tgt_mask)
        else:
            out = None

        if vid_concepts != None:
            bsz, _, D = tgt.shape
            num_props = vid_concepts.shape[0]//bsz
            # tgt = tgt.unsqueeze(1).expand(bsz, num_props, -1, D).contiguous().view(bsz*num_props, -1, D)
            # tgt_mask = tgt_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

            tgt = ops.broadcast_to(tgt.unsqueeze(1), (bsz, num_props, -1, D)).reshape(bsz*num_props, -1, D)
            tgt_mask = ops.broadcast_to(tgt_mask.unsqueeze(1), (bsz, num_props, -1)).reshape(bsz*num_props, -1)

            cross_out, _ = self.decoder(vid_concepts, vid_concept_mask, tgt, tgt_mask)
        else:
            cross_out = None

        return out, cross_out