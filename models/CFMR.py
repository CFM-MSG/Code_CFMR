import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, XavierUniform, XavierNormal, Constant

from models.transformer import VideoEncoder, TextEncoder, CrossDecoder

import pdb

class CFMR(nn.Cell):
    def __init__(self, config):
        super().__init__()
        # self.dropout = 1 - config['dropout']
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config["sigma"]
        self.use_negative = config['use_negative']
        self.guass_center_num = config['guass_center_num']
        self.guass_width_num = config['guass_width_num']
        self.width_thresh = config['width_thresh']
        self.guass_width_num_training = config['guass_width_num_training']
        self.width_thresh_training = config['width_thresh_training']
        self.num_props = self.guass_center_num * self.guass_width_num
        self.num_concepts = config['num_concept']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']

        # self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size'])
        # self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        self.frame_fc = nn.Dense(in_channels=config['frames_input_size'], out_channels=config['hidden_size'])
        self.word_fc = nn.Dense(in_channels=config['words_input_size'], out_channels=config['hidden_size'])
        
        # self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        # self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        # self.txt_pred_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        # self.vid_pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        self.mask_vec = Parameter(initializer(Constant(0.), [config['words_input_size']], mindspore.float32), requires_grad=True)
        self.start_vec = Parameter(initializer(Constant(0.), [config['words_input_size']], mindspore.float32), requires_grad=True)
        self.txt_pred_vec = Parameter(initializer(Constant(0.), [config['words_input_size']], mindspore.float32), requires_grad=True)
        self.vid_pred_vec = Parameter(initializer(Constant(0.), [config['frames_input_size']], mindspore.float32), requires_grad=True)

        # self.tmp_0 = mindspore.Parameter(ops.zeros_like(pos_samilarity), requires_grad=False)

        self.vid_encoder = VideoEncoder(**config['Transformers'], concept_nums=config['num_concept'])
        self.txt_encoder = TextEncoder(**config['Transformers'], concept_nums=config['num_concept'])
        self.cross_decoder = CrossDecoder(**config['Transformers'], concept_nums=config['num_concept'])

        # self.fc_comp = nn.Linear(config['hidden_size'], self.vocab_size)
        self.fc_comp = nn.Dense(in_channels=config['hidden_size'], out_channels=self.vocab_size)
 
        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, 20)

    def construct(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, glance, **kwargs):
        bsz, n_frames, _ = frames_feat.shape
        # vid_pred_vec = self.vid_pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        vid_pred_vec = ops.broadcast_to(self.vid_pred_vec.view(1, 1, -1), (bsz, 1, -1))

        # frames_feat = torch.cat([frames_feat, vid_pred_vec], dim=1)
        # frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = P.Concat(1)([frames_feat, vid_pred_vec])
        frames_feat = ops.dropout(frames_feat, p=self.dropout, training=self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len + 1)


        words_feat[:, 0] = self.start_vec
        words_feat[:, -1] = self.txt_pred_vec
        words_pos = self.word_pos_encoder(words_feat)
        # words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = ops.dropout(words_feat, p=self.dropout, training=self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)
        words_mask[:,-1] = 1

        # print(1)
        if self.training:
            # gauss_center = glance.unsqueeze(1).expand(-1, self.guass_width_num_training).reshape(-1)
            # gauss_width = torch.tensor(np.arange(0, self.guass_width_num_training)+1).type_as(frames_feat) * self.width_thresh_training * 1/self.guass_width_num_training
            # gauss_width = gauss_width.unsqueeze(dim=0).expand(bsz, -1).reshape(-1)

            gauss_center = ops.broadcast_to(glance.unsqueeze(1), (-1, self.guass_width_num_training)).reshape(-1)
            gauss_width = ops.cast(ops.arange(0, self.guass_width_num_training)+1, frames_feat.dtype) * self.width_thresh_training * 1/self.guass_width_num_training
            gauss_width = ops.broadcast_to(gauss_width.unsqueeze(0), (bsz, -1)).reshape(-1)
        else:
            # gauss_center = (torch.linspace(0, n_frames, steps=self.guass_center_num)/n_frames).type_as(frames_feat)
            # gauss_center = gauss_center.unsqueeze(dim=0).expand(self.guass_width_num,-1).reshape(-1)
            # gauss_center = gauss_center.unsqueeze(dim=0).expand(bsz, -1).reshape(-1)
            # gauss_width = torch.tensor(np.arange(0, self.guass_width_num)+1).type_as(frames_feat) * self.width_thresh * 1/self.guass_width_num
            # gauss_width = gauss_width.unsqueeze(dim=1).expand(-1, self.guass_center_num).reshape(-1)
            # gauss_width = gauss_width.unsqueeze(dim=0).expand(bsz, -1).reshape(-1)
            gauss_center = ops.cast(ops.linspace(0, n_frames, steps=self.guass_center_num)/n_frames, frames_feat.dtype)
            gauss_center = ops.broadcast_to(gauss_center.unsqueeze(0), (self.guass_width_num,-1)).reshape(-1)
            gauss_center = ops.broadcast_to(gauss_center.unsqueeze(0), (bsz, -1)).reshape(-1)
            gauss_width = ops.cast(ops.arange(0, self.guass_width_num)+1, frames_feat.dtype) * self.width_thresh * 1/self.guass_width_num
            gauss_width = ops.broadcast_to(gauss_width.unsqueeze(1), (-1, self.guass_center_num)).reshape(-1)
            gauss_width = ops.broadcast_to(gauss_width.unsqueeze(0), (bsz, -1)).reshape(-1)
        
        # print(2)
        # downsample for effeciency
        props_len = n_frames//4
        keep_idx = ops.linspace(0, n_frames-1, steps=props_len).long()
        frames_feat = ops.cat((frames_feat[:, keep_idx], frames_feat[:,-2:-1]), axis=1)
        frames_mask = ops.cat((frames_mask[:, keep_idx], frames_mask[:,-2:-1]), axis=1)
        props_len += 1
        # print(3)
        # semantic completion
        if self.training:
            gauss_weight = self.generate_gauss_weight(props_len-1, gauss_center, gauss_width)
            # gauss_weight = torch.cat((gauss_weight, torch.zeros((gauss_weight.shape[0], 1)).type_as(gauss_weight)),  dim=-1)

            # props_feat = frames_feat.unsqueeze(1).expand(bsz, self.guass_width_num, -1, -1).contiguous().view(bsz*self.guass_width_num, props_len, -1)
            # props_mask = frames_mask.unsqueeze(1).expand(bsz, self.guass_width_num, -1).contiguous().view(bsz*self.guass_width_num, -1)

            gauss_weight = ops.cast(ops.cat((gauss_weight, ops.zeros((gauss_weight.shape[0], 1))), axis=-1), gauss_weight.dtype)
            # print(frames_feat.shape, self.guass_width_num, props_len, bsz)
            # print(frames_feat.unsqueeze(1).broadcast_to((bsz, self.guass_width_num, -1, -1)).is_contiguous())
            # props_feat = ops.broadcast_to(frames_feat.unsqueeze(1), (bsz, self.guass_width_num, -1, -1)).contiguous().view(bsz*self.guass_width_num, props_len, -1)
            props_feat = ops.broadcast_to(frames_feat.unsqueeze(1), (bsz, self.guass_width_num, -1, -1)).reshape(bsz*self.guass_width_num, props_len, -1)
            # props_mask = ops.broadcast_to(frames_mask.unsqueeze(1), (bsz, self.guass_width_num, -1)).contiguous().view(bsz*self.guass_width_num, -1)
            props_mask = ops.broadcast_to(frames_mask.unsqueeze(1), (bsz, self.guass_width_num, -1)).reshape(bsz*self.guass_width_num, -1)

        else:
            # props_feat = frames_feat.unsqueeze(1).expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, props_len, -1)
            # props_mask = frames_mask.unsqueeze(1).expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
            # gauss_weight = self.generate_gauss_weight(props_len-1, gauss_center, gauss_width)
            # gauss_weight = torch.cat((gauss_weight, torch.zeros((gauss_weight.shape[0], 1)).type_as(gauss_weight)),  dim=-1)

            gauss_weight = self.generate_gauss_weight(props_len-1, gauss_center, gauss_width)
            gauss_weight = ops.cast(ops.cat((gauss_weight, ops.zeros((gauss_weight.shape[0], 1))), axis=-1), gauss_weight.dtype)
            # props_feat = ops.broadcast_to(frames_feat.unsqueeze(1), (bsz, self.num_props, -1, -1)).contiguous().view(bsz*self.num_props, props_len, -1)
            props_feat = ops.broadcast_to(frames_feat.unsqueeze(1), (bsz, self.num_props, -1, -1)).reshape(bsz*self.num_props, props_len, -1)
            # props_mask = ops.broadcast_to(frames_mask.unsqueeze(1), (bsz, self.num_props, -1)).contiguous().view(bsz*self.num_props, -1)
            props_mask = ops.broadcast_to(frames_mask.unsqueeze(1), (bsz, self.num_props, -1)).reshape(bsz*self.num_props, -1)

        # print(4)
        masked_words_feat, _ = self._mask_words(words_feat, words_len, weights=weights)
        masked_words_feat = masked_words_feat + words_pos
        masked_words_feat = masked_words_feat[:, :-2]
        masked_words_mask = words_mask[:, :-2]
        pos_weight = gauss_weight/gauss_weight.max(axis=-1, keepdims=True)

        pos_vid_concept = self.vid_encoder(props_feat, props_mask, gauss_weight=pos_weight)

        word_concept = self.txt_encoder(words_feat, words_mask)
        # print(5)
        if self.training:
            h, cross_h = self.cross_decoder(pos_vid_concept, None, word_concept, None, masked_words_feat, masked_words_mask)
            words_logit = self.fc_comp(h)
            cross_words_logit = self.fc_comp(cross_h)
        else:
            cross_words_logit = None
            words_logit = None
        # print(6)
        if self.use_negative and self.training:
            
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(props_len, gauss_center, gauss_width, kwargs['epoch'])
            
            neg_vid_concept_1 = self.vid_encoder(props_feat, props_mask, gauss_weight=neg_1_weight)
            _, neg_cross_h_1 = self.cross_decoder(neg_vid_concept_1, None, tgt = masked_words_feat, tgt_mask = masked_words_mask)
            neg_words_logit_1 = self.fc_comp(neg_cross_h_1)
  
            neg_vid_concept_2 = self.vid_encoder(props_feat, props_mask, gauss_weight=neg_2_weight)
            _, neg_cross_h_2 = self.cross_decoder(neg_vid_concept_2, None, tgt = masked_words_feat, tgt_mask = masked_words_mask)
            neg_words_logit_2 = self.fc_comp(neg_cross_h_2)

            ref_concept = self.vid_encoder(frames_feat, frames_mask)
            _, ref_cross_h = self.cross_decoder(ref_concept, None, tgt = masked_words_feat, tgt_mask = masked_words_mask)
            ref_words_logit = self.fc_comp(ref_cross_h)

        else:
            neg_vid_concept_1 = None
            neg_cross_h_1 = None
            neg_vid_concept_2 = None
            neg_cross_h_2 = None
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_concept = None
            ref_words_logit = None
        # print(7)
        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'pos_words_logit': cross_words_logit,
            'ref_words_logit':  ref_words_logit,

            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': masked_words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,

            'pos_vid_concepts': pos_vid_concept,
            'neg_vid_concepts_1': neg_vid_concept_1,
            'neg_vid_concepts_2': neg_vid_concept_2,
            'ref_concept': ref_concept,
            'txt_concepts': word_concept
        }

    # def forward_txt(self, words_feat, words_len, **kwargs):

    #     words_feat[:, 0] = self.start_vec.cuda()
    #     words_feat[:, -1] = self.txt_pred_vec.cuda()
    #     words_feat = F.dropout(words_feat, self.dropout, self.training)
    #     words_feat = self.word_fc(words_feat)
    #     words_mask = _generate_mask(words_feat, words_len + 1)
    #     words_mask[:,-1] = 1
    #     word_concept = self.txt_encoder(words_feat, words_mask)

    #     return {
    #         'txt_concepts': word_concept
    #     }


    # def forward_vid(self, frames_feat, frames_len, **kwargs):
    #     bsz, n_frames, _ = frames_feat.shape
    #     vid_pred_vec = self.vid_pred_vec.view(1, 1, -1).expand(bsz, 1, -1)

    #     frames_feat = torch.cat([frames_feat, vid_pred_vec], dim=1)
    #     frames_feat = F.dropout(frames_feat, self.dropout, self.training)
    #     frames_feat = self.frame_fc(frames_feat)
    #     frames_mask = _generate_mask(frames_feat, frames_len + 1)

    #     # generate Gaussian masks

    #     gauss_center = (torch.linspace(0, n_frames, steps=self.guass_center_num)/n_frames).type_as(frames_feat)
    #     gauss_width = torch.tensor(np.arange(0, self.guass_width_num)+1).type_as(frames_feat) * self.width_thresh * 1/self.guass_width_num

        # gauss_center = gauss_center.unsqueeze(dim=0).expand(self.guass_width_num,-1).reshape(-1)
        # gauss_center = gauss_center.unsqueeze(dim=0).expand(bsz, -1).reshape(-1)
        # gauss_width = gauss_width.unsqueeze(dim=1).expand(-1, self.guass_center_num).reshape(-1)
        # gauss_width = gauss_width.unsqueeze(dim=0).expand(bsz, -1).reshape(-1)
        
        # # downsample for effeciency
        # props_len = n_frames//4
        # keep_idx = torch.linspace(0, n_frames-1, steps=props_len).long()
        # frames_feat = torch.cat((frames_feat[:, keep_idx], frames_feat[:,-2:-1]), dim=1)
        # frames_mask = torch.cat((frames_mask[:, keep_idx], frames_mask[:,-2:-1]), dim=1)
        # props_len += 1

        # props_feat = frames_feat.unsqueeze(1).expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, props_len, -1)
        # props_mask = frames_mask.unsqueeze(1).expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        # gauss_weight = self.generate_gauss_weight(props_len-1, gauss_center, gauss_width)
        # gauss_weight = torch.cat((gauss_weight, torch.zeros((gauss_weight.shape[0], 1)).type_as(gauss_weight)),  dim=-1)
            
        # pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]

        # pos_vid_concept = self.vid_encoder(props_feat, props_mask, gauss_weight=pos_weight)

        # return {
        #     'width': gauss_width,
        #     'center': gauss_center,
        #     'pos_vid_concepts': pos_vid_concept,
        # }


    def generate_gauss_weight(self, props_len, center, width):
        # weight = ops.linspace(0, 1, props_len)
        # weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        # center = center.unsqueeze(-1)
        # width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        # w = 0.3989422804014327
        # weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        # return weight/weight.max(dim=-1, keepdim=True)[0]
        weight = ops.linspace(0, 1, props_len)
        # print(center.size(0))
        weight = ops.broadcast_to(weight.view(1, -1), (center.shape[0], -1))
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma
        w = 0.3989422804014327
        weight = w/width*ops.exp(-(weight-center)**2/(2*width**2))
        # print(weight.max(axis=-1, keepdims=True))
        return weight/weight.max(axis=-1, keepdims=True)


    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma/2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w/w1*ops.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(axis=-1, keepdims=True)

        weight = ops.linspace(0, 1, props_len)
        weight = ops.broadcast_to(weight.view(1, -1), (center.shape[0], -1))

        left_width = ops.clamp(center-width/2, min=0)
        left_center = left_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        right_width = ops.clamp(1-center-width/2, min=0)
        right_center = 1 - right_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1-right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(ops.zeros([words_feat.shape[1]], dtype=mindspore.uint8))
            if l < 1:
                continue
            p = weights[i, :l].numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            # print(masked_words, choices)
            masked_words[-1][mindspore.tensor(choices)] = 1
        
        masked_words = ops.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(words_feat.shape) + token
        masked_words_vec = masked_words_vec.masked_fill(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.shape[1]:
        mask = None
    else:
        mask = []
        for l in x_len:
            # mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask.append(ops.zeros([x.shape[1]], dtype=mindspore.int32))
            mask[-1][:l] = 1
        mask = ops.stack(mask, axis=0)
    return mask


class SinusoidalPositionalEmbedding(nn.Cell):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = ops.exp(ops.arange(half_dim, dtype=mindspore.float32) * -emb)
        emb = ops.arange(num_embeddings, dtype=mindspore.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = ops.cat([ops.sin(emb), ops.cos(emb)], axis=1).view(num_embeddings, -1)

        if embedding_dim % 2 == 1:
            emb = ops.cat([emb, ops.zeros(num_embeddings, 1)], axis=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def construct(self, input, **kwargs):
        bsz, seq_len, _ = input.shape
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.shape[0]:
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
