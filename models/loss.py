from operator import index
import torch
import torch.nn.functional as F
import pdb


def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def reconstruction_loss(words_logit, words_id, words_mask, txt_concepts, pos_words_logit, ref_words_logit, num_props, **kwargs):
    
    final_loss = 0
    rec_loss, txt_acc = cal_nll_loss(words_logit, words_id, words_mask)
    bsz, num_concepts, _ = txt_concepts.shape

    rec_loss = rec_loss.mean() * kwargs["alpha_2"]
    final_loss += rec_loss


    if pos_words_logit != None:
        words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
        words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
        nll_loss, vid_acc = cal_nll_loss(pos_words_logit, words_id1, words_mask1)
        min_nll_loss = nll_loss.view(bsz, num_props).min(dim=-1)[0].mean() * kwargs["alpha_1"]
        final_loss += min_nll_loss
        
    
    loss_dict = {
        'Reconstruction Loss:': final_loss.item(),
        '(1) txt_rec_loss:': rec_loss.item(),
        '(2) vid_rec_loss:': min_nll_loss.item() if pos_words_logit != None else 0,
    }

    return final_loss, loss_dict, txt_acc, vid_acc


def div_loss(pos_vid_concepts, txt_concepts, gauss_weight, num_concepts=10, num_props=5, **kwargs):
    loss = 0
    bsz = txt_concepts.shape[0]

    if pos_vid_concepts != None:
        target = torch.eye(num_concepts).unsqueeze(0).cuda()
        source = torch.matmul(F.normalize(pos_vid_concepts,dim=-1, p=2), F.normalize(pos_vid_concepts,dim=-1, p=2).transpose(1, 2))
        vid_div_loss = (torch.norm(target - source, dim=(1, 2))**2).mean()
        loss += vid_div_loss


    if txt_concepts != None:
        target = torch.eye(num_concepts).unsqueeze(0).cuda()
        source = torch.matmul(F.normalize(txt_concepts,dim=-1, p=2), F.normalize(txt_concepts,dim=-1, p=2).transpose(1, 2))
        # div_loss = kwargs["alpha_2"]*(torch.norm(target - source, dim=(1, 2))**2).mean()
        txt_div_loss = (torch.norm(target - source, dim=(1, 2))**2).mean()
        loss += txt_div_loss

        loss_dict = {
            'Diversity Loss:': loss.item(),
            '(1) vid_div_loss': vid_div_loss.item(),
            '(2) txt_div_loss': txt_div_loss.item()
        }

    return loss, loss_dict


def multi_concept_loss(pos_vid_concepts, txt_concepts, neg_vid_concepts_1, neg_vid_concepts_2, pos_words_logit, ref_concept, words_id, words_mask, use_ref_words_sam=True, num_concepts=4, num_props=5, **kwargs):

    pos_vid_concepts = pos_vid_concepts.detach()
    neg_vid_concepts_1 = neg_vid_concepts_1.detach()
    neg_vid_concepts_2 = neg_vid_concepts_2.detach()
    ref_concept = ref_concept.detach()
    
    loss = 0
    _, num_concepts, D = pos_vid_concepts.shape

    bsz = pos_words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    nll_loss, acc = cal_nll_loss(pos_words_logit, words_id1, words_mask1)
    _, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    idx = idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_concepts, D)
    pos_vid_concept = pos_vid_concepts.view(bsz, num_props, num_concepts, -1).gather(index=idx, dim=1).squeeze()
    neg_vid_concept_1 = torch.gather(neg_vid_concepts_1.view(bsz, num_props, num_concepts, -1), index=idx, dim=1).squeeze()
    neg_vid_concept_2 = torch.gather(neg_vid_concepts_2.view(bsz, num_props, num_concepts, -1), index=idx, dim=1).squeeze()

    pos_samilarity = torch.matmul(F.normalize(pos_vid_concept,dim=-1, p=2), F.normalize(txt_concepts,dim=-1, p=2).transpose(1, 2))
    pos_samilarity = torch.diagonal(pos_samilarity, dim1=-1, dim2=-2)

    pos_mse_loss = F.mse_loss(pos_vid_concept, txt_concepts, reduction='none').mean()
    loss += pos_mse_loss

    if neg_vid_concepts_1 != None:
        neg_samilarity_1 = torch.matmul(F.normalize(neg_vid_concept_1,dim=-1, p=2), F.normalize(txt_concepts,dim=-1, p=2).transpose(1, 2))
        neg_samilarity_1 = torch.diagonal(neg_samilarity_1, dim1=-1, dim2=-2)

        tmp_0 = torch.zeros_like(pos_samilarity).cuda()
        tmp_0.requires_grad = False
        samilarity_loss_1 = torch.max(neg_samilarity_1 - pos_samilarity + kwargs["margin_4"], tmp_0).sum(dim=-1).mean()
        loss = loss + samilarity_loss_1

    if neg_vid_concepts_2 != None:
        neg_samilarity_2 = torch.matmul(F.normalize(neg_vid_concept_2,dim=-1, p=2), F.normalize(txt_concepts,dim=-1, p=2).transpose(1, 2))
        neg_samilarity_2 = torch.diagonal(neg_samilarity_2, dim1=-1, dim2=-2)
        tmp_0 = torch.zeros_like(pos_samilarity).cuda()
        tmp_0.requires_grad = False
        samilarity_loss_2 = torch.max(neg_samilarity_2 - pos_samilarity + kwargs["margin_4"], tmp_0).sum(dim=-1).mean()
        loss = loss + samilarity_loss_2

    if ref_concept != None and use_ref_words_sam:
        ref_samilarity = torch.matmul(F.normalize(ref_concept,dim=-1, p=2), F.normalize(txt_concepts,dim=-1, p=2).transpose(1, 2))
        ref_samilarity = torch.diagonal(ref_samilarity, dim1=-1, dim2=-2)
        tmp_0 = torch.zeros_like(pos_samilarity).cuda()
        tmp_0.requires_grad = False
        ref_samilarity_loss = torch.max(ref_samilarity - pos_samilarity + kwargs["margin_3"], tmp_0).sum(dim=-1).mean()
        loss = loss + ref_samilarity_loss

    loss_dict = {
        'Multimodal Concept Loss:': loss.item(),
        '(1) pos_mse_loss': pos_mse_loss.item(),
        '(2) samilarity_loss_1': samilarity_loss_1.item() if neg_vid_concepts_1 != None else 0, 
        '(3) samilarity_loss_2': samilarity_loss_2.item() if neg_vid_concepts_2 != None else 0, 
        '(4) ref_samilarity_loss': ref_samilarity_loss.item() if ref_concept != None and use_ref_words_sam else 0, 
    }
    

    return loss, loss_dict



def ivc_loss(pos_words_logit, words_id, words_mask, num_props, neg_words_logit_1=None, neg_words_logit_2=None, ref_words_logit=None, use_ref_words_rec=True, **kwargs):
    bsz = pos_words_logit.shape[0]//num_props

    words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(pos_words_logit, words_id1, words_mask1)
    min_nll_loss, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    rank_loss = 0


    if ref_words_logit is not None and use_ref_words_rec:
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        ref_loss = torch.max(min_nll_loss - ref_nll_loss + kwargs["margin_1"], tmp_0)
        rank_loss = rank_loss + ref_loss.mean()
    
    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id1, words_mask1)
        neg_nll_loss_1 = torch.gather(neg_nll_loss_1.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_1 = torch.max(min_nll_loss - neg_nll_loss_1 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_1.mean()
    
    if neg_words_logit_2 is not None:
        neg_nll_loss_2, neg_acc_2 = cal_nll_loss(neg_words_logit_2, words_id1, words_mask1)
        neg_nll_loss_2 = torch.gather(neg_nll_loss_2.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_2 = torch.max(min_nll_loss - neg_nll_loss_2 + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_2.mean()

    

    loss = kwargs['alpha_1'] * rank_loss

    return loss, {
        'Intra-Video Loss': loss.item(),
        '(1) hinge_loss_neg1': neg_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
        '(2) hinge_loss_neg2': neg_loss_2.mean().item() if neg_words_logit_2 is not None else 0.0,
        '(3) hinge_loss_ref': ref_loss.mean().item() if ref_words_logit is not None and use_ref_words_rec else 0.0,
    }
