import collections
import json
import logging
import os
import pdb
from turtle import width
import mindspore.ops as ops

import numpy as np
# import torch
# import torch.nn.functional as F
import mindspore as ms

from tqdm import tqdm

from models.loss import  reconstruction_loss, ivc_loss, multi_concept_loss, div_loss
from utils import TimeMeter, AverageMeter


def info(msg):
    print(msg)
    logging.info(msg)


class MainRunner:
    def __init__(self, args):
        self.args = args
        self.args['device_target'] = 'GPU'
        self._init_context()
        
        self._build_dataset()

        self.args['model']['config']['vocab_size'] = self.train_set.vocab_size
        self.args['model']['config']['max_epoch'] = self.args['train']['max_num_epochs']

        self._build_model()

        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0

            self._build_txt_optimizer()
            self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)

    def _init_context(self):
        device_id = int(os.getenv('DEVICE_ID', '0'))
        # ms.set_context(mode=ms.PYNATIVE_MODE, device_target=self.args['device_target'], device_id=device_id, deterministic='ON', save_graphs=True, save_graphs_path="graphs/model.ms")
        # GRAPH_MODE
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=self.args['device_target'], device_id=device_id, deterministic='ON')

    def train(self):
        best_results = None

        for epoch in range(1, self.args['train']['max_num_epochs']+1):
            info('Start Epoch {}'.format(epoch))
            self.model_saved_path = self.args['train']['model_saved_path']
            os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
            save_path = os.path.join(self.model_saved_path, 'model-{}.ckpt'.format(epoch))
            txt_acc, vid_acc = self._train_model(epoch=epoch)
            self._save_model(save_path)
            results = self._eval_model(epoch=epoch)
            if best_results is None or results['R@1,mIoU'].avg > best_results['R@1,mIoU'].avg:
                best_results = results
                os.system('cp %s %s'%(save_path, os.path.join(self.model_saved_path, 'model-best.ckpt')))
                info('Best results have been updated.')
            info('=' * 60)
            info('reconstruction txt_acc {:.4f}'.format(txt_acc))
            info('reconstruction vid_acc {:.4f}'.format(vid_acc))

        msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in best_results.items()])
        info('Best results:')
        info('|'+msg+'|')
    
    def forward_fn(self, epoch, net_input):
        # print("forward fn")
        # output = self.model(epoch=epoch, **net_input)
        output = self.model(epoch=epoch, **net_input)
        ## Semantic Loss
        # print("Calculate Semantic Loss...")
        loss, loss_dict, txt_acc, vid_acc = reconstruction_loss(**output, num_props=self.model.guass_width_num, use_min=True,**self.args['loss'])

        # Concept Loss 
        # print("Calculate Concept Loss...")
        conc_loss, conc_loss_dict = multi_concept_loss(**output, **self.args['loss'], num_props=self.model.guass_width_num, num_concepts=10)
        loss_dict.update(conc_loss_dict)
        loss = loss + conc_loss

        # Diversity Loss
        # print("Calculate Diversity Loss...")
        conc_loss, conc_loss_dict = div_loss(**output, **self.args['loss'], num_props=self.model.guass_width_num, num_concepts=self.model.num_concepts)
        loss_dict.update(conc_loss_dict)
        loss = loss + conc_loss
            
        # Intra-video Loss
        # print("Calculate Intra-video Loss...")
        rnk_loss, rnk_loss_dict = ivc_loss(**output, num_props=self.model.guass_width_num, use_div_loss=True, **self.args['loss'])
        loss_dict.update(rnk_loss_dict)
        loss = loss + rnk_loss
        # print(loss_dict, loss)
        self.loss_dict = loss_dict
        return loss, txt_acc, vid_acc
 
    def _train_model(self, epoch, **kwargs):
        self.model.set_train(True)

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.8f}, '.format(epoch, bid, curr_lr)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            info(msg)

        display_n_batches, bid = 200, 0
        # display_n_batches, bid = 5, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())
        total_txt_acc = 0
        total_vid_acc = 0

        for bid, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training epoch {}".format(epoch)):
            # self.optimizer.zero_grad()
            batch = batch[0]
            # print(batch)
            net_input = batch['net_input']
            bsz = net_input["frames_feat"].shape[0]
            
            (loss, txt_acc, vid_acc), grads = self.grad_fn(epoch, net_input)

            # print(grads)
            loss_dict = self.loss_dict
            grads = ms.ops.clip_by_norm(grads, max_norm=10)
            self.optimizer(grads)

            self.lr_scheduler.step(self.num_updates)
            self.num_updates += 1
            curr_lr = self.optimizer.param_groups[0]['lr'].item()

            time_meter.update() 

            total_txt_acc += txt_acc
            total_vid_acc += vid_acc


            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()
        return total_txt_acc/len(self.train_loader), total_vid_acc/len(self.train_loader)
        # return 0, 0

 
    def _eval_model(self, save=None, epoch=0):
        # self.model.eval()
        self.model.set_train(False)
        
        metrics_logger = collections.defaultdict(lambda: AverageMeter())
        l2_normalize = ops.L2Normalize(axis=-1, epsilon=1e-12)

        for bid, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Eval epoch {}".format(epoch)):
            batch = batch[0]
            durations = np.asarray([i[1].asnumpy() for i in batch['raw']])
            # gt = np.asarray([i[2] for i in batch['raw']])
            gt = np.asarray([ops.stack(i[2]).asnumpy() for i in batch['raw']])
            net_input = batch['net_input']

            bsz = len(durations)
            num_props = self.model.num_props
            k = min(num_props, 5)



            output = self.model(epoch=epoch, **net_input)
                    
            pos_vid_concept = output['pos_vid_concepts']
            txt_concepts = output['txt_concepts']

            num_concept = pos_vid_concept.shape[1]

            proposal = l2_normalize(pos_vid_concept).reshape(bsz*num_props, num_concept, -1)
            txt_conc = ops.broadcast_to(l2_normalize(txt_concepts).unsqueeze(1), (bsz, num_props, num_concept, -1)).reshape(bsz*num_props, num_concept, -1)


            pos_samilarity = ops.diagonal(ops.bmm(proposal, txt_conc.transpose(0, 2, 1)), dim1=-2, dim2=-1).sum(axis=-1)
            pos_samilarity = pos_samilarity.reshape(bsz, num_props)
            idx = pos_samilarity.argsort(axis=-1, descending=True)

            width = output['width'].view(bsz, num_props).gather_elements(index=idx, dim=-1)
            center = output['center'].view(bsz, num_props).gather_elements(index=idx, dim=-1)


            selected_props = ops.stack([ops.clamp(center-width/2, min=0), ops.clamp(center+width/2, max=1)], axis=-1)

            selected_props = selected_props.numpy()
            gt = gt / durations[:, np.newaxis]

            # print(durations, gt)
            res = top_1_metric(selected_props[:, 0], gt)
        
            for key, v in res.items():
                metrics_logger['R@1,'+key].update(v, bsz)
                    
            res = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt)
            for key, v in res.items():
                metrics_logger['R@%d,'%(k)+key].update(v, bsz)

        msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items()])
        info('|'+msg+'|')
        return metrics_logger
        
    def _build_dataset(self):
        import datasets as da
        import pickle
        import mindspore.dataset as ds
        args = self.args['dataset']
        cls = getattr(da, args['dataset'], None)
        with open(args['vocab_path'], 'rb') as fp:
            vocab = pickle.load(fp)
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True, split='train')
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args, split='test')
        self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args, split='val') if args['val_data'] else None
        info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
        batch_size = self.args['train']['batch_size']
        

        self.train_loader = ds.GeneratorDataset(self.train_set, column_names=['samples'], shuffle=True, num_parallel_workers=4)
        self.train_loader = self.train_loader.batch(batch_size,  per_batch_map=self.train_set.collate_data, num_parallel_workers=4)
        self.test_loader = ds.GeneratorDataset(self.test_set, column_names=['samples'], shuffle=True, num_parallel_workers=4)
        self.test_loader = self.test_loader.batch(batch_size,  per_batch_map=self.test_set.collate_data, num_parallel_workers=4)
        if args['val_data']:
            self.val_loader = ds.GeneratorDataset(self.val_set, column_names=['samples'], shuffle=True, num_parallel_workers=4)
            self.val_loader = self.val_loader.batch(batch_size,  per_batch_map=self.val_set.collate_data, num_parallel_workers=4)
        else:
            self.val_loader = None

    def _build_model(self):
        model_config = self.args['model']
        import models
        from models.transformer import TransformerEncoder

        self.model = getattr(models, model_config['name'], None)(model_config['config'])
        print("building model......")

    def _build_optimizer(self):
        from mindspore.experimental import optim
        from optimizers.lr_schedulers_ms import InverseSquareRootSchedule
        
        args = self.args['train']["optimizer"]

        self.optimizer = optim.Adam(self.model.trainable_params(), lr=args['lr'])
        self.lr_scheduler = InverseSquareRootSchedule(self.optimizer, args)

    def _build_txt_optimizer(self):
        from mindspore.experimental import optim
        from optimizers.lr_schedulers_ms import InverseSquareRootSchedule
        
        args = self.args['train']["optimizer"]
        self.txt_optimizer = optim.Adam(self.model.trainable_params(), lr=args['lr'])
        self.txt_lr_scheduler = InverseSquareRootSchedule(self.optimizer, args)

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.trainable_params(),
        }
        ms.save_checkpoint(state_dict['model_parameters'], path)
        info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        ms.load_checkpoint(path, self.model)

        info('load model from {}, num_updates {}.'.format(path, self.num_updates))


def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


# [nb, 2], [nb, 2]
def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result

def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


# def apply_to_sample(f, sample):
#     if len(sample) == 0:
#         return {}

#     def _apply(x):
#         if torch.is_tensor(x):
#             return f(x)
#         elif isinstance(x, dict):
#             return {
#                 key: _apply(value)
#                 for key, value in x.items()
#             }
#         elif isinstance(x, list):
#             return [_apply(x) for x in x]
#         else:
#             return x

#     return _apply(sample)


# def move_to_cuda(sample):
#     def _move_to_cuda(tensor):
#         return tensor.cuda()

#     return apply_to_sample(_move_to_cuda, sample)
