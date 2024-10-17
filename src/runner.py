import sys
import os
sys.path.append(os.getcwd())
import math
import time
import json
import argparse

import torch
from torch.utils.data import DataLoader

from src.model import get_model
from src.model.utils.optimization import BertAdam, EMA
from src.dataloader.mecd_dataset import MECDDataset, sentences_collate
from src.engine.train import train_epoch
from src.engine.valid import eval_epoch
from src.utils import save_parsed_args_to_json, is_distributed, init_seed, create_save_dir, dist_log


class runner():

    @staticmethod
    def train_logic(epoch_i, model, training_data, optimizer, ema, device, opt):
        if is_distributed():
            training_data.sampler.set_epoch(epoch_i)
            dist_log('Setting sampler seed: {}'.format(training_data.sampler.epoch))

        start = time.time()
        if ema is not None and epoch_i != 0:  # * use normal parameters for training, not EMA model
            ema.resume(model)
        train_loss, train_acc = train_epoch(
            model, training_data, optimizer, ema, device, opt, epoch_i)

        dist_log('[Training]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min'
                 .format(ppl=math.exp(min(train_loss, 100)), acc=100 * train_acc, elapse=(time.time() - start) / 60.))

        # * Note here GT words are used to predicted next words, the same as training case!
        if ema is not None:
            ema.assign(model)  # * EMA model

        return model, ema, train_loss, train_acc

    @staticmethod
    def eval_logic(model, validation_data, device, opt):
        start = time.time()

        val_loss, val_acc, relation_acc, accuracy_relation_0, accuracy_relation_1, r0_ratio, precision, complete_acc \
            = eval_epoch(model, validation_data, device, opt)
        F1_score = 2 * precision * accuracy_relation_1 / (precision + accuracy_relation_1)
        dist_log(
            '[Val]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min, Relation_acc{relationacc:3.3f},'
            'relation0_acc{acc_relation_0:3.3f}, relation1_acc(Recall){acc_relation_1:3.3f}, Percision{percision:3.3f},'
            'F1_score{F1_score:3.3f}, 0_ratio{ratio:3.3f}, complete_acc{complete_acc:3.3f}'
            .format(ppl=math.exp(min(val_loss, 100)), acc=100 * val_acc, elapse=(time.time() - start) / 60.,
                    relationacc=100 * relation_acc, acc_relation_0=100 * accuracy_relation_0,
                    acc_relation_1=100 * accuracy_relation_1, percision=precision * 100, F1_score=F1_score,
                    ratio=100 * r0_ratio, complete_acc=100 * complete_acc))

        return model, val_loss, val_acc

    @staticmethod
    def save_checkpoint(epoch_i, model, opt):

        if hasattr(model, 'module'):
            _model = model.module
        else:
            _model = model

        checkpoint = {
            'model': _model.state_dict(),  # * EMA model
            'model_cfg': _model.config,
            'epoch': epoch_i}

        if opt.save_mode == 'all':
            model_name = opt.save_model + '_e{}.chkpt'.format(epoch_i)
            torch.save(checkpoint, model_name)

        return model

    @staticmethod
    def run(model, training_data, validation_data, device, opt):
        def get_pytorch_model_info(model: torch.nn.Module) -> (dict, list):
            """
            输入一个PyTorch Model对象，返回模型的总参数量（格式化为易读格式）以及每一层的名称、尺寸、精度、参数量、是否可训练和层的类别。

            :param model: PyTorch Model
            :return: (总参数量信息, 参数列表[包括每层的名称、尺寸、数据类型、参数量、是否可训练和层的类别])
            """
            params_list = []
            total_params = 0
            total_params_non_trainable = 0

            for name, param in model.named_parameters():
                # 获取参数所属层的名称
                layer_name = name.split('.')[0]
                # 获取层的对象
                layer = dict(model.named_modules())[layer_name]
                # 获取层的类名
                layer_class = layer.__class__.__name__

                params_count = param.numel()
                trainable = param.requires_grad
                params_list.append({
                    'tensor': name,
                    'layer_class': layer_class,
                    'shape': str(list(param.size())),
                    'precision': str(param.dtype).split('.')[-1],
                    'params_count': str(params_count),
                    'trainable': str(trainable),
                })
                total_params += params_count
                if not trainable:
                    total_params_non_trainable += params_count

            total_params_trainable = total_params - total_params_non_trainable

            total_params_info = {
                'total_params': total_params,
                'total_params_trainable': total_params_trainable,
                'total_params_non_trainable': total_params_non_trainable
            }

            return total_params_info, params_list

        total_params_info, params_list = get_pytorch_model_info(model)
        # CLIP 就不用pretrain了, bert 用
        if opt.use_pretrain:
            state_dict = torch.load(opt.weights_path)
            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict['model'].items() if k in model_state_dict}
            model.load_state_dict(filtered_state_dict, strict=False)
            if filtered_state_dict is not None:
                print("load pretrain from:{}".format(opt.weights_path))

        param_optimizer = list(model.named_parameters())

        if opt.use_shared_txt_emb:
            for parameter in model.module.embeddings_dec2.txt_emb.parameters():
                parameter.requires_grad = False

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if (p.requires_grad and (not any(nd in n for nd in no_decay)))],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if (p.requires_grad and any(nd in n for nd in no_decay))],
             'weight_decay': 0.0}
        ]

        if opt.ema_decay != -1:
            ema = EMA(opt.ema_decay)
            for name, p in model.named_parameters():
                if p.requires_grad:
                    ema.register(name, p.data)
        else:
            ema = None

        num_train_optimization_steps = len(training_data) * opt.n_epoch
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=opt.lr,
                             warmup=opt.lr_warmup_proportion,
                             t_total=num_train_optimization_steps,
                             schedule='warmup_linear')

        log_train_file = None
        log_valid_file = None
        if opt.log and opt.local_rank <= 0:
            log_train_file = opt.log + '.train.log'
            log_valid_file = opt.log + '.valid.log'

            dist_log('Training performance will be written to file: {} and {}'.format(
                log_train_file, log_valid_file))

            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss,ppl,accuracy\n')
                log_vf.write('epoch,loss,ppl,accuracy,METEOR,BLEU@4,CIDEr,ROUGE\n')

        prev_best_score = 0.
        for epoch_i in range(opt.n_epoch):

            dist_log('[Epoch {}]'.format(epoch_i))

            model, ema, train_loss, train_acc = runner.train_logic(epoch_i, model, training_data, optimizer, ema, device, opt)

            model, val_loss, val_acc = runner.eval_logic(model, validation_data, device, opt)

            if epoch_i >= opt.trans_sta_epoch:
                model = runner.save_checkpoint(epoch_i, model, opt)
                cfg_name = opt.save_model + '.cfg.json'
                save_parsed_args_to_json(opt, cfg_name)

        return model


def get_args():
    '''parse and preprocess cmd line args'''
    parser = argparse.ArgumentParser()

    # * basic settings
    parser.add_argument('--dset_name', type=str, default='activitynet',
                        help='Name of the dataset, will affect data loader, evaluation, etc')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='your_path/captions',
                        help='dir containing the splits data files')
    parser.add_argument('--res_root_dir', type=str, default='results',
                        help='dir to containing all the results')

    # * training config -- batch/lr/eval etc.
    parser.add_argument('--n_epoch', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='inference batch size')
    parser.add_argument('--trans_batch_size', type=int, default=16, help='tranlating batch size')
    parser.add_argument('--lr', type=float, default=16e-5)
    parser.add_argument('--lr_warmup_proportion', default=0.1, type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10% of training.')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Use soft target instead of one-hot hard target')
    parser.add_argument('--grad_clip', type=float, default=1, help='clip gradient, -1 == disable')
    parser.add_argument('--ema_decay', default=0.9999, type=float,
                        help='Use exponential moving average at training, float in (0, 1) and -1: do not use.  '
                             'ema_param = new_param * ema_decay + (1-ema_decay) * last_param')

    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Don\'t use pin_memory=True for dataloader. '
                             'ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num subprocesses used to load the data, 0: use main process')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='all',
                        help='all: save models at each epoch; best: only save the best model')
    parser.add_argument('--no_cuda', action='store_true', help='run on cpu')

    parser.add_argument('--evaluate_mode', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--score_mode', type=str, default='event', choices=['event'])
    parser.add_argument('--trans_sta_epoch', type=int, default=20)

    # * model overall config
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--intermediate_size', type=int, default=768)
    parser.add_argument('--vocab_size', type=int, help='number of words in the vocabulary')
    parser.add_argument('--word_vec_size', type=int, default=300)
    parser.add_argument('--video_feature_size', type=int, default=2048, help='2048 appearance + 1024 flow')
    parser.add_argument('--max_v_len', type=int, default=50, help='max length of video feature')
    parser.add_argument('--max_t_len', type=int, default=30,
                        help='max length of text (sentence or paragraph)')
    parser.add_argument('--max_cot_len', type=int, default=50,
                        help='cot max length')
    parser.add_argument('--max_n_len', type=int, default=11,
                        help='for recurrent, max number of sentences')

    # * initialization config
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='number of transformer layers')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--initializer_range', type=float, default=0.02)

    # * visual encoder config
    parser.add_argument('--K', type=int, default=4, help='number of cascades')
    parser.add_argument('--loss_aux_weight', type=float, default=0.1, help='')
    parser.add_argument('--momentum_aux_m', type=float, default=0.999)

    # * linguistic encoder config
    parser.add_argument('--glove_path', type=str,
                        default='./captions/min3_word2idx_6B.json',
                        help='extracted GloVe vectors')
    parser.add_argument('--glove_version', type=str, default=None, help='extracted GloVe vectors')
    parser.add_argument('--freeze_glove', action='store_true', help='do not train GloVe vectors')
    parser.add_argument('--share_wd_cls_weight', action='store_true',
                        help='share weight matrix of the word embedding with the final classifier, ')

    # * cascade decoder config
    parser.add_argument('--num_dec1_blocks', type=int, default=4)
    parser.add_argument('--num_dec2_blocks', type=int, default=4)
    parser.add_argument('--loss_aux_caption', type=float, default=0.25)
    parser.add_argument('--loss_main_caption', type=float, default=0.25)
    parser.add_argument('--use_shared_txt_emb', action='store_true')
    # * scheduled sampling2024
    parser.add_argument('--disable_scheduled_sampling', action='store_true')
    parser.add_argument('--scheduled_method', type=str, default='probability', choices=['confidence', 'probability'])
    parser.add_argument('--conf_replace_tr', type=float, default=0.5)
    parser.add_argument('--decay_k', type=int, default=10)
    # * sentence embedding and conf hyper-parameters
    parser.add_argument('--sentence_emb_aggregation_mode', type=str, default='max', choices=['mean', 'max', 'weighted'])
    parser.add_argument('--conf_bucket_size', type=int, default=10)
    parser.add_argument('--conf_temperature', type=float, default=1.0)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--w_CEloss', type=float, default=0.0)
    parser.add_argument('--w_CEloss_words', type=float, default=0.0)
    parser.add_argument('--mix_weight', type=float, default=0.3)
    parser.add_argument('--w_hybrid_loss1', type=float, default=1.0,
                        help='hybrid_cross_entropy_loss')
    parser.add_argument('--w_compensate', type=float, default=0.05,
                        help='causal compensate')
    parser.add_argument('--hybrid_quant', type=float, default=0.05,
                        help='sentence_similarity_constructive_loss_weight')
    parser.add_argument('--weights_path', type=str,
                        default='the_path_to _pretraining_weight',
                        help='pretrain_weights')
    parser.add_argument('--decay_epoch', type=int, default=4, help='ce_loss weight decay epoch')
    parser.add_argument('--use_decay', type=bool, default=False, help='whether use ce_loss weight decay epoch')
    parser.add_argument('--use_pretrain', type=bool, default=True, help='whether use pretrain or not')
    parser.add_argument('--json_path', type=str, default='train.json')
    parser.add_argument('--loss_aux_weight2', type=float, default=1e-6, help='mse_loss_weight')
    parser.add_argument('--use_existence', type=bool, default=True,
                        help='whether use EXISTENCE descriptions')
    parser.add_argument('--logits_mix_weight', type=float, default=0.05, help='causal_logits_weight')
    parser.add_argument('--del_words', type=int, default=0, help='words masked every caption sentences')
    parser.add_argument('--mask_frames', type=int, default=0, help='frames to be masked every event')
    parser.add_argument('--multi_chains_b', type=int, default=1, help='multi chains max(1, (num_sen+b) // k)')
    parser.add_argument('--multi_chains_k', type=int, default=3, help='multi chains max(1, (num_sen+b) // k)')
    parser.add_argument('--validation_set', type=str,
                        default='./captions/val_mecd_complete.json',
                        help='If not None, conducting complete causal graph reasoning.')
    # * post process and compatibility check
    opt = parser.parse_args()
    opt.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    opt.cuda = not opt.no_cuda
    opt.pin_memory = not opt.no_pin_memory

    if opt.share_wd_cls_weight:
        assert opt.word_vec_size == opt.hidden_size, \
            'hidden size has to be the same as word embedding size when ' \
            'sharing the word embedding weight and the final classifier weight'

    if opt.K == 1:
        opt.loss_main_caption = 1.
        opt.loss_aux_caption = 0.
        opt.disable_scheduled_sampling = True

    return opt


def main():
    import sys
    sys.path.append(os.getcwd())
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    opt = get_args()
    init_seed(opt.seed, cuda_deterministic=True)

    train_dataset = MECDDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir,
        max_t_len=opt.max_t_len, max_v_len=opt.max_v_len, max_n_len=opt.max_n_len, max_cot_len=opt.max_cot_len,
        del_words=opt.del_words, mask_frames=opt.mask_frames, json_path=opt.json_path, use_existence=opt.use_existence,
        multi_chains_k=opt.multi_chains_k, multi_chains_b=opt.multi_chains_b,
        mode='train', K=opt.K, word2idx_path=None if opt.glove_path is None else opt.glove_path
    )
    val_dataset = MECDDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir,
        max_t_len=opt.max_t_len, max_v_len=opt.max_v_len, max_n_len=opt.max_n_len, max_cot_len=opt.max_cot_len,
        del_words=opt.del_words, mask_frames=opt.mask_frames, json_path=opt.json_path, use_existence=opt.use_existence,
        validation_set=opt.validation_set,
        mode=opt.evaluate_mode, K=opt.K, word2idx_path=None if opt.glove_path is None else opt.glove_path
    )

    device = torch.device('cuda' if opt.cuda else 'cpu')
    train_sampler, val_sampler = None, None

    opt = create_save_dir(opt)

    train_loader = DataLoader(train_dataset, collate_fn=sentences_collate,
                              batch_size=opt.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=opt.num_workers, pin_memory=opt.pin_memory,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset, collate_fn=sentences_collate,
                            batch_size=opt.val_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=opt.pin_memory,
                            sampler=val_sampler)

    opt.vocab_size = len(train_dataset.word2idx)
    print(json.dumps(vars(opt), indent=4, sort_keys=True))

    model = get_model(opt)
    torch.cuda.set_device(opt.local_rank)

    model = model.to(device)

    runner().run(model, train_loader, val_loader, device, opt)


if __name__ == '__main__':
    main()
