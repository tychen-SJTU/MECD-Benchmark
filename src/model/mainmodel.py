import sys
import json
import copy
import math
import token

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.loss import SoftTargetCrossEntropy
# from easydict import EasyDict as edict

import torch.utils.checkpoint as cp

from .utils.optimization import LabelSmoothingLoss
from .modules import BertLayerNorm, BertLMPredictionHead, RelationHead, Compensate_Head
from .visual_model import VideoEncodingTrans
from .loss import InfoCE_Loss
from .linguistic_model import TextEmbedding, BertEmbeddingsWithVideo, BertEmbeddingsWithSentence, BertEncoder, \
    BertEncoderSen
from ..utils import dist_log

import logging

logger = logging.getLogger(__name__)


def _concat_list_of_tensors(_list):
    outs = []
    for param in _list:
        out = torch.stack(param, dim=0).contiguous().view(-1, *param[0].shape[1:])
        outs.append(out)
    return outs


class Reasoner(nn.Module):
    def __init__(self, config):
        super(Reasoner, self).__init__()
        self.config = config

        self.video_encoder = VideoEncodingTrans(config, add_postion_embeddings=True)
        self.projection_head_q = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(),
                                               nn.Linear(config.hidden_size, config.hidden_size))

        self.video_encoder_aux = VideoEncodingTrans(config, add_postion_embeddings=True)
        self.projection_head_k = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(),
                                               nn.Linear(config.hidden_size, config.hidden_size))
        self.m = self.config.momentum_aux_m

        for q, k in zip((self.video_encoder.parameters(), self.projection_head_q.parameters()),
                        (self.video_encoder_aux.parameters(), self.projection_head_k.parameters())):
            for param_q, param_k in zip(q, k):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        self.relation_head = RelationHead(config)
        self.compensate_head = Compensate_Head(config)
        self.CE_loss = torch.nn.CrossEntropyLoss()
        if self.config.K > 1:
            shared_embeddings_dec = BertEmbeddingsWithSentence(config, add_postion_embeddings=True)
            shared_decoder = BertEncoderSen(config)
            shared_pred_head = BertLMPredictionHead(config, None)

            self.embeddings_decs = nn.ModuleList([
                BertEmbeddingsWithVideo(config, add_postion_embeddings=True) if k == 0 else \
                    shared_embeddings_dec \
                for k in range(self.config.K)
            ])
            self.decoders = nn.ModuleList(
                [BertEncoder(config) if k == 0 else shared_decoder for k in range(self.config.K)])
            self.pred_heads = nn.ModuleList(
                [BertLMPredictionHead(config, None) if k == 0 else shared_pred_head for k in range(self.config.K)])
        else:
            # * in list form just for compatibility
            self.embeddings_decs = nn.ModuleList([BertEmbeddingsWithVideo(config, add_postion_embeddings=True)])
            self.decoders = nn.ModuleList([BertEncoder(config)])
            self.pred_heads = nn.ModuleList([BertLMPredictionHead(config, None)])

        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)
        self.apply(self.init_bert_weights)

        # * scheduled sampling
        self.probs = 0.

        # * confidence embedding
        self.conf_bucket_size = config.conf_bucket_size

        # * sentence embedding
        if self.config.sentence_emb_aggregation_mode == 'weighted':
            self._weighted_para = nn.Sequential(
                nn.Linear(config.hidden_size, 4 * config.hidden_size),
                nn.GELU(),
                nn.Linear(4 * config.hidden_size, 1),
            )

    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # * scheduled sampling
    def _update_scheduled_sampling(self, _pred, input_ids):
        _pred = torch.cat(
            [(torch.zeros((int(_pred.shape[0]), 1, int(_pred.shape[2])))).to(_pred.device).float(), _pred], dim=1)[:,
                :-1]
        # * pred_size S*B,L,vocab_size -> S*B,L
        pred = _pred.max(2)[1]
        # * for each word, p prob to be replaced by predicted results
        # * S*B,L
        prob = torch.softmax(_pred, dim=2).max(2)[0]
        replace_mask = (prob > self.config.conf_replace_tr)
        # * DO NOT replace video ids + [BOS]
        replace_mask[:, :(self.config.max_v_len + 1)] = False

        input_ids[replace_mask] = pred[replace_mask]

        return input_ids

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.decay_prob(epoch, k=self.config.decay_k)

    def decay_prob(self, i, or_type=3, k=3000):
        if or_type == 1:  # Linear decay
            or_prob_begin, or_prob_end = 1., 0.
            or_decay_rate = (or_prob_begin - or_prob_end) / 10.
            ss_decay_rate = 0.1
            prob = or_prob_begin - (ss_decay_rate * i)
            if prob < or_prob_end:
                prob_i = or_prob_end
                dist_log('[Linear] schedule sampling probability do not change {}'.format(prob_i))
            else:
                prob_i = prob
                dist_log('[Linear] decay schedule sampling probability to {}'.format(prob_i))

        elif or_type == 2:  # Exponential decay
            prob_i = np.power(k, i)
            dist_log('[Exponential] decay schedule sampling probability to {}'.format(prob_i))

        elif or_type == 3:  # Inverse sigmoid decay
            prob_i = k / (k + np.exp((i / k)))
            dist_log('[Inverse] decay schedule sampling probability to {}'.format(prob_i))
        self.probs = prob_i

        return prob_i

    def get_word_orcale_tokens(self, _pred, prev_output_tokens, epsilon=1e-6):
        _gumbel_noise = 0.5

        B, L = prev_output_tokens.size()
        pred_logits = _pred[:, self.config.max_v_len:]
        # B x L x V
        pred_logits.add_(-torch.log(-torch.log(torch.Tensor(
            pred_logits.size()).to(pred_logits.device).uniform_(0, 1) + epsilon) + epsilon)) / _gumbel_noise

        pred_tokens = torch.max(pred_logits, dim=-1)[1]
        bos_idx = prev_output_tokens[0, self.config.max_v_len]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1)).to(pred_tokens.device)), pred_tokens], dim=1)[:, :-1]

        sample_gold_prob = self.probs
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens[:, self.config.max_v_len:],
                                                              dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        updated_tokens = prev_output_tokens[:, self.config.max_v_len:] * sample_gold_mask + pred_tokens * (
                1 - sample_gold_mask)
        prev_output_tokens[:, self.config.max_v_len:] = updated_tokens

        return prev_output_tokens

    def _manage_scheduled_sampling(self, prediction_scores_dec1_pass1, input_ids):
        if self.config.scheduled_method == 'confidence':
            input_ids = self._update_scheduled_sampling(prediction_scores_dec1_pass1.detach(), input_ids)
        elif self.config.scheduled_method == 'probability':
            input_ids = self.get_word_orcale_tokens(prediction_scores_dec1_pass1.detach(), input_ids)
        else:
            raise ValueError("Unsupported Method {}".format(self.config.scheduled_method))

        return input_ids

    # * forward (train or translate)
    def _dec1_pass(self, input_ids, input_masks, token_type_ids, video_embeddings, masked_video_embeddings,
                   masked_input_masks, all_input_ids):
        global last_all_masked_ids, last_all_masked_mask

        embeddings_dec1 = self.embeddings_decs[0](input_ids, token_type_ids,
                                                  video_embeddings=video_embeddings)  # (N, L, D)
        '''
        causal inference start 1.get all masked ids, mask, emb
        '''
        zeros_tensor = torch.zeros_like(video_embeddings)
        all_masked_ids_list, all_masked_mask_list = [], []
        for j in range(0, len(all_input_ids)):
            if j < len(all_input_ids) - 1:
                all_masked_ids_list.append(all_input_ids[j][j, :])
                all_masked_mask_list.append(masked_input_masks[j][j, :])
            else:
                last_all_masked_ids = all_input_ids[j][j:, :]
                last_all_masked_mask = masked_input_masks[j][j:, :]

        all_masked_ids = torch.cat((torch.stack(all_masked_ids_list, dim=0), last_all_masked_ids), dim=0)
        all_masked_mask = torch.cat((torch.stack(all_masked_mask_list, dim=0), last_all_masked_mask), dim=0)

        (current_mask_ids_list, next_mask_ids_list, current_emb_list, next_emb_list,
         current_mask_mask_list, next_mask_mask_list) = [], [], [], [], [], []
        '''
        causal inference 2.get current & next masked ids, mask, emb
        '''
        for j in range(0, len(all_input_ids) - 1):
            current_mask_ids = copy.deepcopy(all_masked_ids)
            current_mask_ids[j, :] = input_ids[j, :]
            next_mask_ids = copy.deepcopy(all_masked_ids)
            next_mask_ids[j + 1:, :] = input_ids[j + 1:, :]
            current_mask_ids_list.append(current_mask_ids)
            next_mask_ids_list.append(next_mask_ids)

            current_mask_mask = copy.deepcopy(all_masked_mask)
            current_mask_mask[j, :] = input_masks[j, :]
            next_mask_mask = copy.deepcopy(all_masked_mask)
            next_mask_mask[j + 1:, :] = input_masks[j + 1:, :]
            current_mask_mask_list.append(current_mask_mask)
            next_mask_mask_list.append(next_mask_mask)

            current_emb = copy.deepcopy(zeros_tensor)
            current_emb[:, 50 * j:50 * (j + 1), :] = video_embeddings[:, 50 * j:50 * (j + 1), :]
            current_emb_list.append(current_emb)
            next_emb = copy.deepcopy(zeros_tensor)
            next_emb[:, 50 * (j + 1):50 * len(masked_video_embeddings), :] \
                = video_embeddings[:, 50 * (j + 1):50 * len(masked_video_embeddings), :]
            next_emb_list.append(next_emb)
        '''
        causal inference 3.get current & next output list
        '''
        current_output_list = []
        for ids, mask, emb in zip(current_mask_ids_list, current_mask_mask_list, current_emb_list):
            current_embeddings_dec1 = self.embeddings_decs[0](ids, token_type_ids,
                                                              video_embeddings=emb)  # (N, L, D)
            current_output_list.append(self.decoders[0](
                current_embeddings_dec1, mask, output_all_encoded_layers=False)[-1])  # both outputs are list

        next_output_list = []
        for ids, mask, emb in zip(next_mask_ids_list, next_mask_mask_list, next_emb_list):
            next_embeddings_dec1 = self.embeddings_decs[0](ids, token_type_ids,
                                                           video_embeddings=emb)  # (N, L, D)
            next_output_list.append(self.decoders[0](
                next_embeddings_dec1, mask, output_all_encoded_layers=False)[-1])  # both outputs are list
        '''
        causal inference end
        '''
        masked_embeddings_dec1 = []
        k = 0
        for embeddings in masked_video_embeddings:
            single_embeddings_dec1 = self.embeddings_decs[0](all_input_ids[k], token_type_ids,
                                                             video_embeddings=embeddings)  # (N, L, D)
            masked_embeddings_dec1.append(single_embeddings_dec1)
            k = k + 1
        '''
        input sentences output visual+textual embedding
        decoder: bert encoder 160,72,768 -> head 160,72,4761
        '''
        dec1_outputs = self.decoders[0](
            embeddings_dec1, input_masks, output_all_encoded_layers=False)[-1]  # both outputs are list

        masked_dec1_outputs = []
        i = 0
        for dec in masked_embeddings_dec1:
            single_masked_dec1_outputs = self.decoders[0](
                dec, masked_input_masks[i], output_all_encoded_layers=False)[-1]  # both outputs are list
            masked_dec1_outputs.append(single_masked_dec1_outputs)
            i = i + 1
        # print(relation)
        masked_preds = []
        for output in masked_dec1_outputs:
            mask_pred = self.pred_heads[0](output)
            masked_preds.append(mask_pred)
        prediction_scores_dec1 = self.pred_heads[0](dec1_outputs)  # (S*B, L, vocab_size)

        del current_mask_ids_list, next_mask_ids_list, current_emb_list, next_emb_list, (
            current_mask_mask_list), next_mask_mask_list, all_masked_ids_list, all_masked_mask_list, \
            next_mask_mask, next_emb, zeros_tensor, all_masked_ids, all_masked_mask

        return (prediction_scores_dec1, dec1_outputs, masked_preds, masked_dec1_outputs,
                current_output_list, next_output_list)

    def forward_for_translate(self, query_clip, video_embeddings, input_ids, token_type_ids, input_masks,
                              sentence_embedding=None,
                              embeddings_layer=None, decoder_layer=None, prediction_head=None,
                              confidence_vector=None
                              ):
        embeddings = embeddings_layer(input_ids, token_type_ids, query_clip=query_clip,
                                      video_embeddings=video_embeddings,
                                      sentence_embeddings=sentence_embedding)  # (N, L, D)
        encoded_layer_outputs = decoder_layer(
            embeddings, input_masks, output_all_encoded_layers=False,
            confidence_vector=confidence_vector)  # both outputs are list
        prediction_scores = prediction_head(encoded_layer_outputs[-1])  # (N, L, vocab_size)

        return prediction_scores

    # * sentence embeddings
    def _meta_sen_emb_construction(self, metas, input_masks, dec1_outputs):
        B, S, D = metas

        sentence_embeddings = (input_masks.int().unsqueeze(-1) * dec1_outputs)[:, -(self.config.max_t_len):, :]

        if self.config.sentence_emb_aggregation_mode == 'mean':
            sentence_embeddings = torch.div(torch.sum(sentence_embeddings, dim=1),
                                            torch.sum(input_masks.int()[:, -(self.config.max_t_len):], dim=1).unsqueeze(
                                                -1))
        elif self.config.sentence_emb_aggregation_mode == 'max':
            sentence_embeddings = torch.max(sentence_embeddings, dim=1).values
        elif self.config.sentence_emb_aggregation_mode == 'weighted':
            _weighted = torch.softmax(self._weighted_para(sentence_embeddings), dim=1)
            sentence_embeddings = torch.bmm(_weighted.transpose(1, 2), sentence_embeddings)
            sentence_embeddings = sentence_embeddings.squeeze(1)
        else:
            raise ValueError("Unsupported Aggregation Mode {}".format(self.config.sentence_emb_aggregation_mode))

        sentence_embeddings = sentence_embeddings.view(S, B, D).transpose(0, 1)

        return sentence_embeddings

    def construct_sentence_emb_for_translate(self, input_ids_list_prev, input_masks_list_prev, token_type_ids_list_prev,
                                             video_embeddings,
                                             embeddings_layer, decoder_layer, pred_head, prev_sentence_embeddings=None,
                                             prev_confidence_vector=None):
        # * dec1
        input_ids, token_type_ids, input_masks = \
            _concat_list_of_tensors([input_ids_list_prev, token_type_ids_list_prev, input_masks_list_prev])

        embeddings_prev = embeddings_layer(input_ids, token_type_ids, video_embeddings=video_embeddings,
                                           sentence_embeddings=prev_sentence_embeddings)  # (N, L, D)

        prev_outputs = decoder_layer(
            embeddings_prev, input_masks, confidence_vector=prev_confidence_vector, output_all_encoded_layers=False)[
            -1]  # both outputs are list
        prediction_scores_prev = pred_head(prev_outputs)  # (S*B, L, vocab_size)

        B, S, D = input_ids_list_prev[0].shape[0], len(input_ids_list_prev), prev_outputs.shape[-1]
        metas = (B, S, D)

        sentence_embeddings, confidence_vector = self._conf_n_sen_emb_construction(metas, prediction_scores_prev,
                                                                                   prev_outputs, input_masks)

        return sentence_embeddings, prev_outputs, confidence_vector

    def _conf_n_sen_emb_construction(self, metas, prediction_scores_dec1, dec1_outputs, input_masks, n_S=1):
        B, S, D = metas

        _sentence_embeddings = self._meta_sen_emb_construction(metas, input_masks, dec1_outputs)
        sentence_embeddings = _sentence_embeddings.repeat(n_S, 1, 1)

        confidence_vector = self._meta_conf_vec_construction(metas, prediction_scores_dec1, _sentence_embeddings,
                                                             input_masks)

        confidence_vector = confidence_vector.repeat(n_S, 1)
        try:
            assert confidence_vector.max() < self.config.conf_bucket_size
            assert confidence_vector.min() >= 0
        except:
            confidence_vector = torch.clamp(confidence_vector, min=0, max=(self.config.conf_bucket_size - 1))

        N = (self.config.max_v_len + self.config.max_n_len + (self.config.max_t_len + self.config.max_cot_len))
        _sentence_embeddings = torch.zeros(n_S * B, N, self.config.hidden_size, dtype=sentence_embeddings.dtype,
                                           device=sentence_embeddings.device)
        _sentence_embeddings[:,
        self.config.max_v_len:self.config.max_v_len + self.config.max_n_len] = sentence_embeddings

        N = (self.config.max_v_len + self.config.max_n_len + (self.config.max_t_len + self.config.max_cot_len))
        assert confidence_vector.shape[1] == S
        # whatever in training or inference, return S*B. and would be clipped later
        if confidence_vector.shape[0] != (S * B):
            confidence_vector = confidence_vector.repeat(S, 1)

        _confidence_vector = confidence_vector.view(S, B, S)[0].view(B, S, 1)  # * B,S,1
        _confidence_vector = _confidence_vector.repeat(1, 1, N)
        _confidence_vector = _confidence_vector.transpose(0, 1)
        _confidence_vector = _confidence_vector.reshape(-1, N)  # * S,B,N -> S*B,N

        _confidence_vector[:, self.config.max_v_len:self.config.max_v_len + self.config.max_n_len] = confidence_vector

        assert _confidence_vector.max() < self.config.conf_bucket_size
        assert _confidence_vector.min() >= 0

        return _sentence_embeddings, _confidence_vector

    # * confidence embeddings
    @torch.no_grad()
    def _meta_conf_vec_construction(self, metas, prediction_scores_dec1, senten_emb, input_masks):
        confidence_vector = self._generate_conf_matrix_pred(metas, prediction_scores_dec1, input_masks)  # * B, S
        return confidence_vector.detach()

    @torch.no_grad()
    def _generate_conf_matrix_pred(self, metas, pred, input_masks):
        temprature = self.config.conf_temperature
        B, S, _ = metas
        # * mask: S*B,N
        # * pred: S*B, N, voab
        _word_score = torch.max(pred, dim=-1).values  # * S*B, N
        _word_score = (input_masks.int() * _word_score)[:, -(self.config.max_t_len):]  # * S*B, L

        _sen_score = torch.div(torch.sum(_word_score, dim=1),
                               torch.sum(input_masks.int()[:, -(self.config.max_t_len):], dim=1))  # * S*B,
        _sen_score = _sen_score.view(S, B).transpose(0, 1)  # * B, S
        _sen_score = torch.softmax(_sen_score / temprature, dim=1)

        # * quantize
        _sen_score = (_sen_score * self.conf_bucket_size).floor().long()

        return _sen_score

    @torch.no_grad()
    def _momentum_update_aux_encoder(self):
        """
        Momentum update of the aux encoder
        """
        for param_q, param_k in zip(self.video_encoder.parameters(), self.video_encoder_aux.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self,
                encoder_input, unmasked_encoder_input,
                input_ids_list, input_masks_list, token_type_ids_list, all_input_mask, all_input_ids, relation, opt,
                epoch, mode, input_labels_list=None
                ):
        # * forward while training or validation

        global current_prediction_scores, all_masked_current_prediction_scores, ave_n, ave_n_d
        caption_loss = 0.
        prediction_scores_list = []
        # print(relation)
        # * masked forward
        #
        '''
        video information ebedding input:pretrained video feature
        video_encoder: bertlayer
        '''
        '''
        distance of input pretrained feature
        '''
        # distance_relation = relation[0][0]
        # distance_relation = [int(char) for char in distance_relation]
        num_events = len(relation) + 1
        distance_p, distance_n, num_p, num_n = 0, 0, 0, 0
        ave_p, ave_n, ave_n_d, ave_p_d = 0, 0, 0, 0
        # while distance test
        # for i in range(0, len(encoder_input['all_masked_video_features'])):
        #     if distance_relation[i] == 1:
        #         distance_p += torch.norm(encoder_input['all_masked_video_features'][i]
        #                                  - encoder_input['video_features'])
        #         num_p += 1
        #     else:
        #         distance_n += torch.norm(encoder_input['all_masked_video_features'][i]
        #                                  - encoder_input['video_features'])
        #         num_n += 1
        # ave_p = distance_p / num_p
        # if num_n != 0:
        #     ave_n = distance_n / num_n

        video_embeddings, masked_video_embeddings = self.video_encoder(**encoder_input)
        gt_embeddings = self.video_encoder(**unmasked_encoder_input)
        final_embeddings = gt_embeddings[:, 50 * (num_events - 1):50 * num_events, :]
        # context_embeddings = []
        # aux video loss
        _video_embeddings = self.projection_head_q(video_embeddings)
        # * unmasked forward
        with torch.no_grad():
            self._momentum_update_aux_encoder()  # update the key encoder

            unmasked_video_embeddings = self.video_encoder_aux(**unmasked_encoder_input)
            _unmasked_video_embeddings = self.projection_head_k(unmasked_video_embeddings)

            _unmasked_video_embeddings = _unmasked_video_embeddings * unmasked_encoder_input['video_mask'][..., None]
            _unmasked_video_embeddings = _unmasked_video_embeddings.detach_()

        # * only unmasked part need to calculate loss
        _video_embeddings = _video_embeddings * unmasked_encoder_input['video_mask'][..., None]

        aux_loss = self.config.loss_aux_weight * F.mse_loss(_video_embeddings, _unmasked_video_embeddings,
                                                            reduction='sum')
        aux_loss = aux_loss / torch.sum(unmasked_encoder_input['video_mask']).detach_()
        caption_loss += aux_loss

        B = video_embeddings.shape[0]
        # * dec1
        input_labels, input_ids, token_type_ids, input_masks = \
            _concat_list_of_tensors(
                [input_labels_list[0], input_ids_list[0], token_type_ids_list[0], input_masks_list[0]])
        masked_input_masks = _concat_list_of_tensors(list[0] for list in all_input_mask)
        masked_all_input_ids = _concat_list_of_tensors(list[0] for list in all_input_ids)
        # all_input_ids = [sublist[0] for sublist in all_input_ids]
        if not self.config.disable_scheduled_sampling:
            with (torch.no_grad()):
                (prediction_scores_dec1_pass1, _, all_prediction_scores_dec1_pass1, _,
                 current_embedding_list, next_embedding_list) = self._dec1_pass(
                    input_ids, input_masks, token_type_ids, video_embeddings, masked_video_embeddings,
                    masked_input_masks, masked_all_input_ids)
                input_ids = self._manage_scheduled_sampling(prediction_scores_dec1_pass1, input_ids)
                j = 0
                for predction in all_prediction_scores_dec1_pass1:
                    masked_all_input_ids[j] = self._manage_scheduled_sampling(predction, masked_all_input_ids[j])
                    j = j + 1
        (prediction_scores_dec1, dec1_outputs, all_prediction_scores_dec1_pass1, all_dec1_outputs,
         current_embedding_list, next_embedding_list) = (
            self._dec1_pass(input_ids, input_masks, token_type_ids, video_embeddings
                            , masked_video_embeddings, masked_input_masks, masked_all_input_ids))

        B, S, D = video_embeddings.shape[0], len(input_ids_list[0]), dec1_outputs.shape[-1]
        metas = (B, S, D)

        # * iterative decoder
        prev_outputs = dec1_outputs
        prev_prediction_scores = prediction_scores_dec1
        prev_outputs_all = all_dec1_outputs
        prev_prediction_scores_all = all_prediction_scores_dec1_pass1
        for k in range(1, self.config.K):
            caption_loss += self.config.loss_aux_caption * self.loss_func(
                prev_prediction_scores.view(-1, self.config.vocab_size),
                input_labels.view(-1))

            # * sentence embeddings and confidence embeddings
            sentence_embeddings, confidence_vector = self._conf_n_sen_emb_construction(metas, prev_prediction_scores,
                                                                                       prev_outputs, input_masks, n_S=S)
            all_sentence_embeddings, all_confidence_vector = [], []
            for k1 in range(0, len(prev_outputs_all)):
                sentence_embeddings_masked, confidence_vector_masked = (
                    self._conf_n_sen_emb_construction(metas, prev_prediction_scores_all[k1], prev_outputs_all[k1],
                                                      masked_input_masks[k1], n_S=S))
                all_sentence_embeddings.append(sentence_embeddings_masked)
                all_confidence_vector.append(confidence_vector_masked)

            # * dec2
            input_labels, input_ids, token_type_ids, input_masks = \
                _concat_list_of_tensors(
                    [input_labels_list[k], input_ids_list[k], token_type_ids_list[k], input_masks_list[k]])
            masked_input_masks = _concat_list_of_tensors(list[k] for list in all_input_mask)
            masked_all_input_ids = _concat_list_of_tensors(list[k] for list in all_input_ids)

            # * update input ids with previous prediction
            prev_pred_id = prev_prediction_scores.max(2).values  # S*B, L
            input_ids[:, -(self.config.max_t_len + self.config.max_cot_len)] = (
                prev_pred_id[:, -(self.config.max_t_len + self.config.max_cot_len)].detach())

            prev_prediction_scores_all = [score.max(2).values for score in prev_prediction_scores_all]
            m = 0
            for ids in masked_all_input_ids:
                ids[:, -(self.config.max_t_len + self.config.max_cot_len)] = (
                    prev_prediction_scores_all[m][:, -(self.config.max_t_len + self.config.max_cot_len)].detach())
                m = m + 1
            _video_embeddings = prev_outputs[:, :(self.config.max_v_len), :]
            prev_outputs_all = [outputs[:, :(self.config.max_v_len), :] for outputs in prev_outputs_all]

            embeddings_dec2 = self.embeddings_decs[k](input_ids, token_type_ids,
                                                      video_embeddings=_video_embeddings,
                                                      sentence_embeddings=sentence_embeddings
                                                      )  # (N, L, D)
            all_masked_current_outputs, all_masked_current_prediction_scores = [], []
            for k1 in range(0, len(prev_outputs_all)):
                masked_embeddings_dec2 = self.embeddings_decs[k](masked_all_input_ids[k1], token_type_ids,
                                                                 video_embeddings=prev_outputs_all[k1],
                                                                 sentence_embeddings=all_sentence_embeddings[k1]
                                                                 )
                masked_output = self.decoders[k](
                    masked_embeddings_dec2, masked_input_masks[k1], output_all_encoded_layers=False,
                    confidence_vector=confidence_vector)[
                    -1]
                all_masked_current_outputs.append(masked_output)
                all_masked_current_prediction_scores.append(self.pred_heads[k](masked_output))

            current_outputs = self.decoders[k](
                embeddings_dec2, input_masks, output_all_encoded_layers=False, confidence_vector=confidence_vector)[
                -1]  # both outputs are list

            # * final pred
            current_prediction_scores = self.pred_heads[k](current_outputs)  # (S*B, L, vocab_size)
            prediction_scores_list = current_prediction_scores.view(S, B, *current_prediction_scores.shape[1:])

            # * for next decoder
            prev_outputs = current_outputs  # 12，92，768
            prev_prediction_scores = current_prediction_scores  # 12，92，4761(total words dictionary num)
            prev_outputs_all = all_masked_current_outputs  # n-1 list * 12，92，768
            prev_prediction_scores_all = all_masked_current_prediction_scores

        distance_p, distance_n, num_p, num_n = 0, 0, 0, 0
        # while distance test

        # for i in range(0, len(encoder_input['all_masked_video_features'])):
        #     if distance_relation[i] == 1:
        #         distance_p += torch.norm(prev_outputs_all[i]
        #                                  - prev_outputs)
        #         num_p += 1
        #     else:
        #         distance_n += torch.norm(prev_outputs_all[i]
        #                                  - prev_outputs)
        #         num_n += 1
        # ave_p_d = distance_p / num_p
        # if num_n != 0:
        #     ave_n_d = distance_n / num_n

        # similarity compensation loss
        aux_loss2 = 0
        for i in range(0, len(prev_outputs_all)):
            if relation[i] == 0:
                aux_loss2 += F.mse_loss(prev_outputs, prev_outputs_all[i],
                                        reduction='sum')

        # aggregation ground truth visual information while reasoning in the relation head
        Logits_list = self.relation_head(
            torch.cat((prev_outputs[(num_events - 1), :, :].unsqueeze(0), final_embeddings),
                      dim=1),
            [torch.cat((out[(num_events - 1), :, :].unsqueeze(0), final_embeddings), dim=1)
             for out in prev_outputs_all])

        compensate_list = self.compensate_head(
            [torch.cat((out[(num_events - 1), :, :].unsqueeze(0), final_embeddings), dim=1)
             for out in current_embedding_list],
            [torch.cat((out[(num_events - 1), :, :].unsqueeze(0), final_embeddings), dim=1)
             for out in next_embedding_list])
        # generate complete causal graph while inference
        complete_graph = True
        if complete_graph and mode == 'val':
            # print("output whole graph while validation")  # all_relation
            complete_lists = []
            for i, out in enumerate(prev_outputs_all[:-1]):
                unmask_pred_list, mask_pred_list = [], []
                for j in range(i, len(prev_outputs_all) - 1):
                    unmask_pred = torch.cat(
                        (prev_outputs[j, :, :].unsqueeze(0), gt_embeddings[:, 50 * j:50 * (j + 1), :]), dim=1)
                    mask_pred = torch.cat((out[j, :, :].unsqueeze(0), gt_embeddings[:, 50 * j:50 * (j + 1), :]), dim=1)
                    unmask_pred_list.append(unmask_pred)
                    mask_pred_list.append(mask_pred)
                complete_list = self.relation_head(unmask_pred_list, mask_pred_list, True)
                complete_list = [logits[0, :] for logits in complete_list]
                mse_values = [torch.nn.functional.mse_loss(unmask_pred, mask_pred).item() for unmask_pred, mask_pred in
                              zip(unmask_pred_list, mask_pred_list)]
                mse_mean = sum(mse_values) / len(mse_values)
                quant = opt.hybrid_quant
                quant_sim_list = [
                    torch.tensor([-quant, quant]).cuda() if mse > mse_mean else torch.tensor([quant, -quant]).cuda()
                    for mse in mse_values]
                complete_list_after_sim = [tensor1 + tensor2 for tensor1, tensor2 in zip(complete_list, quant_sim_list)]
                complete_lists.append(complete_list_after_sim)

        # loss calculation  Ce_loss2: causal compensation Ce_loss3: initial logits Ce_loss:final logits
        Ce_loss, Ce_loss2, Ce_loss3 = 0, 0, 0
        compensate_relation = [torch.tensor(relation_item).cuda().repeat(compensate_list[0].shape[0])
                               for relation_item in relation]
        relation = [torch.tensor(relation_item).cuda().repeat(input_labels[0, :].view(-1).shape[0]) for relation_item in
                    relation]

        # initial logits loss & causal compensation loss
        for i in range(0, len(Logits_list)):
            if i < len(Logits_list) - 1:  # causal compensation
                Ce_loss2 += self.CE_loss(compensate_list[i].cuda(), compensate_relation[i])
            Ce_loss3 += self.CE_loss(Logits_list[i].cuda(), relation[i])

        # caption loss
        if self.config.K == 1:
            current_prediction_scores = prev_prediction_scores
            prediction_scores_list = current_prediction_scores.view(S, B, *current_prediction_scores.shape[1:])

        caption_loss += self.config.loss_main_caption * self.loss_func(
            current_prediction_scores[:len(prev_outputs_all), :, :].view(-1, self.config.vocab_size),
            input_labels[:len(prev_outputs_all), :].view(-1))

        result_list = [logits[0, :] for logits in Logits_list]

        # similarity compensation
        mse_values = [torch.nn.functional.mse_loss(prev_output[(num_events - 1), :, :],
                                                   prev_outputs[(num_events - 1), :, :]).item()
                      for prev_output in prev_outputs_all]
        mse_mean = sum(mse_values) / len(mse_values)
        quant = opt.hybrid_quant
        output_list = [torch.tensor([-quant, quant]).cuda() if mse > mse_mean else torch.tensor([quant, -quant]).cuda()
                       for mse in mse_values]

        # generate the final prediction
        result_list = [tensor1 + tensor2 for tensor1, tensor2 in zip(result_list, output_list)]
        compensate_list = [com_list[0] for com_list in compensate_list]
        compensate_list.append(torch.tensor([0, 0]).cuda())
        final_result_list = [(1 - opt.logits_mix_weight) * tensor1 + opt.logits_mix_weight * tensor2 for
                             tensor1, tensor2 in zip(result_list, compensate_list)]
        for i in range(0, len(final_result_list)):
            Ce_loss += self.CE_loss(final_result_list[i].cuda(), relation[i][0])

        # complete loss function
        loss = (caption_loss +
                opt.w_CEloss * Ce_loss3 +
                self.config.loss_aux_weight2 * aux_loss2 +
                opt.w_hybrid_loss1 * Ce_loss +
                opt.w_compensate * Ce_loss2)

        if mode == 'val':
            return loss, prediction_scores_list, final_result_list, caption_loss, ave_n, ave_p, ave_n_d, ave_p_d, complete_lists
        else:
            return loss, prediction_scores_list, final_result_list, caption_loss, ave_n, ave_p, ave_n_d, ave_p_d