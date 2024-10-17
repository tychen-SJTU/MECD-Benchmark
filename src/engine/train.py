import torch
import torch.nn as nn

from src.dataloader import cal_performance, prepare_batch_inputs, MECDDataset
from src.utils import is_distributed, reduce_tensor, get_world_size, dist_log


def freeze_parameters(m, requires_grad=False):
    if m is None:
        return

    if isinstance(m, nn.Parameter):
        m.requires_grad = requires_grad
    else:
        for p in m.parameters():
            p.requires_grad = requires_grad

    return m

def train_epoch(model, training_data, optimizer, ema, device, opt, epoch):
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    n_relation_correct = 0
    n_relation_sum = 0

    if not opt.disable_scheduled_sampling:
        # model.module.set_epoch(epoch)
        model.set_epoch(epoch)

    for batch_idx, batch in enumerate(training_data):
        niter = epoch * len(training_data) + batch_idx

        # * prepare data
        batched_data = prepare_batch_inputs(batch, device=device, non_blocking=opt.pin_memory)
        aux_idx_text = batched_data['aux_idx_text']
        aux_idx_text = [int(data.cpu()) for data in aux_idx_text]
        num_sen = batched_data['num_sen']
        random_index_list = batched_data['random_index_list']
        random_index_list = sorted([int(data.cpu()) for data in random_index_list])
        # * decoders
        input_labels = batched_data['gt']
        input_ids = batched_data['decoder_input']['input_ids']
        input_masks = batched_data['decoder_input']['input_mask']
        token_type_ids = batched_data['decoder_input']['token_type_ids']
        all_input_ids = batched_data['decoder_input']['all_input_ids']
        all_input_mask = batched_data['decoder_input']['all_input_mask']
        relation = batched_data['decoder_input']['relation']

        # * forward & backward 2 5 0
        optimizer.zero_grad()
        relation = relation[0][0]
        relation = [int(char) for char in relation]
        for i, aux in enumerate(aux_idx_text):
            relation[random_index_list[i]] = relation[random_index_list[i]] | relation[aux_idx_text[i]]
        if not opt.use_pretrain:
            model.embeddings_decs[0].textmodel.eval()
            model.embeddings_decs[0].textmodel = freeze_parameters(model.embeddings_decs[0].textmodel)
        loss, pred_scores_list, Logits_list, _, _, _, _, _ = model(
            encoder_input=batched_data['encoder_input'], unmasked_encoder_input=batched_data['unmasked_encoder_input'],
            input_ids_list=input_ids, input_masks_list=input_masks, token_type_ids_list=token_type_ids,
            all_input_mask=all_input_mask,
            all_input_ids=all_input_ids, relation=relation, opt=opt, epoch=epoch, mode='train',
            input_labels_list=input_labels)
        # relation = relation[0][0]
        relation = torch.tensor(relation)
        predict_list = [0 if tensor[0] > tensor[1] else 1 for tensor in Logits_list]
        predict = torch.tensor(predict_list)

        # * normalized by n_sentence and batch size
        if loss is not None:
            n_sen = num_sen.detach().sum()
            if is_distributed() and get_world_size() > 1:
                reduced_n_sen = reduce_tensor(n_sen.float())
            else:
                reduced_n_sen = n_sen

            loss /= (reduced_n_sen.float().item() * get_world_size())
            loss /= float(opt.batch_size)

        if not torch.isnan(loss):
            loss.backward()

        if opt.grad_clip != -1:  # * enable, -1 == disable
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

        optimizer.step()

        # * update model parameters with ema
        if ema is not None:
            ema(model, niter)

        # * keep logs
        n_correct = 0
        n_word = 0
        input_labels = [label[:len(Logits_list) + 1] for label in input_labels]
        for pred, gold in zip(pred_scores_list[:num_sen, :, :, :], input_labels[-1]):
            n_correct += cal_performance(pred, gold)
            valid_label_mask = gold.ne(MECDDataset.IGNORE)
            n_word += valid_label_mask.sum()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss

        sum = relation.size(0)
        correct = predict.eq(relation).sum().item()
        n_relation_correct += correct
        n_relation_sum += sum

        if batch_idx % 50 == 0:
            dist_log('[Training epoch{}:{}/{}] acc:{:.5f} loss:{:.5f} relation_acc:{:.5f}'.format(
                epoch, batch_idx, len(training_data), float(n_word_correct / float(n_word_total)),
                float(total_loss / float(n_word_total)), float(n_relation_correct / float(n_relation_sum))))
            n_relation_correct = 0
            n_relation_sum = 0
    if is_distributed() and get_world_size() > 1:
        reduced_n_word_total = reduce_tensor(n_word_total.float())
        reduced_total_loss = reduce_tensor(total_loss.float())
        reduced_n_word_correct = reduce_tensor(n_word_correct.float())
    else:
        reduced_n_word_total = n_word_total
        reduced_total_loss = total_loss
        reduced_n_word_correct = n_word_correct

    loss_per_word = 1.0 * reduced_total_loss / reduced_n_word_total
    accuracy = 1.0 * reduced_n_word_correct / reduced_n_word_total

    return float(loss_per_word), float(accuracy)

