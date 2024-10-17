import torch

from src.dataloader import cal_performance, prepare_batch_inputs, MECDDataset
from src.utils import is_distributed, reduce_tensor, get_world_size, dist_log


def eval_epoch(model, validation_data, device, opt):
    '''The same setting as training, where ground-truth word x_{t-1}
    is used to predict next word x_{t}, not realistic for real inference'''
    model.eval()

    total_loss = 0
    ppl_loss = 0
    n_word_total = 0
    n_word_correct = 0
    n_relation_correct = 0
    n_relation_sum = 0
    n_relation_correct_0 = 0
    n_relation_sum_0 = 0
    n_relation_correct_1 = 0
    n_relation_sum_1 = 0
    all_pred0 = 0
    all_pred1 = 0
    complete_n_relation_correct = 0
    complete_n_relation_sum = 0
    ave_n_total, ave_p_total, ave_n_d_total, ave_p_d_total, num_positive, num_negative = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        dist_log('Start evaluating.......')

        for batch in validation_data:
            # * prepare data
            batched_data = prepare_batch_inputs(batch, device=device, non_blocking=opt.pin_memory)

            num_sen = batched_data['num_sen']
            # * decoders
            input_labels = batched_data['gt']
            input_ids = batched_data['decoder_input']['input_ids']
            input_masks = batched_data['decoder_input']['input_mask']
            token_type_ids = batched_data['decoder_input']['token_type_ids']
            all_input_ids = batched_data['decoder_input']['all_input_ids']
            all_input_mask = batched_data['decoder_input']['all_input_mask']
            relation = batched_data['decoder_input']['relation']
            all_relation = batched_data['decoder_input']['all_relation'][0]
            complete_relation = all_relation.split(',')
            relation = relation[0][0]
            relation = torch.tensor([int(char) for char in relation])
            complete_relation = [torch.tensor([int(char) for char in relation]) for relation in complete_relation]
            loss, pred_scores_list, Logits_list, caption_loss, ave_n, ave_p, ave_n_d, ave_p_d, complete_lists = model(
                encoder_input=batched_data['encoder_input'],
                unmasked_encoder_input=batched_data['unmasked_encoder_input'],
                input_ids_list=input_ids, input_masks_list=input_masks, token_type_ids_list=token_type_ids,
                all_input_mask=all_input_mask, all_input_ids=all_input_ids,
                relation=relation, opt=opt, epoch=0, mode='val', input_labels_list=input_labels)

            predict_list = [0 if tensor[0] > tensor[1] else 1 for tensor in Logits_list]
            for i, complete_list in enumerate(complete_lists):
                predict_complete = torch.tensor([0 if tensor[0] > tensor[1] else 1 for tensor in complete_list])
                complete_correct = predict_complete.eq(complete_relation[i+1]).sum().item()
                complete_sum = complete_relation[i+1].size(0)
                complete_n_relation_correct += complete_correct
                complete_n_relation_sum += complete_sum

            predict = torch.tensor(predict_list)
            num_positive += 1
            if ave_n != 0:
                num_negative += 1
            ave_n_total += ave_n
            ave_p_total += ave_p
            ave_n_d_total += ave_n_d
            ave_p_d_total += ave_p_d

            # * keep logs
            n_correct = 0
            n_word = 0

            input_labels = [label[:(len(Logits_list) + 1)] for label in input_labels]
            for pred, gold in zip(pred_scores_list[:num_sen, :, :, :], input_labels[-1]):
                n_correct += cal_performance(pred, gold)
                valid_label_mask = gold.ne(MECDDataset.IGNORE)
                n_word += valid_label_mask.sum()

            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss
            ppl_loss += caption_loss

            sum = relation.size(0)
            correct = predict.eq(relation).sum().item()
            n_relation_correct += correct
            n_relation_sum += sum

            correct_0 = predict.eq(relation).logical_and(relation.eq(0)).sum().item()
            n_relation_correct_0 += correct_0
            n_relation_sum_0 += relation.eq(0).sum().item()

            # Calculate the correct predictions for class 1
            correct_1 = predict.eq(relation).logical_and(relation.eq(1)).sum().item()
            n_relation_correct_1 += correct_1
            n_relation_sum_1 += relation.eq(1).sum().item()

            all_pred0 += predict.eq(0).sum().item()
            all_pred1 += predict.eq(1).sum().item()
    # print(ave_n_total/num_negative)
    # print(ave_p_total/num_positive)
    # print(ave_n_d_total/num_negative)
    # print(ave_p_d_total/num_positive)
    if is_distributed() and get_world_size() > 1:
        reduced_n_word_total = reduce_tensor(n_word_total.float())
        reduced_total_loss = reduce_tensor(ppl_loss.float())
        reduced_n_word_correct = reduce_tensor(n_word_correct.float())
    else:
        reduced_n_word_total = n_word_total
        reduced_total_loss = ppl_loss
        reduced_n_word_correct = n_word_correct

    loss_per_word = 1.0 * reduced_total_loss / reduced_n_word_total
    accuracy = 1.0 * reduced_n_word_correct / reduced_n_word_total
    accuracy_relation = float(1.0 * float(n_relation_correct) / float(n_relation_sum))
    accuracy_relation_0 = float(1.0 * float(n_relation_correct_0) / float(n_relation_sum_0))
    precision = float(1.0 * float(n_relation_correct_1) / float(all_pred1))
    # percision = float(1.0 * float(n_relation_correct_0) / float(all_pred0))
    if n_relation_sum_1 != 0:
        accuracy_relation_1 = float(1.0 * float(n_relation_correct_1) / float(n_relation_sum_1))
    else:  # during all zero test
        accuracy_relation_1 = 1.00
    r0_ratio = float(1.0 * float(n_relation_sum_0) / float(n_relation_sum))

    accuracy_relation_complete = float(1.0 * float(complete_n_relation_correct) / float(complete_n_relation_sum))

    return (float(loss_per_word), float(accuracy), accuracy_relation, accuracy_relation_0, accuracy_relation_1,
            r0_ratio, precision, accuracy_relation_complete)
