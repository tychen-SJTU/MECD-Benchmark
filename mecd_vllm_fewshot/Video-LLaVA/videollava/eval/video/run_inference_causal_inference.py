import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import argparse
import json
import numpy as np

import torch
import transformers
from tqdm import tqdm
from torchvision import utils as vutils

from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_VID_START_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VID_END_TOKEN, IMAGE_TOKEN_INDEX
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
QA_NUM = 100


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_video_format(video_path):
    if os.path.exists(video_path + '.mp4'):
        video_path = video_path + '.mp4'
    elif os.path.exists(video_path + '.mkv'):
        video_path = video_path + '.mkv'
    elif os.path.exists(video_path + '.avi'):
        video_path = video_path + '.avi'
    elif os.path.exists(video_path + '.mov'):
        video_path = video_path + '.mov'
    elif os.path.exists(video_path + '.webm'):
        video_path = video_path + '.webm'
    else:
        video_path = None
    return video_path


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_qa', help='Path to the ground truth file containing question and answer.',
                        required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--support_num", type=int, default=1)
    parser.add_argument("--rope_scaling_factor", type=int, default=1)

    return parser.parse_args(['--model_path', 'your_path/Video-LLaVA-ft/checkpoints/llava_causal_discovery_lora',
                              '--cache_dir', 'cache_dir/',
                              '--video_dir',
                              'your_path/Video-LLaVA/eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/val',
                              '--gt_file_qa',
                              'your_path/Video-LLaVA/eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/QA_val_reason_fewshot.json',
                              '--output_dir',
                              'your_path/Video-LLaVA/eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/Video-LLaVA-7B',
                              '--output_name', 'informer_support_3_rope_4',
                              '--num_chunks', '1',
                              '--chunk_idx', '0',
                              "--model_max_length", "4096",
                              '--support_num', '3',
                              '--rope_scaling_factor', '1'
                              ])


# get_model_output(model, processor['video'], tokenizer, question, task_id, args)
def get_model_output(model, video_processor, tokenizer, video, qs, task_id, times, args):
    video_token_template = DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VID_END_TOKEN
    # [0, 0, 1, 0, 1, 1, 1, 0]

    conv_mode = "llava_causal_inference"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    # conv.append_message(conv.roles[1], informer_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    video_set = []
    # get the feature of the main video
    # video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().to(args.device)
    video_tensor = video_processor.preprocess(video, times=times, return_tensors='pt')['pixel_values'].bfloat16().to(
        args.device)

    video_set.append(video_tensor)

    video_batch = torch.cat(video_set, dim=0)

    # print(video_tensor.shape)
    # print(prompt)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    '''
    images (X_modalities) [
            [img_feature, img_feature, video_feature, audio_feature],
            ['image', 'image', 'video', 'audio']
            ]
    '''

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            # images=[[video_tensor], ['video']],
            images=video_batch,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)

    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path,
                                                                     "your_path/Video-LLaVA/cache_dir/models--LanguageBind--Video-LLaVA-7B/snapshots/e16778e47e589512d7e69f964850c8cad775a335",
                                                                     model_name,
                                                                     rope_scaling_factor=args.rope_scaling_factor)
    model = model.to(args.device)

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
    # gt_questions_answers = json.load(open(args.gt_file_qa, "r"))
    # only test a subset
    # gt_questions_answers = gt_questions_answers[: QA_NUM]

    # answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    # ans_file = open(answers_file, "w")
    final_answer = []

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results

    causal_inference_file = './captions/test.json'
    with open(causal_inference_file, 'r') as f:
        causal_inference_list = json.load(f)

    pos_num = 0
    total_num = 0
    total_false_num = 0
    false_num = 0
    result = []
    # Iterate over each sample in the ground truth file
    for task_id, sample in tqdm(enumerate(causal_inference_list)):
        key = [s for s in sample][0]
        ce_sample = sample[key]
        description = ce_sample['sentences']
        duration = ce_sample['duration']
        times = ce_sample['timestamps']
        times.append(duration)

        question_head = "Text description of {} events:".format(len(description))
        question_tail = "The probability output should be (length should be {}):".format(len(description) - 1)
        question_body = ""
        for line in description:
            question_body += line.replace('.', ',')
        question_body = question_body[:-1] + '.'
        question = question_head + question_body + question_tail
        answer = ce_sample['relation']

        # sample_set = {'id': task_id, 'question': question, 'answer': answer}
        # main_video_path = '/mnt/sdb/dataset/Activitynet/v1-3'
        video_name = '/mnt/sdb/dataset/Activitynet/v1-3/train/' + key
        video_path = get_video_format(video_name)
        if video_path is None:
            video_name = '/mnt/sdb/dataset/Activitynet/v1-3/val/' + key
            video_path = get_video_format(video_name)
        # main_video_path = get_video_format(main_video_path)
        # multi_video_set = multi_video_set[:args.support_num]
        # try:
        # Run inference on the video and add the output to the list
        main_video_path = video_path
        output = get_model_output(model, processor['video'], tokenizer, main_video_path, question, task_id, times, args)
        gt_answer = list(answer[0])
        # output = [int(out) for out in output]
        # output = output[1: -1].split(',')
        # print(gt_answer)
        # count=0
        # while len(gt_answer) != len(output) and count < 5:
        #     print('Retry...')
        #     output = get_model_output(model, processor['video'], tokenizer, main_video_path, question, task_id, times, args)
        #     gt_answer = list(answer[0])
        #     output = output[1: -1].split(',')
        #     print(gt_answer)
        #     print(output)
        #     count = count+1

        # gt_answer = np.array([int(char) for char in gt_answer])
        # output = np.array([int(char) for char in output])
        # # output = output[:gt_answer.shape[0]]
        # zero_answer = np.zeros_like(gt_answer)

        # if len(zero_answer) != len(output):
        #     false_num += 1
        # else:
        #     pos_num += (zero_answer == output).sum()
        #     total_num += len(zero_answer)
        # relation={}
        # try:
        gt_answer = np.array([int(char) for char in gt_answer])
        output = np.array([int(char) for char in output])
        # output = output[:gt_answer.shape[0]]
        zero_answer = np.zeros_like(gt_answer)

        if len(gt_answer) == len(output):
            pos_num += (gt_answer == output).sum()
            total_num += len(zero_answer)
            total_false_num += len(zero_answer)

        relation = {"name": key, "gt": gt_answer.tolist(), "pred": output.tolist()}
        result.append(relation)

        # except:
        #     false_num = false_num+1
        #     total_false_num += len(gt_answer)
        #     relation={"name":key, "gt":gt_answer.tolist(), "pred":[]}
        #     result.append(relation)

        # sample_set['pred'] = output
        # output_list.append(sample_set)
        # final_answer.append(sample_set)
    print("Accuracy: {}".format(pos_num / total_num))
    print("Accuracy(w false): {}".format(pos_num / total_false_num))
    print("False Number: {}".format(false_num))
    with open('your_path/Video-LLaVA/videollava_in_context_relation_pred.json', 'w') as f:
        json.dump(result, f, indent=4)

    # with open(answers_file, 'w') as f:
    #     json.dump(final_answer, f)
    # ans_file.close()
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
