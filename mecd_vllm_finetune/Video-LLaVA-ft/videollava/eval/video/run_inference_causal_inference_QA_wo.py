
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append("/home/hetianyao/Video-LLaVA")
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
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


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
    parser.add_argument('--gt_file_qa', help='Path to the ground truth file containing question and answer.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--support_num", type=int, default=1)
    parser.add_argument("--rope_scaling_factor", type=int, default=1)

    return parser.parse_args([  '--model_path', '/home/hbliu/hg_models/Video-LLaVA-7B',
                                '--cache_dir', 'cache_dir/',
                                '--video_dir', '/home/tychen/Video-LLaVA/eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/val',
                                '--gt_file_qa', '/home/tychen/MECD/VAR-main/causal_qa_val.json',
                                '--output_dir', '/home/tychen/Video-LLaVA/eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/Video-LLaVA-7B',
                                '--output_name', 'informer_support_3_rope_4',
                                '--num_chunks', '1',
                                '--chunk_idx', '0',
                                "--model_max_length", "4096",
                                '--support_num', '3',
                                '--rope_scaling_factor', '1'
                            ])

def get_model_output(model, video_processor, tokenizer, video, qs, task_id, times, args):

    conv_mode = "v1_mmtag"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    # conv.append_message(conv.roles[1], informer_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    video_set = []
    # get the feature of the main video
    # video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().to(args.device)
    video_tensor = video_processor.preprocess(video, times=times, return_tensors='pt')['pixel_values'].bfloat16().to(args.device)

    video_set.append(video_tensor)

    video_batch = torch.cat(video_set, dim=0)

    # print(video_tensor.shape)
    # print(prompt)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

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

    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    global qasample, two_input_list, two_input_str, question, sample, key
    model_name = get_model_name_from_path(args.model_path)

    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, rope_scaling_factor=args.rope_scaling_factor)
    model = model.to(args.device)
    gt_questions_answers = json.load(open(args.gt_file_qa, "r"))

    # answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    # ans_file = open(answers_file, "w")
    final_answer = []

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results

    causal_inference_file = '/home/tychen/MECD/VAR-main/captions/activitynet/val_caption_small3_updated_large.json'
    with open(causal_inference_file, 'r') as f:
        causal_inference_list = json.load(f)

    result = []
    # Iterate over each sample in the ground truth file
    for qasample in tqdm(gt_questions_answers):
        question = qasample['question']
        question_id = qasample['question_id']
        for task_id, cau_sample in tqdm(enumerate(causal_inference_list)):
            key = [s for s in cau_sample][0]
            if key[2:] == qasample['video_name']:
                sample = cau_sample
                break
        # question = 'Please answer: ' + question.capitalize() + '? '
        ce_sample = sample[key]
        description = ce_sample['sentences']
        duration = ce_sample['duration']
        times = ce_sample['timestamps']

        times = times[:len(description)]
        times.append(duration)
        # one_input_list = list(range(len(times)))
        video_name = '/mnt/sdb/dataset/Activitynet/v1-3/train/' + key
        video_path = get_video_format(video_name)
        if video_path is None:
            video_name = '/mnt/sdb/dataset/Activitynet/v1-3/val/' + key
            video_path = get_video_format(video_name)

        main_video_path = video_path
        output = get_model_output(model, processor['video'], tokenizer, main_video_path, question, task_id, times, args)
        relation = {"name": key, "pred": output, "question_id": question_id}
        result.append(relation)

    with open('/home/tychen/Video-LLaVA/causal_qa_video_llava_wo.json', 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)