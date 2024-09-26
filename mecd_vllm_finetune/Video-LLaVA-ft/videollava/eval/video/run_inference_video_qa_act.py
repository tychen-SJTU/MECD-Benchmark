
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import math
import argparse
import json

import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_VID_START_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VID_END_TOKEN, IMAGE_TOKEN_INDEX
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
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
    return parser.parse_args([  '--model_path', '/home/tychen/Video-LLaVA/checkpoints/videollava-7b',
                                '--cache_dir', 'cache_dir/',
                                '--video_dir', 'eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/val',
                                '--gt_file_qa', '/home/tychen/MECD/VAR-main/causal_qa_val.json',
                                '--output_dir', 'eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/Video-LLaVA-7B',
                                '--output_name', 'informer_support_3_rope_4',
                                '--num_chunks', '1',
                                '--chunk_idx', '0',
                                "--model_max_length", "4096",
                                '--support_num', '3',
                                '--rope_scaling_factor', '1'
                            ])


def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if model.config.mm_use_x_start_end:
        qs = DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_VIDEO_TOKEN + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    # video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().to(args.device)
    
    # print(video_tensor.shape)
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
            images=video_tensor,
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
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)

    # Load both ground truth file containing questions and answers
    # with open(args.gt_file_question) as file:
    #     gt_questions = json.load(file)
    # with open(args.gt_file_answers) as file:
    #     gt_answers = json.load(file)

    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    # gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in tqdm(video_formats):  # Added this line
            temp_path = os.path.join(args.video_dir, f"v_{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                # try:
                # Run inference on the video and add the output to the list
                output = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
                sample_set['pred'] = output
                output_list.append(sample_set)
                # except Exception as e:
                #     print(f"Error processing video file '{video_name}': {e}")
                ans_file.write(json.dumps(sample_set) + "\n")
                break
    
    ans_file.close()
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
