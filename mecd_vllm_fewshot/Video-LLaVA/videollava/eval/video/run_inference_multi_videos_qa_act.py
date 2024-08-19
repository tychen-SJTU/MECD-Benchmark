
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import argparse
import json

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
        raise Exception("Video {} not found!".format(video_path))
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

    return parser.parse_args([  '--model_path', 'checkpoints/videollava-7b',
                                '--cache_dir', 'cache_dir/',
                                '--video_dir', 'eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/val',
                                '--gt_file_qa', 'eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/QA_val_reason_fewshot.json',
                                '--output_dir', 'eval/GPT_Multi_Videos_QA/Activitynet_Multi_Videos_QA/Video-LLaVA-7B',
                                '--output_name', 'vanilla',
                                '--num_chunks', '1',
                                '--chunk_idx', '0',
                                "--model_max_length", "4096",
                                '--support_num', '0',
                                '--rope_scaling_factor', '1'
                            ])

def get_model_output(model, video_processor, tokenizer, video, support_video_set, qs, task_id, support_num, args):
    video_token_template = "<image><image><image><image><image><image><image><image>"
    # if model.config.mm_use_x_start_end:
    #     qs = DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VID_END_TOKEN + '\n' + qs
    # else:
    #     qs = DEFAULT_VIDEO_TOKEN + '\n' + qs
    
    main_video_prompt = video_token_template + '\n '
    
    support_videos_prompt = ""
    for id in range(args.support_num):
        support_videos_prompt = support_videos_prompt + "related video-{}: ".format(id+1) + video_token_template + '\n '
    
    complete_prompt = main_video_prompt + ' ' + support_videos_prompt + ' ' + qs

    conv_mode = "llava_multi_videos"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], complete_prompt)
    # conv.append_message(conv.roles[1], informer_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    video_set = []
    # get the feature of the main video
    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().to(args.device)
    video_set.append(video_tensor)
    
    # get the feature of the support videos
    for support_video in support_video_set:
        video_tensor = video_processor.preprocess(support_video, return_tensors='pt')['pixel_values'].half().to(args.device)
        video_set.append(video_tensor)
    
    video_batch = torch.cat(video_set, dim=0)
    
    # task_dir_path = 'video_frame/task_{}'.format(task_id)
    # if not os.path.exists(task_dir_path):
    #     os.makedirs(task_dir_path)
    
    # for video_id, video in enumerate(video_batch):
    #     video_dir_path = os.path.join(task_dir_path, 'video_{}'.format(video_id))
    #     if not os.path.exists(video_dir_path):
    #         os.makedirs(video_dir_path)
        
    #     # video de-normalization
    #     std = torch.tensor(OPENAI_DATASET_STD)[:, None, None, None]
    #     mean = torch.tensor(OPENAI_DATASET_MEAN)[:, None, None, None]
        
    #     video_output = (video.cpu() * std + mean)
    #     frame_num = video_output.shape[1]
        
    #     for frame_id in range(frame_num):
    #         frame = video_output[:, frame_id, :, :]
    #         frame_path = os.path.join(video_dir_path, 'frame_{}.jpg'.format(frame_id))
    #         vutils.save_image(frame, frame_path)
    
    video_batch = video_batch[: support_num+1]
    
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
    
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, rope_scaling_factor=args.rope_scaling_factor)
    model = model.to(args.device)
    gt_questions_answers = json.load(open(args.gt_file_qa, "r"))
    # only test a subset
    gt_questions_answers = gt_questions_answers[: QA_NUM]
    
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    # ans_file = open(answers_file, "w")
    final_answer = []

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results

    # Iterate over each sample in the ground truth file
    
    for task_id, sample in tqdm(enumerate(gt_questions_answers)):
        video_name = 'v_' + sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = sample['answer']

        sample_set = {'id': id, 'question': question, 'answer': answer}
        multi_video_set = sample['fewshot_set']
        
        main_video_path = os.path.join(args.video_dir, video_name)
        main_video_path = get_video_format(main_video_path)
    
        for id, support_video in enumerate(multi_video_set):
            support_video = 'v_' + support_video
            video_path = os.path.join(args.video_dir, support_video)
            video_path = get_video_format(video_path)
            multi_video_set[id] = video_path
        
        # multi_video_set = multi_video_set[:args.support_num]
        # try:
        # Run inference on the video and add the output to the list
        output = get_model_output(model, processor['video'], tokenizer, main_video_path, multi_video_set, question, task_id, args.support_num, args)
        sample_set['pred'] = output
        output_list.append(sample_set)
        # except Exception as e:
        #     print(f"Error processing video file '{video_name}': {e}")
        # ans_file.write(json.dumps(sample_set) + "\n")
        final_answer.append(sample_set)

    with open(answers_file, 'w') as f:
        json.dump(final_answer, f)
    # ans_file.close()
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
