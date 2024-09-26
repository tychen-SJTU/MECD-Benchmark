from utils.config import Config

config_file = "configs/config.json"
cfg = Config.from_file(config_file)
import os
import io

from models.videochat2_it import VideoChat2_it
from utils.easydict import EasyDict
import torch
import json
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList
import pandas
from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop,
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

import matplotlib.pyplot as plt

from peft import get_peft_model, LoraConfig, TaskType
import copy

# load stage2 model
cfg.model.vision_encoder.num_frames = 4
model = VideoChat2_it(config=cfg.model)

# add lora to run stage3 model
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=16, lora_alpha=32, lora_dropout=0.
)
model.llama_model = get_peft_model(model.llama_model, peft_config)

state_dict = torch.load("your_path/videochat_models/videochat2_7b_stage3.pth", "cpu")

if 'model' in state_dict.keys():
    msg = model.load_state_dict(state_dict['model'], strict=False)
else:
    msg = model.load_state_dict(state_dict, strict=False)
# print(msg)

model = model.eval()
model = model.cuda()
print('Load the VideoChat2 model')


def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret


# def get_prompt(conv):
#     ret = conv.system
#     for role, message in conv.messages:
#         ret += conv.sep
#         if message:
#             ret += role + ": " + message
#         else:
#             ret += role + ":"
#     return ret

def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.llama_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
           repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([835]).to("cuda:0"),
        torch.tensor([2277, 29937]).to("cuda:0")]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    # conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    # print(output_text)
    output_text = output_text.split('\n')[-1]  # remove the stop sign '###'
    # print(output_text)
    output_text = output_text.replace('] ###', '')  # remove the stop sign '###'
    # print(output_text)
    # output_text = output_text.strip()
    # conv.messages[-1][1] = output_text
    return output_text


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, timestamps, duration_annotation, num_segments=8, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_stamps = [time[0] + 0.5 * (time[1] - time[0]) for time in timestamps]
    frame_indices = np.array([int(time * num_frames / duration_annotation) for time in frame_stamps])
    images_group = list()
    video_path = video_path.split('/')[-1][:-4]

    for i, frame_index in enumerate(frame_indices):
        img = Image.fromarray(vr[frame_index].numpy())
        file_name = f"{i}.jpg"
        if not os.path.exists(os.path.join('your_path/video_chat2/0set', video_path)):
            os.makedirs(os.path.join('your_path/video_chat2/0set', video_path))
        save_path = os.path.join('your_path/video_chat2/0set', video_path, file_name)
        img.save(save_path)
        images_group.append(img)

    return images_group


def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # generate checkpoint position embedding

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")

    if n_position != pre_n_position:
        T = ckpt_num_frame  # checkpoint frame
        P = 14  # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5)  # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame  # checkpoint frame
        new_T = cur_frame  # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5)  # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3)  # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

    return sinusoid_table


if __name__ == '__main__':

    causal_inference_file = 'your_path/MECD/VAR-main/mecd0set_0.json'
    with open(causal_inference_file, 'r') as f:
        causal_inference_list = json.load(f)

    # Iterate over each sample in the ground truth file
    for task_id, sample in tqdm(enumerate(causal_inference_list)):
        key = [s for s in sample][0]
        ce_sample = sample[key]
        description = ce_sample['sentences']
        duration = ce_sample['duration']
        times = ce_sample['timestamps']

        vid_val_root = '/mnt/sdb/dataset/Activitynet/v1-3/val'
        vid_train_root = '/mnt/sdb/dataset/Activitynet/v1-3/train'

        vid_path = os.path.join(vid_val_root, "%s.mp4" % key)
        if not os.path.exists(vid_path):
            vid_path = os.path.join(vid_val_root, "%s.mkv" % key)
            if not os.path.exists(vid_path):
                vid_path = os.path.join(vid_train_root, "%s.mkv" % key)
                if not os.path.exists(vid_path):
                    vid_path = os.path.join(vid_train_root, "%s.mp4" % key)
        # print(vid_path)
        img = load_video(vid_path, times, duration, num_segments=len(description), return_msg=True, resolution=224)