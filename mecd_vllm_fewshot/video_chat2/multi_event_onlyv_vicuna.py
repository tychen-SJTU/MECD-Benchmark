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

# from IPython.display import Video, HTML

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
    initial_frames = len(timestamps)
    if initial_frames > 8:
        timestamps = timestamps[-8:]
    if initial_frames < 8:
        event_lengths = [time[1] - time[0] for time in timestamps]
        sorted_indices = sorted(range(len(event_lengths)), key=lambda i: event_lengths[i], reverse=True)
        for i in range(0, 8 - initial_frames):
            new_stamps = (timestamps[sorted_indices[i]][0] + 0.5 *
                          (timestamps[sorted_indices[i]][1] - timestamps[sorted_indices[i]][0]))
            timestamps.append([round(timestamps[sorted_indices[i]][0], 2), round(new_stamps, 2)])
            timestamps.append([round(new_stamps, 2), round(timestamps[sorted_indices[i]][1], 2)])
        for i in range(0, 8 - initial_frames):
            del timestamps[sorted_indices[i]]
    timestamps = sorted(timestamps, key=lambda x: x[0])
    assert len(timestamps) == 8
    frame_stamps = [time[0] + 0.5 * (time[1] - time[0]) for time in timestamps]
    frame_indices = np.array([int(time * num_frames / duration_annotation) for time in frame_stamps])

    frame_indices0 = get_index(num_frames, num_segments)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std)
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
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
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table



if __name__ == '__main__':
    # Define task
    task="Task: Each video consists of n events, and the text description of each event has been given correspondingly (separated by " ",). You need to judge whether the first n-1 events in the video are the cause of the last event, the probability of the cause 0(irrelevant) or 1(relevant) is expressed as the output. Here are several example of judging whether the first n-1 events in the video are the cause of the last event:"
    example1 = '''First example: Text description of 6 events:\
           A man wearing a black clothes is sharpening the knife on a stone,\
           The man beside him on blue long sleeves drawing something on the table with his finger,\
           The man turn the knife to sharpen the other side and then wipe it with paper towel,\
           The man in navy blue shirt point to the knife while the man sharpening the knife tries to sharpen it one hand,\
           He continues sharpening the knife, turn it again to further sharpen the other side and wipe it again with paper towel,\
           Throws the old and dirty paper towel and reach the roll of paper towel and clean the knife.\
           The probability output should be (length should be 5): [0, 0, 1, 0, 1].'''
           
    example2 = '''Second example: Text description of 9 events: \
            A woman holds a yellow ball behind her neck,\
            She turns around and launches the ball onto a field,\
            People run to measure the distance,\
            A man in a red shirt holds a ball behind his neck,\
            He turns around and launches the ball on the field,\
            People run to measure the distance of it,\
            Another woman holds a ball behind her neck,\
            She also turns around and launches the ball onto the field,\
            People then run over to measure the distance.\
            The probability output should be (length should be 8): [0, 0, 0, 0, 0, 0, 1, 1]'''
            
    example3 = '''Third example: Text description of 9 events:\
            The white water polo team huddles together,\
            The game begins and blue scores while deep in the white defense to tie the game,\
            White answers with a powerful goal to grab the lead back,\
            Blue moves in and after a couple of passes ties the game back up,\
            White scores to take the lead, and blue comes right back to tie once again,\
            A scramble for the ball results in white gaining a 2-on-1 advantage and a goal,\
            They push their lead further late and put the game out of reach, winning 14-10,\
            The team celebrates together in the water,\
            The team celebrates their gold metal at the podium.\
            The probability output should be (length should be 8): [0, 0, 1, 0, 1, 1, 1, 0]'''
    request_prompt = "The assistant should give probability output to the user's input video descriptions."
    sep=" "
    task_prompt = task+sep+example1+sep+example2+sep+example3+sep+request_prompt
    
    # task = 'Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n'
    answer_prompt=None
    # question_prompt = '\nOnly give the best option.'
    # return_prompt='('

    causal_inference_file = 'your_path/MECD/VAR-main/captions/activitynet/val_caption_small3_updated_large.json'
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
        question_head = "Visual input of {} events:".format(len(description))
        question_tail = " The probability output should be (length should be {}): [".format(len(description)-1)
        # answer_prompt = "The probability output should be (length should be {}): [".format(len(description)-1)
        times = ce_sample['timestamps']
        duration = ce_sample['duration']
        if len(times) < 9:
            one_input_list = list(range(len(times)))
            if len(times) < 8:
                event_lengths = [time[1] - time[0] for time in times]
                sorted_indices = sorted(range(len(event_lengths)), key=lambda i: event_lengths[i], reverse=True)
                two_input_list = sorted(sorted_indices[: (8 - len(times))])
                one_input_list = [x for x in one_input_list if x not in two_input_list]
            length_list = []
            for i in range(0, len(times)):
                if i in one_input_list:
                    length_list.append(1)
                else:
                    length_list.append(2)
            visual_body = []
            for i, length in enumerate(length_list):
                if length == 1:
                    visual_body.append(f"Image {sum(length_list[:i])} represents event {i}.")
                else:
                    visual_body.append(f"Image {sum(length_list[:i])}, {sum(length_list[:i]) + 1} represent event {i}.")
            visual_body = " ".join(visual_body)
            question = question_head + visual_body + question_tail
            # question = question_head + question_body

            gt_answer = ce_sample['relation']

            sample_set = {'id': task_id, 'question': question, 'answer': gt_answer}

            vid_val_root = '/mnt/sdb/dataset/Activitynet/v1-3/val'
            vid_train_root = '/mnt/sdb/dataset/Activitynet/v1-3/train'

            vid_path = os.path.join(vid_val_root, "%s.mp4"%key)
            if not os.path.exists(vid_path):
                vid_path = os.path.join(vid_val_root, "%s.mkv"%key)
                if not os.path.exists(vid_path):
                    vid_path = os.path.join(vid_train_root, "%s.mkv"%key)
                    if not os.path.exists(vid_path):
                        vid_path = os.path.join(vid_train_root, "%s.mp4"%key)
            print(vid_path)
            vid, msg = load_video(vid_path, times, duration, num_segments=8, return_msg=True, resolution=224)
            new_pos_emb = get_sinusoid_encoding_table(n_position=(224//16)**2*8, cur_frame=8)
            model.vision_encoder.encoder.pos_embed = new_pos_emb
            # The model expects inputs of shape: T x C x H x W
            TC, H, W = vid.shape
            video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")
            img_list = []
            with torch.no_grad():
                image_emb, _ = model.encode_img(video, "")
            img_list.append(image_emb)

            # infer rewrite QA
            chat = EasyDict({
                "system": "",
                "roles": ("Human", "Assistant"),
                "messages": [],
                "sep": " "
            })
            # prompt_r = task_prompt + question_ra + question_prompt
            chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])
            chat.messages.append([chat.roles[0], task_prompt])
            chat.messages.append([chat.roles[0], question])

            # chat.messages.append([chat.roles[1], ""])

            output = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=100, answer_prompt=answer_prompt, print_res=False)
            # print(output)
            gt_answer = list(gt_answer[0])
            output = output.split(',')

            try:
                gt_answer = np.array([int(char) for char in gt_answer])
                output = np.array([int(char) for char in output])
                print(output)
                print(gt_answer)
                # output = output[:gt_answer.shape[0]]
                zero_answer = np.zeros_like(gt_answer)


                pos_num += (gt_answer == output).sum()
                total_num += len(zero_answer)
                total_false_num += len(zero_answer)
                relation={"name":key, "gt":gt_answer.tolist(), "pred":output.tolist()}
                result.append(relation)

            except:
                false_num = false_num+1
                total_false_num += len(gt_answer)
                relation = {"name": key, "gt": gt_answer.tolist(), "pred": []}
                result.append(relation)

    print("Accuracy: {}".format(pos_num / total_num))
    print("Accuracy(w false): {}".format(pos_num / total_false_num))
    print("False Number: {}".format(false_num))
    with open('./video_chat2_relation_pred_onlyv2.json', 'w') as f:
        json.dump(result, f, indent=4)