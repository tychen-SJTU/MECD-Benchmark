from utils.config import Config
config_file = "configs/config.json"
cfg = Config.from_file(config_file)
import os
import io

from models.videochat2_it import VideoChat2_it
from utils.easydict import EasyDict
import torch
import json

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

from IPython.display import Video, HTML

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
    
    conv.messages.append([conv.roles[1], answer_prompt])
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
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text, output_token.cpu().numpy()

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

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
    results = {}
    test_csv_file = 'your_path/TimeCraft/datasets/nextgqa/sub_anno/llama/sorted_test_rewrite.csv'
    df = pandas.read_csv(test_csv_file)
    finished_video_ids = []

    # Define task
    task = 'Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n'
    answer_prompt="Best option:("
    question_prompt = '\nOnly give the best option.'
    return_prompt='('


    for index, row in df.iterrows():
        video_id = row['video_id']
        question = row['question'] + '?'
        gt_answer = row['answer']
        qid = row['qid']
        a0, a1, a2, a3, a4 = row['a0'], row['a1'], row['a2'], row['a3'], row['a4']
        r_a0, r_a1, r_a2, r_a3, r_a4, = row['r_a0'], row['r_a1'], row['r_a2'], row['r_a3'], row['r_a4']
        rewritten_options = [r_a0, r_a1, r_a2, r_a3, r_a4]
        ori_options = [a0,a1,a2,a3,a4]
        rewritten_options.remove(gt_answer)
        rewritten_options.insert(ori_options.index(gt_answer), gt_answer)
        s_r_a0, s_r_a1, s_r_a2, s_r_a3, s_r_a4 = rewritten_options

        answer_id = [a0,a1,a2,a3,a4].index(gt_answer) + 1
        answer_r_id = [r_a0, r_a1, r_a2, r_a3, r_a4].index(gt_answer) + 1
        if video_id not in finished_video_ids:
            # Load new video
            print('Test video: ', video_id)
            finished_video_ids.append(video_id)
            vid_path = 'your_path/videos/NExTVideo/' + str(video_id) + '.mp4'
            vid, msg = load_video(vid_path, num_segments=8, return_msg=True, resolution=224)
            new_pos_emb = get_sinusoid_encoding_table(n_position=(224//16)**2*8, cur_frame=8)
            model.vision_encoder.encoder.pos_embed = new_pos_emb
            # The model expects inputs of shape: T x C x H x W
            TC, H, W = vid.shape
            video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")
            img_list = []
            with torch.no_grad():
                image_emb, _ = model.encode_img(video, "")
            img_list.append(image_emb)

        question_ra = 'Q: ' + question + '\n(A) ' + r_a0 + '\n(B) ' + r_a1 + '\n(C) ' + r_a2 + '\n(D) ' + r_a3 + '\n(E) ' + r_a4
        question_ra_s = 'Q: ' + question + '\n(A) ' + s_r_a0 + '\n(B) ' + s_r_a1 + '\n(C) ' + s_r_a2 + '\n(D) ' + s_r_a3 + '\n(E) ' + s_r_a4

        # infer rewrite QA
        chat = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        prompt_r = task + question_ra + question_prompt
        chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])
        ask(prompt_r, chat)
        llm_message_r = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=100, answer_prompt=answer_prompt, print_res=True)[0]
        final_output_r = return_prompt + llm_message_r.strip().split('\n')[0]

        # infer rewrite QA with same order with original options
        chat = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        prompt_r_s = task + question_ra_s + question_prompt
        chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])
        ask(prompt_r_s, chat)
        llm_message_r_s = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=100, answer_prompt=answer_prompt, print_res=True)[0]
        final_output_r_s = return_prompt + llm_message_r_s.strip().split('\n')[0]

        # record results
        if video_id not in results:
            results[video_id] = {}
        # results[video_id][qid] = {'question':question, 'gt_answer':gt_answer, 'gt_idx_r': answer_r_id, \
        #                           'gt_idx': answer_id, 'pred_r':final_output_r}
        results[video_id][qid] = {'question':question, 'gt_answer':gt_answer, 'gt_idx_r': answer_r_id, \
                                  'gt_idx_r_s': answer_id, 'pred_r':final_output_r, 'pred_r_s': final_output_r_s}
        if index % 100 == 0:
            with open('your_path/Ask-Anything/video_chat2/results/whole_test/videochat2_llama.json', 'w') as f:
                json.dump(results, f)
    
    # Save results into json file
    with open('your_path/Ask-Anything/video_chat2/results/whole_test/videochat2_llama.json', 'w') as f:
        json.dump(results, f)