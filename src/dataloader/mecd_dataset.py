import os
import math
import nltk
import numpy as np
from PIL import Image
import torch
import random
import copy
import torchvision
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from src.dataloader.transforms import ToTorchFormatTensor, Stack, GroupNormalize, GroupMultiScaleCrop, \
    GroupRandomHorizontalFlip
from src.utils import load_json, flat_list_of_lists
from torch.nn.utils.rnn import pad_sequence
import logging

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


class MECDDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+sentence+text joint sequence

    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"

    SEN_TOKEN = "[SEN]"  # used as placeholder in the clip+sentence+text joint sequence
    MSK_TOKEN = "[MSK]"

    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    SEN = 7
    MSK = 8
    IGNORE = -1  # used to calculate loss

    VID_TYPE = 0
    TEX_TYPE = 1
    SEN_TYPE = 2

    def __init__(self, dset_name, data_dir,
                 max_t_len, max_v_len, max_n_len, max_cot_len, del_words, mask_frames, json_path, use_existence, K,
                 multi_chains_b=1, multi_chains_k=3,
                 mode="train",
                 validation_set='',
                 sample_mode='uniform',
                 word2idx_path=None):
        self.aux_idx_text = None
        self.multi_chains_b = multi_chains_b
        self.multi_chains_k = multi_chains_k
        self.dset_name = dset_name
        self.validation_set = validation_set
        self.data_dir = data_dir  # containing training data
        self.json_path = json_path
        self.mode = mode
        self.new_length = 1
        self.num_segments = 8
        self.sample_mode = sample_mode
        self.random = True
        self.use_existence = use_existence
        self.K = K  # * number of cascade decoders
        self.txt = "output_train.txt"
        self.del_words = del_words
        self.mask_frames = mask_frames
        assert K >= 1

        normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_augmentation = torchvision.transforms.Compose([GroupMultiScaleCrop(224, self.random, [1, .875, .75, .66]),
                                                             GroupRandomHorizontalFlip(self.random, is_flow=True)])
        transform_rgb = torchvision.transforms.Compose([
            train_augmentation,
            Stack(),
            ToTorchFormatTensor(),
            normalize,
        ])
        self.transform = transform_rgb
        meta_dir = os.path.join(self.data_dir, dset_name)

        if (word2idx_path is None) or (not os.path.exists(word2idx_path)):
            logging.info('[WARNING] word2idx_path load failed, use default path.')
            word2idx_path = os.path.join(data_dir, 'vocab_feature', 'word2idx.json')
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.video_feature_dir = os.path.join('/feature_kit/video_feature_train')
        # self.frame_to_second = self._load_duration()

        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len  # video max length
        self.max_t_len = max_t_len  # sentence max length
        self.max_cot_len = max_cot_len
        self.max_n_len = max_n_len

        # data entries
        self.data = None
        self.set_data_mode(mode=mode)
        # self.get_txt()
        # self.fix_missing()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items = self.convert_example_to_features(self.data[index])
        return items

    def set_data_mode(self, mode):
        """mode: `train` or `val` or `test`"""
        assert mode in ['train', 'val', 'test']
        logging.info("Mode {}".format(mode))
        data_path = os.path.join(self.data_dir, self.dset_name, "{}{}".format(mode, self.json_path))
        if mode == 'val':
            data_path = "./captions/val.json"
            # during zero - causal relation test modified when 0set
        '''
        load data
        '''
        self._load_data(data_path)

    def _load_data(self, data_path):
        logging.info("Loading data from {}".format(data_path))
        raw_data_list = load_json(data_path)
        data = []

        for raw_data in raw_data_list:
            for video_id, video_data in raw_data.items():
                original_str = '0' * (len(video_data['sentences']) - 1)
                random_str = ''.join(random.choice(['0', '1']) for _ in range(len(original_str)))
                random_str = [random_str]
                if self.mode == 'train':
                    video_info = {
                        'name': video_id,
                        'duration': video_data.get('duration', None),
                        'timestamps': video_data.get('timestamps', None),
                        'sentences': video_data.get('sentences', None),
                        'relation': video_data.get('relation', random_str),
                        'cot': video_data.get('cot', None),
                        'existence': video_data.get('Existence', None)
                    }
                else:
                    video_info = {
                        'name': video_id,
                        'duration': video_data.get('duration', None),
                        'timestamps': video_data.get('timestamps', None),
                        'sentences': video_data.get('sentences', None),
                        'relation': video_data.get('relation', random_str),
                        'cot': video_data.get('cot', None),
                        'existence': video_data.get('Existence', None),
                        'all_relation': video_data.get('all_relation', None)
                    }
                data.append(video_info)
        self.data = data

        logging.info("Loading complete! {} examples in total.".format(len(self)))

    def get_frames(self, example):
        """filter our videos with no feature file"""
        captions = example['sentences']
        relation = example['relation']
        cot = example['cot']
        existence = example['existence']
        if self.mode == 'val':
            all_relation = example['all_relation']
            return captions, relation, cot, existence, all_relation
        else:
            return captions, relation, cot, existence

    def random_delete_word(self, sentence):
        words = sentence.split()

        if len(words) > self.del_words:
            indices_to_delete = random.sample(range(len(words)), self.del_words)
            indices_to_delete.sort(reverse=True)

            for index in indices_to_delete:
                del words[index]

        return ' '.join(words)

    def mask_single_feature(self, feature, index):
        feature[index]['video_feature'] = np.zeros_like(feature[index]['video_feature'])
        feature[index]['video_mask'] = [1] * len(feature[index]['video_mask'])
        feature[index]['video_tokens'] = [self.CLS_TOKEN] + [self.VID_TOKEN] * (self.max_v_len - 2) + [self.SEP_TOKEN]
        feature[index]['cot_text_mask'] = [0] * len(feature[index]['cot_text_mask'])

        return feature

    def convert_example_to_features(self, example):

        global _idx, random_index_list, index_pair, all_relation
        example_name = example["name"]
        if self.mode == 'val':
            captions, relation, cot, existence, all_relation = self.get_frames(example)
        else:
            captions, relation, cot, existence = self.get_frames(example)
        if captions is None:
            print(example_name)
        # load res200 pretrain
        if os.path.exists(os.path.join(self.video_feature_dir, "{}_resnet.npy".format(example_name[2:]))):
            feat_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(example_name[2:]))
        else:
            feat_path_resnet = os.path.join("./feature_kit/video_feature_val",
                                            "{}_resnet.npy".format(example_name[2:]))

        video_feature = np.load(feat_path_resnet)

        num_sen = len(captions)
        single_video_features = []

        all_patialy_masked_feature = []

        # every event to extract video feature from whole feature
        for clip_idx in range(num_sen):
            if len(captions) != len(example['timestamps']):
                print("annotation error in caption or timestamps")
            if clip_idx < num_sen - 1:
                if cot is not None:
                    if existence is None or len(cot) != len(existence):
                        print("annotation error in cot or existence")
                        print(example_name)

                    cur_data = self.clip_sentence_to_feature(example, example_name, captions[clip_idx],
                                                             video_feature,
                                                             clip_idx, cot[clip_idx], existence[clip_idx])
                else:
                    cur_data = self.clip_sentence_to_feature(example, example_name, captions[clip_idx],
                                                             video_feature,
                                                             clip_idx, '', existence[clip_idx])
            else:
                cur_data = self.clip_sentence_to_feature(example, example_name, captions[clip_idx],
                                                         video_feature,
                                                         clip_idx, '', '')
            single_video_features.append(cur_data)

        '''
        pred the last event
        '''
        last_feature = single_video_features[-1]
        _mask_idx = num_sen - 1
        assert _mask_idx < num_sen

        # * visual feature for aux loss
        '''
        f_feat -> whole video feature for encoder
        '''
        # rows_to_zero = np.random.choice(50, 10, replace=False)
        # for single_video_feature in single_video_features:
        #     single_video_feature['video_feature'][rows_to_zero, :] = 0

        _, f_video_all_mask, f_feat, f_video_temporal_tokens = self._construct_entire_video_features(
            single_video_features)

        # mask the final event visual part
        single_video_features[_mask_idx]['video_feature'] = np.zeros_like(
            single_video_features[_mask_idx]['video_feature'])
        single_video_features[_mask_idx]['video_mask'] = [1] * len(single_video_features[_mask_idx]['video_mask'])
        single_video_features[_mask_idx]['video_tokens'] = [self.CLS_TOKEN] + [self.VID_TOKEN] * (
                self.max_v_len - 2) + [self.SEP_TOKEN]
        random_index_list = None
        # 1. create all masked visual pairs and chain of thoughts
        if self.mode == 'train':
            index_pair = []
            random_index_list = random.sample(list(range(num_sen-1)),
                                              max(1, (num_sen + self.multi_chains_b) // self.multi_chains_k))
            for index in range(0, num_sen - 1):
                if index not in random_index_list:
                    partially_masked_feature = copy.deepcopy(single_video_features)
                    partially_masked_feature = self.mask_single_feature(partially_masked_feature, index)
                else:
                    # aux_index = random.choice([x for x in random_index_list if x != index])
                    indexes = list(range(0, num_sen-1))
                    filtered_list = [item for item in indexes if item != index]
                    aux_index = random.choice(filtered_list)
                    index_pair.append(aux_index)
                    partially_masked_feature = copy.deepcopy(single_video_features)
                    partially_masked_feature = self.mask_single_feature(partially_masked_feature, index)
                    partially_masked_feature = self.mask_single_feature(partially_masked_feature, aux_index)

                all_patialy_masked_feature.append(partially_masked_feature)
        else:
            # during inference
            for index in range(0, num_sen - 1):
                partially_masked_feature = copy.deepcopy(single_video_features)
                partially_masked_feature = self.mask_single_feature(partially_masked_feature, index)
                all_patialy_masked_feature.append(partially_masked_feature)
        # all visual feature, text feature is only for decoder
        '''
        feat -> whole video feature for decoder (after mask the last event)
        all_feat -> whole video feature for decoder (after mask the last event and a certain event)
        '''
        _, video_all_mask, feat, video_temporal_tokens = self._construct_entire_video_features(single_video_features)
        all_video_all_mask, all_feat, all_video_temporal_tokens = [], [], []

        # 2. create all masked feature
        for masked_feature in all_patialy_masked_feature:
            _, video_all_mask_masked, feat_masked, video_temporal_tokens_masked = self._construct_entire_video_features(
                masked_feature)
            all_video_all_mask.append(video_all_mask_masked)
            all_feat.append(feat_masked)
            all_video_temporal_tokens.append(video_temporal_tokens_masked)

        input_labels_list = [[] for _ in range(self.K)]
        token_type_ids_list = [[] for _ in range(self.K)]
        input_mask_list = [[] for _ in range(self.K)]
        input_ids_list = [[] for _ in range(self.K)]

        full_masked_input_ids_list = []
        full_masked_input_mask_list = []

        def _fill_data(_idx):
            video_tokens = single_video_features[_idx]['video_tokens']
            video_mask = single_video_features[_idx]['video_mask']
            text_tokens = single_video_features[_idx]['text_tokens']
            text_mask = single_video_features[_idx]['text_mask']
            cot_tokens = single_video_features[_idx]['cot_text_tokens']
            cot_mask = single_video_features[_idx]['cot_text_mask']
            return video_tokens, video_mask, text_tokens, text_mask, cot_tokens, cot_mask

        def _fill_masked_data(feature, _idx, k, flag):
            video_mask = feature[_idx]['video_mask']
            text_mask = feature[_idx]['text_mask']
            video_tokens = feature[_idx]['video_tokens']
            text_tokens = feature[_idx]['text_tokens']
            cot_tokens = feature[_idx]['cot_text_tokens']
            cot_mask = feature[_idx]['cot_text_mask']
            existence_tokens = feature[_idx]['existence_text_tokens']
            existence_mask = feature[_idx]['existence_text_mask']
            if _idx == k and flag == 0:
                if self.use_existence:
                    text_tokens = existence_tokens
                    text_mask = existence_mask
                else:
                    text_tokens = ['[PAD]' if token not in ['[PAD]', '[BOS]', '[EOS]'] else token for token in
                                   text_tokens]
                cot_tokens = ['[PAD]' if token not in ['[PAD]', '[BOS]', '[EOS]'] else token for token in cot_tokens]

            elif self.mode == 'train':
                if k in random_index_list and _idx == self.aux_idx_text and flag == 0:
                    if self.use_existence:
                        text_tokens = existence_tokens
                        text_mask = existence_mask
                    else:
                        text_tokens = ['[PAD]' if token not in ['[PAD]', '[BOS]', '[EOS]'] else token for token in
                                       text_tokens]
                    cot_tokens = ['[PAD]' if token not in ['[PAD]', '[BOS]', '[EOS]'] else token for token in cot_tokens]

            return video_tokens, video_mask, text_tokens, text_mask, cot_tokens, cot_mask

        '''
        input_ids_list length = num of decoders
        input_ids_list[i] length = num of events(max events length)
        every input_ids_list[i][j] length = 50 video feature tokens + 30 sentences tokens
        '''
        for _idx in range(self.max_n_len):

            video_tokens, video_mask, text_tokens, text_mask, cot_tokens, cot_mask = _fill_data(
                _idx if _idx < num_sen else 0)

            # * prepare for dec1
            if self.max_cot_len != 0:
                _input_tokens = video_tokens + text_tokens + cot_tokens
                _input_mask = video_mask + text_mask + cot_mask
            else:
                _input_tokens = video_tokens + text_tokens
                _input_mask = video_mask + text_mask
            _input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in _input_tokens]

            _token_type_ids = [self.VID_TYPE] * self.max_v_len + [self.TEX_TYPE] * (self.max_t_len + self.max_cot_len)

            input_ids_list[0].append(np.array(_input_ids).astype(np.int64))
            token_type_ids_list[0].append(np.array(_token_type_ids).astype(np.int64))
            input_mask_list[0].append(np.array(_input_mask).astype(np.float32))
            if self.max_cot_len != 0:
                label_list = text_mask + cot_mask
            else:
                label_list = text_mask
            # * shifted right, `-1` is ignored when calculating CrossEntropy Loss
            _input_labels = \
                [self.IGNORE] * len(video_tokens) + \
                [self.IGNORE if m == 0 else tid for tid, m in zip(_input_ids[-len(label_list):], label_list)][1:] + \
                [self.IGNORE]

            input_labels_list[0].append(np.array(_input_labels).astype(np.int64))

            # * prepare for dec2+
            for k_idx in range(1, self.K):
                sen_tokens = [self.SEN_TOKEN] * self.max_n_len
                sen_mask = [1] * num_sen + [0] * (self.max_n_len - num_sen)
                if self.max_cot_len != 0:
                    _input_tokens = video_tokens + sen_tokens + text_tokens + cot_tokens
                    _input_mask = video_mask + sen_mask + text_mask + cot_mask
                else:
                    _input_tokens = video_tokens + sen_tokens + text_tokens
                    _input_mask = video_mask + sen_mask + text_mask
                _input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in _input_tokens]

                _token_type_ids = [self.VID_TYPE] * self.max_v_len + [self.SEN_TYPE] * self.max_n_len + [
                    self.TEX_TYPE] * (self.max_t_len + self.max_cot_len)

                input_ids_list[k_idx].append(np.array(_input_ids).astype(np.int64))
                token_type_ids_list[k_idx].append(np.array(_token_type_ids).astype(np.int64))
                input_mask_list[k_idx].append(np.array(_input_mask).astype(np.float32))
                if self.max_cot_len != 0:
                    label_list = text_mask + cot_mask
                else:
                    label_list = text_mask
                _input_labels = \
                    [self.IGNORE] * len(video_tokens) + \
                    [self.IGNORE] * len(sen_tokens) + \
                    [self.IGNORE if m == 0 else tid for tid, m in zip(_input_ids[-len(label_list):], label_list)][1:] + \
                    [self.IGNORE]

                input_labels_list[k_idx].append(np.array(_input_labels).astype(np.int64))

        # 3. create all mask pairs and existing or all ['PAD'] masked event caption
        k, m = 0, 0
        for maske_feature in all_patialy_masked_feature:
            masked_input_mask_list = [[] for _ in range(self.K)]
            masked_input_ids_list = [[] for _ in range(self.K)]
            if self.mode == 'train' and k in random_index_list:
                self.aux_idx_text = index_pair[m]
                m = m + 1
            # for all events
            for _idx in range(self.max_n_len):
                if _idx < num_sen:
                    flag = 0
                else:
                    flag = 1
                video_tokens, video_mask, text_tokens, text_mask, cot_tokens, cot_mask \
                    = _fill_masked_data(maske_feature, _idx if _idx < num_sen else 0, k, flag)
                if self.max_cot_len != 0:
                    _input_tokens = video_tokens + text_tokens + cot_tokens
                    _input_mask = video_mask + text_mask + cot_mask
                else:
                    _input_tokens = video_tokens + text_tokens
                    _input_mask = video_mask + text_mask
                _input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in _input_tokens]
                masked_input_ids_list[0].append(np.array(_input_ids).astype(np.int64))

                masked_input_mask_list[0].append(np.array(_input_mask).astype(np.float32))

                # * prepare for dec2+
                for k_idx in range(1, self.K):
                    sen_tokens = [self.SEN_TOKEN] * self.max_n_len
                    sen_mask = [1] * (num_sen) + [0] * (self.max_n_len - num_sen)
                    if self.max_cot_len != 0:
                        _input_tokens = video_tokens + sen_tokens + text_tokens + cot_tokens
                        _input_mask = video_mask + sen_mask + text_mask + cot_mask
                    else:
                        _input_tokens = video_tokens + sen_tokens + text_tokens
                        _input_mask = video_mask + sen_mask + text_mask
                    _input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in _input_tokens]

                    masked_input_ids_list[k_idx].append(np.array(_input_ids).astype(np.int64))
                    masked_input_mask_list[k_idx].append(np.array(_input_mask).astype(np.float32))

            full_masked_input_ids_list.append(masked_input_ids_list)
            full_masked_input_mask_list.append(masked_input_mask_list)
            k = k + 1
        # * ignore all padded sentence
        for k_idx in range(self.K):
            for n_idx in range(num_sen, self.max_n_len):
                input_labels_list[k_idx][n_idx][:] = self.IGNORE
        if random_index_list is None:
            random_index_list = [0]
        if self.mode == 'train':
            if index_pair is None:
                index_pair = [0]
        if self.mode == 'train':
            data = dict(
                random_index_list=random_index_list,
                aux_idx_text=index_pair,
                example_name=example_name,
                num_sen=num_sen,
                # * encoder input
                encoder_input=dict(
                    video_features=feat.astype(np.float32),
                    temporal_tokens=np.array(video_temporal_tokens).astype(np.int64),
                    video_mask=np.array(video_all_mask).astype(np.float32),
                    all_masked_video_features=[feat_single.astype(np.float32) for feat_single in all_feat]
                    ,
                    all_masked_temporal_tokens=[np.array(video_temporal_tokens_single).astype(np.int64)
                                                for video_temporal_tokens_single in
                                                all_video_temporal_tokens],
                    all_masked_video_mask=[np.array(single_video_all_mask).astype(np.float32) for
                                           single_video_all_mask
                                           in all_video_all_mask],
                ),
                unmasked_encoder_input=dict(
                    video_features=f_feat.astype(np.float32),
                    temporal_tokens=np.array(f_video_temporal_tokens).astype(np.int64),
                    video_mask=np.array(f_video_all_mask).astype(np.float32),
                ),
                # * decoder inputs visual+textual
                decoder_input=dict(
                    input_ids=input_ids_list,
                    input_mask=input_mask_list,
                    token_type_ids=token_type_ids_list,
                    all_input_ids=full_masked_input_ids_list,
                    all_input_mask=full_masked_input_mask_list,
                    relation=relation,
                ),
                # * gts for cascaded decoder
                gt=input_labels_list
            )
        else:
            data = dict(
                example_name=example_name,
                num_sen=num_sen,
                # * encoder input
                encoder_input=dict(
                    video_features=feat.astype(np.float32),
                    temporal_tokens=np.array(video_temporal_tokens).astype(np.int64),
                    video_mask=np.array(video_all_mask).astype(np.float32),
                    all_masked_video_features=[feat_single.astype(np.float32) for feat_single in all_feat]
                    ,
                    all_masked_temporal_tokens=[np.array(video_temporal_tokens_single).astype(np.int64)
                                                for video_temporal_tokens_single in
                                                all_video_temporal_tokens],
                    all_masked_video_mask=[np.array(single_video_all_mask).astype(np.float32) for
                                           single_video_all_mask
                                           in all_video_all_mask],
                ),
                unmasked_encoder_input=dict(
                    video_features=f_feat.astype(np.float32),
                    temporal_tokens=np.array(f_video_temporal_tokens).astype(np.int64),
                    video_mask=np.array(f_video_all_mask).astype(np.float32),
                ),
                # * decoder inputs visual+textual
                decoder_input=dict(
                    input_ids=input_ids_list,
                    input_mask=input_mask_list,
                    token_type_ids=token_type_ids_list,
                    all_input_ids=full_masked_input_ids_list,
                    all_input_mask=full_masked_input_mask_list,
                    relation=relation,
                    all_relation=all_relation
                ),
                # * gts for cascaded decoder
                gt=input_labels_list
            )
        return data

    def _construct_entire_video_features(self, single_video_features):
        global clip_feat
        video_tokens = []
        video_mask = []
        feats = []
        video_temporal_tokens = []
        for idx, clip_feat in enumerate(single_video_features):
            video_tokens += clip_feat['video_tokens'].copy()
            video_mask += clip_feat['video_mask'].copy()
            feats.append(clip_feat['video_feature'].copy())

        # * pad videos to max_n_len
        if len(single_video_features) < self.max_n_len:
            pad_v_n = self.max_n_len - len(single_video_features)

            video_tokens += [self.PAD_TOKEN] * self.max_v_len * pad_v_n
            video_mask += [0] * self.max_v_len * pad_v_n

            _feat = [np.zeros_like(single_video_features[0]['video_feature'])] * pad_v_n

            feats.extend(_feat)

        for idx in range(self.max_n_len):
            video_temporal_tokens += [idx] * len(clip_feat['video_tokens'])

        feat = np.concatenate(feats, axis=0)
        return video_tokens, video_mask, feat, video_temporal_tokens

    def clip_sentence_to_feature(self, example, name, event, video_feature, clip_idx, cot, existence):
        """
            get video clip and sentence feature and tokens/masks
        """

        video_name = name
        sentence = event
        frm2sec = example['duration']
        timestamp = example['timestamps']
        # video + text tokens

        feat, video_tokens, video_mask = self._load_indexed_video_feature(video_feature, timestamp[clip_idx], frm2sec)
        '''
        video get feature so far only frames tensor
        '''
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)
        cot_text_tokens, cot_text_mask = self._tokenize_cot_sentence(cot)
        existence_text_tokens, existence_text_mask = self._tokenize_pad_sentence(existence)

        data = dict(
            video_tokens=video_tokens,
            text_tokens=text_tokens,
            cot_text_tokens=cot_text_tokens,
            cot_text_mask=cot_text_mask,
            existence_text_tokens=existence_text_tokens,
            existence_text_mask=existence_text_mask,
            video_mask=video_mask,
            text_mask=text_mask,
            video_feature=feat.astype(np.float32)
        )

        return data

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, frm2sec):
        """convert wall time st_ed to feature index st_ed"""
        st = int(math.floor(timestamp[0] / frm2sec * feat_len))
        ed = int(math.ceil(timestamp[1] / frm2sec * feat_len))
        ed = min(ed, feat_len - 1)
        st = min(st, ed - 1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed

    def _load_indexed_video_feature(self, raw_feat, timestamp, frm2sec):
        """
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat: self.max_v_len
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """
        max_v_l = self.max_v_len - 2
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1
        feat = np.zeros((self.max_v_len, raw_feat.shape[1]))

        if indexed_feat_len > max_v_l:
            # * linear (uniform) sample
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat[1:max_v_l + 1] = raw_feat[downsamlp_indices]

            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            mask = [1] * (max_v_l + 2)
        else:
            valid_l = ed - st + 1
            if raw_feat.shape[0] == 0:
                print("no feat")
            feat[1:valid_l + 1] = raw_feat[st:ed + 1]
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + \
                           [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)
            mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)

        return feat, video_tokens, mask

    def _tokenize_pad_sentence(self, sentence):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_t_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def _tokenize_cot_sentence(self, sentence):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_cot_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


def prepare_batch_inputs(batch, device, non_blocking=False):
    def _recursive_to_device(v):
        if isinstance(v[0], list):
            return [_recursive_to_device(_v) for _v in v]
        elif isinstance(v[0], torch.Tensor):
            return [_v.to(device, non_blocking=non_blocking) for _v in v]
        else:
            return v

    batch_inputs = dict()

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, list):
            batch_inputs[k] = _recursive_to_device(v)
        elif isinstance(v, dict):
            batch_inputs[k] = prepare_batch_inputs(v, device, non_blocking)

    return batch_inputs


def sentences_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    # batch_meta = [[{"name": e['name'],
    #                 } for e in _batch[1]] for _batch in batch]  # change key
    padded_batch = default_collate([e for e in batch])
    # padded_batch = my_collate(batch)
    return padded_batch


def cal_performance(pred, gold):
    gold = gold[:, -pred.shape[1]:]
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(MECDDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum()

    return n_correct
