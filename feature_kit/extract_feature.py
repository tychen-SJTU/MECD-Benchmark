import os
import sys
sys.path.append(os.getcwd())
from feature_kit.pyActionRec.action_classifier import ActionClassifier
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='./feature_kit/video_feature_train')
parser.add_argument("--vid_path", type=str, default='your_path/Activitynet/v1-3')
parser.add_argument("--video_file", type=str, default='your_path/my_list_train.txt')
parser.add_argument("--gpu", type=int, default=1)

args = parser.parse_args()

GPU=args.gpu

all_vid_list = list(open(args.video_file, 'r').readlines())
all_vid_list = [vid.strip() for vid in all_vid_list]

models = []

models = [('./feature_kit/models/resnet200_anet_2016_deploy.prototxt',
           './feature_kit/models/resnet200_anet_2016.caffemodel',
           1.0, 0, True, 224)]

cls = ActionClassifier(models, dev_id=GPU)

process_list = {}
counter = 0
for vid in tqdm(all_vid_list):
    if os.path.isfile(os.path.join(args.data_path, vid+"_resnet.npy")):
        counter += 1
        process_list[vid] = counter
    elif vid not in process_list:
        vid_path_mp4 = os.path.join(args.vid_path, vid + '.mp4')
        vid_path_mkv = os.path.join(args.vid_path, vid + '.mkv')
        if os.path.exists(vid_path_mp4):
            vid_path = vid_path_mp4
        else:
            vid_path = vid_path_mkv
        data_path = os.path.join('./feature_kit/video_feature_train')
        rst = cls.classify(vid_path, data_path)
        if rst != -1:
            print('Processed video: ', vid)
        counter += 1
        process_list[vid] = counter
