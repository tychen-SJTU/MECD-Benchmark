import json
from tqdm import tqdm
import sng_parser

TRAIN_CAPTION_PATH = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/Video-LLaVA-7B_caption/train_captions.json'
TRAIN_SG_PATH = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/Video-LLaVA-7B_caption/train_sg.json'

TEST_CAPTION_PATH = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/Video-LLaVA-7B_caption/test_captions.json'
TEST_SG_PATH = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/Video-LLaVA-7B_caption/test_sg.json'

VAL_CAPTION_PATH = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/Video-LLaVA-7B_caption/val_captions.json'
VAL_SG_PATH = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/Video-LLaVA-7B_caption/val_sg.json'



def refine_caption_file(input_caption_path, output_caption_path):
    sample_list = []
    with open(input_caption_path, 'r') as f:
        lines = f.readlines()
        for sample in lines:
            sample = sample.strip() + ',\n'
            sample_list.append(sample)
    
    with open(output_caption_path, 'w') as f:
        f.write('[')
        for sample in sample_list:
            f.write(sample)
        f.write(']')
        
    return


def caption_to_sg(input_caption_path, output_sg_path):
    with open(input_caption_path, 'r') as f:
        caption_list = json.load(f)
    
    caption_triplets = []
    caption_object_vocabs, caption_relation_vocabs = set(), set()
    
    for sample in tqdm(caption_list):
        triplets = set()
        caption = sample['caption']
        # text parsing
        cap_graph = sng_parser.parse(caption.lower())
        for e in cap_graph['entities']:
            caption_object_vocabs.add(e['lemma_head'])
        for r in cap_graph['relations']:
            caption_relation_vocabs.add(r['lemma_relation'])
            triplet = (cap_graph['entities'][r['subject']]['lemma_head'], 
                       r['lemma_relation'], 
                       cap_graph['entities'][r['object']]['lemma_head'])
            triplets.add(triplet)
        
        sample['triplets'] = list(triplets)
        sample['objects'] = list(caption_object_vocabs)
        sample['relations'] = list(caption_relation_vocabs)
        
        caption_triplets.append(sample)
    
    with open(output_sg_path, 'w') as f:
        json.dump(caption_triplets, f)
    
    return


if __name__ == '__main__':
    caption_to_sg(TRAIN_CAPTION_PATH, TRAIN_SG_PATH)
    caption_to_sg(TEST_CAPTION_PATH, TEST_SG_PATH)
    caption_to_sg(VAL_CAPTION_PATH, VAL_SG_PATH)
