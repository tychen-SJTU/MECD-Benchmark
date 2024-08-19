import json

def filter_reasoning_qa(complete_qa_path, reason_qa_path):
    with open(complete_qa_path, 'r') as f:
        complete_qa_dict = json.load(f)
    
    reason_list = []
    for sample in complete_qa_dict:
        reason_flag = False
        question = sample['question']
        if 'why' in question:
            reason_flag = True
        if 'doing' in question:
            reason_flag = True
        if 'relation' in question:
            reason_flag = True
        if 'occupation' in question:
            reason_flag = True
        if 'dangerous' in question:
            reason_flag = True
        if 'safe' in question:
            reason_flag = True
        if 'how many' in question:
            reason_flag = False
        
        if reason_flag:
            reason_list.append(sample)
    
    with open(reason_qa_path, 'w') as f:
         json.dump(reason_list, f)
    return


if __name__ == "__main__":
    complete_qa_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/train.json'
    reason_qa_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/train_reason.json'
    filter_reasoning_qa(complete_qa_path, reason_qa_path)
    
    complete_qa_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/val.json'
    reason_qa_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/val_reason.json'
    filter_reasoning_qa(complete_qa_path, reason_qa_path)
    
    complete_qa_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/test.json'
    reason_qa_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/test_reason.json'
    filter_reasoning_qa(complete_qa_path, reason_qa_path)
    