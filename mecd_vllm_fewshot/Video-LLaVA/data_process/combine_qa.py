import json
from tqdm import tqdm


train_q_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/train_q.json'
train_a_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/train_a.json'
train_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/train.json'


val_q_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/val_q.json'
val_a_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/val_a.json'
val_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/val.json'

test_q_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/test_q.json'
test_a_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/test_a.json'
test_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/test.json'

test_q_100_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/test_q_100.json'
test_a_100_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/test_a_100.json'
test_100_path = 'eval/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/annotations/test_100.json'


def combine_qa_set(q_path, a_path, qa_output_path):
    with open(q_path, 'r') as f:
        train_q = json.load(f)
    with open(a_path, 'r') as f:
        train_a = json.load(f)
    
    for id in range(len(train_q)):
        assert train_q[id]['question_id'] == train_a[id]['question_id']
        train_q[id]['answer'] = train_a[id]['answer']
    
    with open(qa_output_path, 'w') as f:
        json.dump(train_q, f)




if __name__ == '__main__':
    # # combine train set
    # combine_qa_set(train_q_path, train_a_path, train_path)
    # # combine test set
    # combine_qa_set(test_q_path, test_a_path, test_path)
    # # combine val set
    # combine_qa_set(val_q_path, val_a_path, val_path)
    # combine val 100 set
    combine_qa_set(test_q_100_path, test_a_100_path, test_100_path)
        