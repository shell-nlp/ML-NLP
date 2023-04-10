import json
import os


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


data_dir = os.path.join(os.path.dirname(__file__), "data")
train_set = load_json(os.path.join(data_dir, 'train.json'))
dev_set = load_json(os.path.join(data_dir, 'dev.json'))
test_set = load_json(os.path.join(data_dir, 'test.json'))

saved_path = os.path.join(os.path.dirname(__file__), "data/process_data")
os.makedirs(saved_path, exist_ok=True)

tags = [
    'Request-Etiology', 'Request-Precautions', 'Request-Medical_Advice', 'Inform-Etiology', 'Diagnose',
    'Request-Basic_Information', 'Request-Drug_Recommendation', 'Inform-Medical_Advice',
    'Request-Existing_Examination_and_Treatment', 'Inform-Basic_Information', 'Inform-Precautions',
    'Inform-Existing_Examination_and_Treatment', 'Inform-Drug_Recommendation', 'Request-Symptom',
    'Inform-Symptom', 'Other'
]
tag2id = {tag: idx for idx, tag in enumerate(tags)}


def make_tag(path):
    with open(path, 'w', encoding='utf-8') as f:
        for tag in tags:
            f.write(tag + '\n')


def make_data(samples, path, is_train=True):
    out = ''
    # pid: 对话id     sample 对应的  list
    for pid, sample in samples.items():  # sample is list
        # sent: 一个对话中，对应的一个话
        for sent in sample:
            x = sent['speaker'] + '：' + sent['sentence']
            if is_train:
                assert sent['dialogue_act'] in tag2id
                y = tag2id.get(sent['dialogue_act'])
            else:
                y = ""
            out += (x + '\t' + str(y)).strip() + '\n'
        out += "\n"  # TODO 用于识别一个完整的pid
    print(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(out)
    return out


make_tag(os.path.join(saved_path, 'class.txt'))

make_data(train_set, os.path.join(saved_path, 'train.txt'))
make_data(dev_set, os.path.join(saved_path, 'dev.txt'))
make_data(test_set, os.path.join(saved_path, 'test.txt'), is_train=False)
