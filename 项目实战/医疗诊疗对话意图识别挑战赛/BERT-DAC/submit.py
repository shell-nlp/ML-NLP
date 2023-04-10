import json
import numpy as np
import torch
import torch
from utils import get_time_dif
from transformers import set_seed, BertTokenizerFast
from datasets import Dataset
model = torch.load("./save/score_0.8249 in epoch 3.pt")


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader
    model_names = ["hfl/chinese-macbert-base"]
    check_point = model_names[0]
    test_path = "项目实战/医疗诊疗对话意图识别挑战赛/BERT-DAC/data/process_data/test.txt"
    test = Dataset.from_csv(test_path, sep="\t", names=["context"])
    test_pd = test.to_csv(
        "项目实战/医疗诊疗对话意图识别挑战赛/BERT-DAC/data/process_data/sub.csv", sep="\t", index=False)

    tokenizer = BertTokenizerFast.from_pretrained(check_point)

    def f(batch):
        context = batch["context"]
        output = tokenizer(context)  # 这种方式最好,前提 是最大长度不超过 最大的限制
        try:
            output["labels"] = batch["labels"]
        except:
            return output
        return output
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer)
    test_data = test.map(f, batched=True, remove_columns=[
        'context'], num_proc=6)
    test_loader = DataLoader(test_data, batch_size=64,
                             collate_fn=data_collator, pin_memory=True)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    a = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        logits = output["logits"]
        predict = torch.argmax(logits, dim=-1)
        a = a + predict.cpu().tolist()
    from preprocess import tag2id
    id2tag = {id_: tag for tag, id_ in tag2id.items()}
    print(id2tag)
    res = [id2tag[id_] for id_ in a]
   
    def load_json(path):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    data = load_json(
        "/home/zutcs/other/git/ml-nlp/项目实战/医疗诊疗对话意图识别挑战赛/BERT-DAC/data/test.json")
    i = 0
    c = dict()
    data_ = {}
    for pid, sample in data.items():
        sample_ = []
        for sent in sample:
            sent["dialogue_act"] = res[i]
            i = i+1
            sample_.append(sent)
        data_[pid] = sample_
    with open("/home/zutcs/other/git/ml-nlp/项目实战/医疗诊疗对话意图识别挑战赛/BERT-DAC/data/sun.json","w") as f:
        json.dump(data_,f,ensure_ascii=False,indent= 4)