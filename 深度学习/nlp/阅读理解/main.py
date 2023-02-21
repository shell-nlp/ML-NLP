from transformers import BertModel, BertTokenizer
from datasets import Dataset, load_dataset
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class MyData(Dataset):
    def __init__(self):
        train_data_path = "./data/cmrc2018_public/train.json"
        df_train = pd.read_json(train_data_path)
        data_dict = defaultdict(list)
        for data in df_train["data"]:
            title = data["title"]
            context = data["paragraphs"][0]['context']
            for qa in data['paragraphs'][0]['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text']
                answer_start = qa['answers'][0]["answer_start"]
                answer_end = answer_start + len(answer) - 1
                # input_data = "[CLS]"+context+"[SEP]"
                data_dict["context"].append(context)
                data_dict["question"].append(question)
                data_dict["answer"].append(answer)
                data_dict["answer_start"].append(answer_start)
                data_dict["answer_end"].append(answer_end)
                data_dict["title"].append(title)
        self.data_df = pd.DataFrame(data_dict)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, item):
        context = self.data_df[item]["context"]
        question = self.data_df[item]["question"]
        answer_start = self.data_df[item]["answer_start"]
        answer_end = self.data_df[item]["answer_end"]
        return context, question, answer_start, answer_end


def fc():
    pass


dataset = MyData()
print(dataset)
# DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=fc)


# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
class MyBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
