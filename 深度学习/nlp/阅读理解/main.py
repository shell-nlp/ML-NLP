from transformers import BertModel, BertTokenizer
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
        context = self.data_df.iloc[item]["context"]
        question = self.data_df.iloc[item]["question"]
        answer_start = self.data_df.iloc[item]["answer_start"]
        answer_end = self.data_df.iloc[item]["answer_end"]
        return context, question, answer_start, answer_end


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def collate_fn(batch):
    context = [item[0] for item in batch]
    question = [item[1] for item in batch]
    answer_start = [item[2] for item in batch]
    answer_end = [item[3] for item in batch]
    output = tokenizer(question, context, truncation=True, padding=True, return_tensors="pt")
    output["answer_start"] = torch.tensor(answer_start)
    output["answer_end"] = torch.tensor(answer_end)
    return output


class MyBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')

    def forward(self, input_ids, attention_mask, token_type_ids, answer_start, answer_end, labels=None):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        print(output)


if __name__ == '__main__':

    dataset = MyData()

    ds = DataLoader(dataset=dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    model = MyBert()
    print(model)
    for data in ds:
        output = model(**data)
        break
