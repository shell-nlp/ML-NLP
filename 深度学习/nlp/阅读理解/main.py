from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor
from transformers.data.processors.squad import SquadExample
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def get_squad_dataset(path):
    # Load and process the dataset
    # processor = SquadV2Processor()
    # path = "./data/cmrc2018_public/train.json"
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    examples = []
    for data in dataset["data"]:
        title = data["title"]
        for para in data["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                qas_id = qa["id"]
                question = qa["question"]
                answers = qa["answers"]
                answer_text = answers[0]["text"]
                start_position_character = answers[0]["answer_start"]

                # Create a SquadExample for the current question
                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question,
                    context_text=context,
                    start_position_character=start_position_character,
                    answer_text=answer_text,
                    title=title,
                    answers=answers,
                    is_impossible=False,
                )

                examples.append(example)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=512,
        doc_stride=128,
        max_query_length=64,
        is_training=True,
        return_dataset="pt",
        threads=1,
    )
    return dataset


# class MyData(Dataset):
#     def __init__(self, label="train.json"):
#         train_data_path = "./data/cmrc2018_public/" + label
#         df_train = pd.read_json(train_data_path)
#         data_dict = defaultdict(list)
#         for data in df_train["data"]:
#             title = data["title"]
#             context = data["paragraphs"][0]['context']
#             for qa in data['paragraphs'][0]['qas']:
#                 question = qa['question']
#                 answer = qa['answers'][0]['text']
#                 answer_start = qa['answers'][0]["answer_start"]
#                 answer_end = answer_start + len(answer) - 1
#                 # input_data = "[CLS]"+context+"[SEP]"
#                 data_dict["context"].append(context)
#                 data_dict["question"].append(question)
#                 data_dict["answer"].append(answer)
#                 data_dict["answer_start"].append(answer_start)
#                 data_dict["answer_end"].append(answer_end)
#                 data_dict["title"].append(title)
#         self.data_df = pd.DataFrame(data_dict)
#
#     def __len__(self):
#         return len(self.data_df)
#
#     def __getitem__(self, item):
#         context = self.data_df.iloc[item]["context"]
#         question = self.data_df.iloc[item]["question"]
#         answer_start = self.data_df.iloc[item]["answer_start"]
#         answer_end = self.data_df.iloc[item]["answer_end"]
#         return context, question, answer_start, answer_end
#
#
# def collate_fn(batch):
#     context = [item[0] for item in batch]
#     question = [item[1] for item in batch]
#     answer_start = [item[2] for item in batch]
#     answer_end = [item[3] for item in batch]
#     output = tokenizer(question, context, truncation=True, padding=True, return_tensors="pt")
#     output["answer_start"] = torch.tensor(answer_start)
#     output["answer_end"] = torch.tensor(answer_end)
#     return output


class MyBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.qa = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids, answer_start, answer_end, labels=None):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        token_output = output[0]
        logits = self.qa(token_output)
        start_logits, end_logits = torch.split(logits, 1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        # 防止 开始位置和结束位置超出了 模型的输入  使用clamp忽略这些项
        ignored_index = start_logits.size(1)
        start_positions = answer_start.clamp(0, ignored_index)
        end_positions = answer_end.clamp(0, ignored_index)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return {"loss": total_loss, "start_logits": start_logits, "end_logits": end_logits}


import sys

sys.path.append("..")
from trainer import Trainer

if __name__ == '__main__':
    path = "./data/cmrc2018_public/train.json"
    data = get_squad_dataset(path)
    print(data)
    # train_dataset = MyData()
    # dev_dataset = MyData(label="dev.json")
    # batch_size = 8
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # model = MyBert()
    # trainer = Trainer(model, epochs=10, lr=1e-4)
    # trainer.train(dataset_train=train_dataloader, dataset_evel=dev_dataloader)
