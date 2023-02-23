from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import torch
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def get_squad_dataset(data_dir, filename,
                      max_seq_length=512,
                      doc_stride=128,
                      max_query_length=64,
                      is_training=True,
                      threads=8):
    processor = SquadV2Processor()
    examples = processor.get_train_examples(data_dir=data_dir, filename=filename)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=is_training,
        return_dataset="pt",
        threads=threads,  # 建议多线程比较快
    )
    return dataset


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
    # train_dataset = get_squad_dataset(data_dir="./data/cmrc2018_public", filename="train.json")
    # dev_dataset = get_squad_dataset(data_dir="./data/cmrc2018_public", filename="dev.json")
    import pickle

    # pickle.dump(train_dataset, open("train.pt", "wb"))
    # pickle.dump(dev_dataset, open("dev.pt", "wb"))
    train_dataset = pickle.load(open("train.pt", "rb"))
    dev_dataset = pickle.load(open("dev.pt", "rb"))
    batch_size = 8
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True)
    model = MyBert()
    trainer = Trainer(model, epochs=10, lr=1e-4)
    trainer.train(dataset_train=train_dataloader, dataset_evel=dev_dataloader)
