import os
import sys
import time
import torch
from utils import get_time_dif
from transformers import set_seed, BertTokenizerFast
from datasets import Dataset


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    task = "医疗诊疗对话意图识别挑战赛"
    set_seed(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    model_names = ["hfl/chinese-macbert-base"]
    check_point = model_names[0]
    start_time = time.time()
    print("Loading data...")
    train_path = "项目实战/医疗诊疗对话意图识别挑战赛/BERT-DAC/data/process_data/train.txt"
    dev_path = "项目实战/医疗诊疗对话意图识别挑战赛/BERT-DAC/data/process_data/dev.txt"
    test_path = "项目实战/医疗诊疗对话意图识别挑战赛/BERT-DAC/data/process_data/test.txt"
    train_ = Dataset.from_csv(train_path, sep="\t",
                              names=["context", 'labels'])
    dev = Dataset.from_csv(dev_path, sep="\t", names=["context", 'labels'])
    test = Dataset.from_csv(test_path, sep="\t", names=["context"])

    tokenizer = BertTokenizerFast.from_pretrained(check_point)

    def f(batch):
        context = batch["context"]
        output = tokenizer(context)  # 这种方式最好,前提 是最大长度不超过 最大的限制
        # output = tokenizer(
        #     context, truncation=True, max_length=256, c="max_length")
        try:
            output["labels"] = batch["labels"]
        except:
            return output
        return output
    train_data = train_.map(f, batched=True, remove_columns=[
        'context'], num_proc=1)
    dev_data = dev.map(f, batched=True, remove_columns=[
        'context'], num_proc=6)
    test_data = test.map(f, batched=True, remove_columns=[
        'context'], num_proc=6)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer)
    from torch.utils.data import DataLoader
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              collate_fn=data_collator, pin_memory=True, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size,
                            collate_fn=data_collator, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             collate_fn=data_collator, pin_memory=True)

    from models.berts import Berts
    from util.trainer import Train
    model = Berts(check_point=check_point)

    def compute_metrics(model, batch: dict):
        output = model(**batch)
        logits = output["logits"]
        predict = torch.argmax(logits, dim=-1)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(batch["labels"].cpu(), predict.cpu())
        return acc
    trainer = Train(model=model, epochs=20, lr=1e-4, weight_decay=0,
                    show_batch=50, use_cuda=True,
                    compute_metrics=compute_metrics)
    trainer.train(dataset_train=train_loader,
                  dataset_eval=dev_loader, num_warmup=0.1)
    