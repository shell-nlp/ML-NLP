import time
import torch
# import numpy as np
# from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import get_time_dif
from transformers import set_seed, BertTokenizerFast
from transformers import BertForSequenceClassification
from datasets import Dataset
from train_eval import train
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default="Bert",
                    help='choose a model: Bert, ERNIE')
parser.add_argument('--save_path', type=str, default="",
                    help='the save path of predictions on test set')
args = parser.parse_args()
print(args)


if __name__ == '__main__':

    model_name = args.model  # bert
    task = "医疗诊疗对话意图识别挑战赛"
    print(model_name)
    set_seed(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    check_point = "bert-base-chinese"
    start_time = time.time()
    print("Loading data...")
    train_path = "项目实战/医疗诊疗对话意图识别挑战赛/BERT-DAC/data/process_data/train.txt"
    dev_path = "项目实战/医疗诊疗对话意图识别挑战赛/BERT-DAC/data/process_data/dev.txt"
    test_path = "项目实战/医疗诊疗对话意图识别挑战赛/BERT-DAC/data/process_data/test.txt"
    train_ = Dataset.from_csv(train_path, sep="\t", names=["context", 'labels'])
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
    from sklearn import metrics
    from torch.utils.data import DataLoader
    data_loader = DataLoader(train_data, batch_size=64,
                             collate_fn=data_collator, pin_memory=True)
    dev_loader = DataLoader(dev_data, batch_size=64,
                             collate_fn=data_collator, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=64,
                             collate_fn=data_collator, pin_memory=True)
    
    model = BertForSequenceClassification.from_pretrained(check_point, num_labels=16)


    train(model, data_loader, dev_loader, test_loader, args)
