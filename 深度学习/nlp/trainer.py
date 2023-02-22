import torch.nn as nn
import torch
from sklearn.model_selection import StratifiedKFold
import transformers
from tqdm import tqdm

transformers.logging.set_verbosity_error()


class Trainer(object):
    def __init__(self, model: nn.Module, epochs=20, lr=1e-5, weight_decay=0, show_batch=50, use_cuda=True):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.show_batch = show_batch
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self, dataset_train, dataset_evel=None):
        self.model.train()
        for epoch in range(self.epochs):
            for idx, data in tqdm(enumerate(dataset_train), total=len(dataset_train)):
                data = data.to(self.device)
                loss = self.model(**data)["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if idx % self.show_batch == 0:
                    print('Epoch [{}/{}],batch:{} Loss: {:.4f}'.format(epoch + 1, epoch, idx, loss.item()))
            with torch.no_grad():  # 评估时禁止计算梯度
                self.evaluation(dataset_evel, epoch)

    def score(self, data):
        start_logits = self.model(**data)["start_logits"]
        end_logits = self.model(**data)["end_logits"]
        start_idx = torch.argmax(start_logits, dim=-1)
        end_idx = torch.argmax(end_logits, dim=-1)
        answer_start = data["answer_start"]
        answer_end = data["answer_end"]
        start = start_idx == answer_start
        end = end_idx == answer_end
        value = torch.logical_and(start, end)
        acc = torch.sum(value).item() / len(value)
        return acc

    def evaluation(self, dataset_evel, epoch):
        print("evaluation.....")
        self.model.eval()
        acc_list = []
        for idx, data in tqdm(enumerate(dataset_evel), total=len(dataset_evel)):
            data = data.to(self.device)
            acc = self.score(data)
            acc_list.append(acc)
        acc = sum(acc_list) / len(acc_list) * 100
        print('Epoch [{}/{}], Acc: {:.4f}%'.format(epoch + 1, epoch, acc))
