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
        for epoch in range(self.epochs):
            for idx, data in tqdm(enumerate(dataset_train), total=len(dataset_train)):
                data = data.to(self.device)
                loss = self.model(**data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if idx % self.show_batch == 0:
                    print('Epoch [{}/{}],batch:{} Loss: {:.4f}'.format(epoch + 1, epoch, idx, loss.item()))

    def evaluation(self):
        pass
