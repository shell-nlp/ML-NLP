import torch.nn as nn
import torch


class Trainer(object):
    def __init__(self, model: nn.Module, epochs=20, lr=1e-5, weight_decay=0):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self, dataset_train, dataset_evel=None):
        for epoch in range(self.epochs):
            for data in dataset_train:
                loss = self.model(**data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoch, loss.item()))

    def evel(self):
        pass
