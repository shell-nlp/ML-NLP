import torch.nn as nn
import torch
from sklearn.model_selection import StratifiedKFold, KFold
import transformers
from tqdm import tqdm
transformers.logging.set_verbosity_error()


class Train(object):
    def __init__(self, model: nn.Module, epochs=20, lr=1e-5, weight_decay=0,
                 show_batch=50, use_cuda=True, compute_metrics=None):
        self.model = model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epochs = epochs
        self.cur_epoch = 0
        self.cur_batch = 0
        self.lr = lr
        self.show_batch = show_batch
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.compute_metrics = compute_metrics

    def train(self, dataset_train, dataset_eval=None):
        # total=len(dataset_train), desc='Training')
        bar = tqdm(total=len(dataset_train)*self.epochs, position=0)
        for _ in range(self.epochs):
            self.cur_epoch = self.cur_epoch + 1
            self.model.train()
            for _, batch in enumerate(dataset_train):
                self.cur_batch = self.cur_batch + 1
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.cur_batch % self.show_batch == 0:
                    print(self.cur_batch)
                    print_str = '\nEpoch [{}/{}],batch:{} Loss: {:.4f}'.format(
                        self.epochs, self.cur_epoch, self.cur_batch, loss.item())
                    bar.set_description(print_str)
                    bar.update(self.show_batch)  # 更新进度

                    bar.refresh()  # 立即显示进度条更新结果
                # print(
                #     '\nEpoch [{}/{}],batch:{} Loss: {:.4f}'.format(self.epochs, self.cur_epoch, idx, loss.item()))
            if dataset_eval:
                self.evaluation(dataset_eval)

    @torch.no_grad()
    def evaluation(self, dataset_eval):
        print("evaluation.....")
        self.model.eval()
        score_list = []
        for idx, batch in tqdm(enumerate(dataset_eval), total=len(dataset_eval), desc='Validation'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            score = self.compute_metrics(self.model, batch)
            score_list.append(score)
        score = sum(score_list) / len(score_list) * 100
        print('score: {:.4f} %'.format(score))
