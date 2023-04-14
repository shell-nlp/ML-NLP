from transformers import get_linear_schedule_with_warmup
import os
import torch.nn as nn
import torch
from sklearn.model_selection import StratifiedKFold, KFold
import transformers
from tqdm import tqdm
transformers.logging.set_verbosity_error()


class TrainConfig:
    pass


class Train(object):
    def __init__(self, model: nn.Module, epochs=20, lr=1e-5, weight_decay=0,
                 show_batch=50, use_cuda=True, compute_metrics=None, is_better=True, FGM=False, save_path="./save/"):
        self.model = model
        self.is_better = is_better  # 用于判断指标是越大越好还是越小越好
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epochs = epochs
        self.cur_epoch = 0
        self.cur_batch = 0
        self.best_score = -1e8
        self.best_epoch = -1
        self.lr = lr
        self.show_batch = show_batch
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.compute_metrics = compute_metrics
        self.save_path = save_path
        self.FGM = FGM
        if self.FGM:
            import sys
            import os

            sys.path.append(os.path.dirname(__file__))
            print("FGM已开启...")
            from adversarial_training import FGM
            self.fgm = FGM(self.model)

    def train(self, dataset_train, dataset_eval=None, num_warmup=1000):
        # 学习率衰减
        if 0 < num_warmup <= 1:
            num_warmup_steps = int(self.epochs*len(dataset_train)*num_warmup)
        elif num_warmup > 1:
            num_warmup_steps = num_warmup
        else:
            Exception("num_warmup必须大于0")

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.epochs*len(dataset_train))
        print("\n"*15)
        print("开始训练.....")
        bar = tqdm(total=len(dataset_train)*self.epochs, dynamic_ncols=True)
        eval_score_str = ""
        for _ in range(self.epochs):
            self.model.train()
            self.cur_epoch = self.cur_epoch+1
            for _, batch in enumerate(dataset_train):
                self.cur_batch = self.cur_batch + 1
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]
                loss.backward()
                # 加入FGM对抗训练
                if self.FGM:
                    self.fgm.attack()  # 在embedding上添加对抗扰动
                    loss_adv = self.model(**batch)["loss"]
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    self.fgm.restore()  # 恢复embedding参数
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

                self.cur_loss = loss
                if self.cur_batch % self.show_batch == 0:
                    self.updata_tqdm_des(
                        bar=bar, loss=loss, eval_score_str=eval_score_str)

            # 评估
            if dataset_eval:
                eval_score_str = self.evaluation(dataset_eval)
                self.updata_tqdm_des(bar=bar, loss=loss,
                                     eval_score_str=eval_score_str)

    @torch.no_grad()
    def evaluation(self, dataset_eval):
        self.model.eval()
        score_list = []
        for batch in tqdm(dataset_eval, total=len(dataset_eval), desc='Validation', leave=False, dynamic_ncols=True):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            score = self.compute_metrics(self.model, batch)
            score_list.append(score)
        score = sum(score_list) / len(score_list)
        # 用于判断指标是越大越好还是越小越好
        if not self.is_better:
            score_ = -score
        else:
            score_ = score
        if score_ > self.best_score:
            self.best_score = abs(score_)
            self.best_epoch = self.cur_epoch
            # 当保存地址 非None时，进行保存
            if self.save_path is not None:
                self.save()
        eval_epoch = self.cur_epoch

        eval_score_str = f' | Eval epoch:{eval_epoch} cur_score: {score:.4f} Best_score: {self.best_score:.4f} in epoch {self.best_epoch}'
        return eval_score_str

    def save(self):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model, save_path +
                   f"score_{self.best_score:.4f} in epoch {self.cur_epoch}.pt")

    def load(self):
        self.model = torch.load(self.save_path)

    def updata_tqdm_des(self, bar, loss, eval_score_str):
        print_str = 'Training  Epoch [{}/{}] Loss: {:.4f}'.format(
            self.epochs, self.cur_epoch, loss.item()) + eval_score_str
        bar.set_description(print_str)
        bar.update(self.show_batch)  # 更新进度
        bar.refresh()  # 立即显示进度条更新结果
