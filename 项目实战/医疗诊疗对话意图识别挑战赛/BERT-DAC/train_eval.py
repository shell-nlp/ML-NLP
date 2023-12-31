import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(model, train_iter, dev_iter, test_iter, args):
    num_epochs = 10
    learning_rate = 2e-4
    save_path = ""
    require_improvement = 1000
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=learning_rate,
    #                      warmup=0.05,
    #                      t_total=len(train_iter) * num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, batch in enumerate(train_iter):
            labels = batch["labels"]
            outputs = model(**batch)
            outputs = outputs.logits
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss)
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                p, r, f1, dev_acc, dev_loss = evaluate(model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                # msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                msg = 'Iter: {:>6},  Val P: {:>5.4},  Val R: {:>6.4%},  Val F1: {:>5.4},  Val Acc: {:>6.4%},  Time: {} {}'
                # print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                print(msg.format(total_batch, p, r, f1, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(model, test_iter, args)


def test(config, model, test_iter, args):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    p, r, f1, test_acc, test_loss, test_report, test_confusion, predict_all = evaluate(model, test_iter, test=True)
    # msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    msg = 'Test P: {:>6.4}, Test R: {:>6.4}, Test F: {:>6.4},  Test Acc: {:>6.4%}'
    # print(msg.format(test_loss, test_acc))
    print(msg.format(p, r, f1, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    if args.save_path is not None:
        print("saving predictions to {}.".format(args.save_path))
        # np.save(args.save_path, predict_all)
        np.savez(args.save_path, test_confusion=test_confusion, test_prediction=predict_all)


def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            labels = batch["labels"]
            outputs = model(**batch)
            outputs = outputs.logits
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    p = metrics.precision_score(labels_all, predict_all, average='macro')
    r = metrics.recall_score(labels_all, predict_all, average='macro')
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        from preprocess import tags
        report = metrics.classification_report(labels_all, predict_all, target_names=tags, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        # return acc, loss_total / len(data_iter), report, confusion, predict_all
        return p, r, f1, acc, loss_total / len(data_iter), report, confusion, predict_all
    # return acc, loss_total / len(data_iter)
    return p, r, f1, acc, loss_total / len(data_iter)
