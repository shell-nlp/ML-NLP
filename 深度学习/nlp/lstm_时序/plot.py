import pandas
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from https://zhuanlan.zhihu.com/p/1892951687645339855
df = pandas.read_csv("airline-passengers.csv", engine="python")
df["lag1"] = df["Passengers"].shift(1)
df["lag2"] = df["Passengers"].shift(2)

df = df.dropna()
# 数据准备
X = df[["lag1", "lag2"]].values

y = df["Passengers"].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
# 数据加载
dataset = TensorDataset(X, y)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        print(x.shape)
        print(lstm_out.shape)
        assert 0
        # 计算注意力权重
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        return out


model = LSTMAttention(input_dim=2, hidden_dim=64, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10000
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.unsqueeze(1)  # 添加时间步维度
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
