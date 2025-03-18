import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

'''
为什么使用 Transformer？

LSTM 存在长时依赖问题，股票数据需要长期历史信息，Transformer 通过 自注意力（Self-Attention） 处理全局依赖关系。

LSTM 训练难度较高，而 Transformer 并行计算效率更高。

股价预测更依赖整体趋势，而不是逐步记忆，自注意力机制能更好捕捉趋势信息。
'''

# 设备选择（CUDA 自动检测）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取数据
df = pd.read_csv('Amazon stock data 2000-2025.csv')
df['date'] = pd.to_datetime(df['date'])

# 选择特征
features = ['open', 'high', 'low', 'close', 'volume']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# 构造时间序列数据（使用前 60 天数据预测第 61 天的 `close` 和 `volume`）
def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # 取过去60天的数据预测第61天
X, y = create_sequences(df[features].values, df[['close', 'volume']].values, seq_length)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 转换为Tensor
X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

# 定义Dataset
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)

# Transformer 预测模型
class TransformerRegression(nn.Module):
    def __init__(self, input_size, d_model, num_layers, nhead, output_size):
        super(TransformerRegression, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)  
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
        return out

# 初始化 Transformer 模型
input_size = len(features)  # 5 个特征
d_model = 128
num_layers = 3
nhead = 4
output_size = 2  # 预测 `close` 和 `volume`
model = TransformerRegression(input_size, d_model, num_layers, nhead, output_size).to(device)

# 训练配置
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# TensorBoard 记录
writer = SummaryWriter()

# 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 计算训练集 R²
    y_train_pred = model(X_train).cpu().detach().numpy()
    y_train_true = y_train.cpu().numpy()
    r2_train = 1 - np.sum((y_train_pred - y_train_true) ** 2) / np.sum((y_train_true - y_train_true.mean(axis=0)) ** 2)

    # 记录 Loss 和 R²
    writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
    writer.add_scalar("Accuracy/train", r2_train, epoch)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Train R²: {r2_train:.4f}")
    scheduler.step(total_loss / len(train_loader))


# 保存最优模型
best_mse = float('inf')  # 记录最优 MSE
model_save_path = "v1_best_transformer_model.pth"

def save_model(model, mse):
    global best_mse
    if mse < best_mse:
        best_mse = mse
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ New best model saved with MSE: {mse:.6f}")
        
# 评估并保存模型
def evaluate(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predictions.extend(output.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 计算 R²
    r2_test = 1 - np.sum((predictions - actuals) ** 2) / np.sum((actuals - actuals.mean(axis=0)) ** 2)
    writer.add_scalar("Accuracy/test", r2_test)
    print(f"Test R²: {r2_test:.4f}")
    
    # 计算 MSE 误差
    mse = np.mean((predictions - actuals) ** 2)
    print(f"Test MSE: {mse:.6f}")

    # 保存最优模型
    save_model(model, mse)

    return predictions, actuals

y_pred, y_true = evaluate(model, test_loader)

# 绘制预测 vs 真实数据对比
# def plot_predictions(y_pred, y_true, title):
#     plt.figure(figsize=(12, 6))
#     plt.plot(y_true[:, 0], label="True Close Price", color='blue')
#     plt.plot(y_pred[:, 0], label="Predicted Close Price", color='red', linestyle='dashed')
#     plt.xlabel("Time")
#     plt.ylabel("Normalized Close Price")
#     plt.title(f"{title} - Close Price Prediction")
#     plt.legend()
#     plt.show()

#     plt.figure(figsize=(12, 6))
#     plt.plot(y_true[:, 1], label="True Volume", color='blue')
#     plt.plot(y_pred[:, 1], label="Predicted Volume", color='red', linestyle='dashed')
#     plt.xlabel("Time")
#     plt.ylabel("Normalized Volume")
#     plt.title(f"{title} - Volume Prediction")
#     plt.legend()
#     plt.show()

# plot_predictions(y_pred, y_true, "Transformer Model")

# 反归一化函数
def inverse_transform(scaler, data, feature_index):
    """ 将归一化数据转换回原始值 """
    data = np.array(data).reshape(-1, 1)
    min_val = scaler.data_min_[feature_index]
    max_val = scaler.data_max_[feature_index]
    return data * (max_val - min_val) + min_val

# 反归一化预测值
y_pred_close = inverse_transform(scaler, y_pred[:, 0], feature_index=3)  # `close` 在 features 中的索引是 3
y_pred_volume = inverse_transform(scaler, y_pred[:, 1], feature_index=4)  # `volume` 在 features 中的索引是 4
y_true_close = inverse_transform(scaler, y_true[:, 0], feature_index=3)
y_true_volume = inverse_transform(scaler, y_true[:, 1], feature_index=4)

# 重新绘制曲线
def plot_predictions(y_pred, y_true, title, ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True Values", color='blue')
    plt.plot(y_pred, label="Predicted Values", color='red', linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.show()

plot_predictions(y_pred_close, y_true_close, "Transformer Model - Close Price Prediction", "Close Price ($)")
plot_predictions(y_pred_volume, y_true_volume, "Transformer Model - Volume Prediction", "Volume (shares)")


# 关闭 TensorBoard
writer.close()
