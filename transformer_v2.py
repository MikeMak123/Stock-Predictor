import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

'''
 主要优化方向
✅ 改进目标变量归一化方式（采用标准化 + 均值调整）
✅ 优化 Transformer 结构：

增加 Transformer 层数 & 头数，增强长期趋势建模能力

加入 CNN 卷积层，提取短期特征 ✅ 改进损失函数：

MSELoss → Smooth L1 Loss，减少异常值影响 ✅ 每轮训练后立即进行测试，记录 训练 & 测试 R² 曲线，便于优化
'''

# 设备选择（CUDA 自动检测）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取数据
df = pd.read_csv('Amazon stock data 2000-2025.csv')
df['date'] = pd.to_datetime(df['date'])

# 选择特征
features = ['open', 'high', 'low', 'close', 'volume']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])  # 采用标准化而非 MinMaxScaler

# 构造时间序列数据
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

# Transformer + CNN 预测模型
class TransformerCNN(nn.Module):
    def __init__(self, input_size, d_model, num_layers, nhead, output_size):
        super(TransformerCNN, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        
        # 卷积层（提取短期局部特征）
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # 调整形状以适应 CNN
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # 变回 Transformer 输入格式
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
        return out

# 初始化 Transformer + CNN 模型
input_size = len(features)  # 5 个特征
d_model = 128
num_layers = 4
nhead = 8
output_size = 2  # 预测 `close` 和 `volume`
model = TransformerCNN(input_size, d_model, num_layers, nhead, output_size).to(device)

# 训练配置
criterion = nn.SmoothL1Loss()  # Huber Loss，减少异常值影响
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# TensorBoard 记录
writer = SummaryWriter()

# 训练 & 测试记录
train_r2_scores, test_r2_scores = [], []

# 训练模型（每轮训练后立即测试）
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

    # 计算测试集 R²
    y_test_pred = model(X_test).cpu().detach().numpy()
    y_test_true = y_test.cpu().numpy()
    r2_test = 1 - np.sum((y_test_pred - y_test_true) ** 2) / np.sum((y_test_true - y_test_true.mean(axis=0)) ** 2)

    train_r2_scores.append(r2_train)
    test_r2_scores.append(r2_test)

    writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
    writer.add_scalar("Accuracy/train", r2_train, epoch)
    writer.add_scalar("Accuracy/test", r2_test, epoch)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")
    scheduler.step(total_loss / len(train_loader))

# 绘制训练 & 测试 R² 变化
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_r2_scores, label="Train R²", color="blue")
plt.plot(range(epochs), test_r2_scores, label="Test R²", color="red")
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.title("Training & Testing Accuracy over Epochs")
plt.legend()
plt.savefig("r2_score_over_epochs.png")
plt.show()

# 预测 vs 真实数据可视化 & 保存
def plot_predictions(y_pred, y_true, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:, 0], label="True Close Price", color='blue')
    plt.plot(y_pred[:, 0], label="Predicted Close Price", color='red', linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel("Standardized Close Price")
    plt.title(f"{title} - Close Price Prediction")
    plt.legend()
    plt.savefig(f"{filename}_close.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:, 1], label="True Volume", color='blue')
    plt.plot(y_pred[:, 1], label="Predicted Volume", color='red', linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel("Standardized Volume")
    plt.title(f"{title} - Volume Prediction")
    plt.legend()
    plt.savefig(f"{filename}_volume.png")
    plt.show()

y_pred = model(X_test).cpu().detach().numpy()
y_true = y_test.cpu().numpy()
plot_predictions(y_pred, y_true, "Transformer Model", "prediction_vs_actual")

# 关闭 TensorBoard
writer.close()