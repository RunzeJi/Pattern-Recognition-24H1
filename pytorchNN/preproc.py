import os
import pandas as pd
from tqdm import tqdm, trange
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train_files_path = '../../PR/train'
train_files = os.listdir(train_files_path)
print(f'[PREPROC] Found {len(train_files)} Training Files\n')

all_labels = []

train_files_pb = tqdm(train_files)
train_files_pb.set_description('[PREPROC] Loading CSV Files...')

for file in train_files_pb:
    file_path = os.path.join(train_files_path, file)
    data = pd.read_csv(file_path)
    all_labels.extend(data['type'].unique())

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

X_all = []
y_all = []

for file in tqdm(train_files):
    file_path = os.path.join(train_files_path, file)
    data = pd.read_csv(file_path)
    
    # 转换时间列，提取特征等
    data['time'] = pd.to_datetime(data['time'])
    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek
    data['month'] = data['time'].dt.month
    
    # 使用转换后的标签
    data['type_encoded'] = label_encoder.transform(data['type'])
    
    X = data[['lat', 'lon', '速度', '方向', 'hour', 'day_of_week', 'month']]
    y = data['type_encoded']
    
    X_all.append(X)
    y_all.append(y)

# 将所有数据合并为一个大的 DataFrame
X = pd.concat(X_all, ignore_index=True)
y = pd.concat(y_all, ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from torch.utils.data import Dataset, DataLoader
import torch

class FishingVesselDataset(Dataset):
    def __init__(self, features, labels):
        """
        features: 特征数据，尺寸为 (n_samples, n_features)
        labels: 标签数据，尺寸为 (n_samples,)
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 假设 X_train, y_train, X_test, y_test 已经准备好了
# 将数据转换为 PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 创建 Dataset
train_dataset = FishingVesselDataset(X_train_tensor, y_train_tensor)
test_dataset = FishingVesselDataset(X_test_tensor, y_test_tensor)

# 创建 DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)



import torch.nn as nn
import torch.nn.functional as F

class FishingVesselNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FishingVesselNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化模型
num_features = X_train.shape[1]
num_classes = len(torch.unique(y_train_tensor)) # 假设所有类别都在训练集中出现过
model = FishingVesselNet(num_features, num_classes)


import torch.optim as optim

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20

epochs_pb = tqdm(range(num_epochs))
epochs_pb.set_description('[TORCH] Training')

for epoch in epochs_pb:
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')



correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')
