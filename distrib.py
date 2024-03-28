import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib as mp

# 初始化标签编码器
label_encoder = LabelEncoder()

# 假设所有训练文件位于 'train_files_directory'
train_files_directory = "../train/"
train_files = os.listdir(train_files_directory)

# 首先，收集所有的标签以便拟合 LabelEncoder
all_labels = []
for file in train_files:
    file_path = os.path.join(train_files_directory, file)
    data = pd.read_csv(file_path)
    all_labels.extend(data['type'].unique())

# 拟合 LabelEncoder 到所有可能的标签
label_encoder.fit(all_labels)

# 现在处理每个文件，并转换标签
X_all = []
y_all = []
for file in train_files:
    file_path = os.path.join(train_files_directory, file)
    data = pd.read_csv(file_path)
    
    # 转换时间列，提取特征等
    data['time'] = pd.to_datetime(data['time'])
    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek
    
    # 使用转换后的标签
    data['type_encoded'] = label_encoder.transform(data['type'])
    
    X = data[['lat', 'lon', '速度', '方向', 'hour', 'day_of_week']]
    y = data['type_encoded']
    
    X_all.append(X)
    y_all.append(y)

# 将所有数据合并为一个大的 DataFrame
X = pd.concat(X_all, ignore_index=True)
y = pd.concat(y_all, ignore_index=True)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器，并进行训练
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测，并计算准确率
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
