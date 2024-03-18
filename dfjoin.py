import pandas as pd
import os

train_dir = "../train/"
files = os.listdir(train_dir)

def proc(file_path):
    data = pd.read_csv(file_path)
    data['time'] = pd.to_datetime(data['time'])

    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek

    if 'type' in data.columns:
        data['type_encoded'] = label_encoder


for file in files:
    file_path = os.path.join(train_dir, file)

    df = pd.read_csv()

df = pd.read_csv("")