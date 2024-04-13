import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def read_and_process(file_path):
    # 假设文件名格式是 'train/数字.csv'
    file_number = file_path.split('/')[-1].split('.')[0]
    print(f"Processing file: {file_number}")
    data = pd.read_csv(file_path)
    # 这里可以添加更多的数据处理步骤
    return data

def process_files_concurrently(file_paths):
    with ProcessPoolExecutor() as executor:
        results = executor.map(read_and_process, file_paths)
    return pd.concat(results, ignore_index=True)

def plot_history(history):
    # 绘制训练集和验证集的准确度
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # 绘制训练集和验证集的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def main():
    file_paths = [f'../../PR/train/{i}.csv' for i in range(1, 15000)]
    all_data = process_files_concurrently(file_paths)

    # 处理数据
    features = all_data[['lat', 'lon', '速度', '方向']]
    labels = all_data['type']

    # 创建标签到整数的映射并转换标签
    unique_labels = labels.unique()
    label_to_int = {k: v for v, k in enumerate(unique_labels)}
    encoded_labels = labels.map(label_to_int)
    one_hot_labels = to_categorical(encoded_labels)

    # 重塑特征以匹配LSTM输入
    X = features.values.reshape((features.shape[0], 1, features.shape[1]))
    y = one_hot_labels

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 定义并训练模型
    model = Sequential()
    model.add(LSTM(40, return_sequences=True, input_shape=(1, features.shape[1])))
    model.add(LSTM(40))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # 可视化训练历史
    plot_history(history)

    # 模型评估
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')

    # 使用模型进行预测
    predictions = model.predict(X_test[:5])
    print(predictions)

if __name__ == '__main__':
    main()
