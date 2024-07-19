from Util.dataloader import DataLoad
import numpy as np
from tensorflow.keras.utils import to_categorical
import keras
from keras.callbacks import ModelCheckpoint
from models import CRNN

# 训练数据路径和模型保存路径
data_path = r".\data\train_after_process"
model_path = r".\save_model\CRNN"

"""训练CRNN网络"""
# pattern: 0表示n类，无子文件夹；1表示n类，有n个子文件夹 delta_dim: 表示差分的阶数
Data_loader = DataLoad(data_path, pattern=1, delta_dim=1)
train_x, valid_x, train_y, valid_y, label_dict = Data_loader.train_data()
np.save('save_model/label_dict.npy', label_dict)
print('X_train.shape:{} X_valid.shape:{}'.format(train_x.shape, valid_x.shape))
print('Y_train.shape:{} Y_valid.shape:{}'.format(train_y.shape, valid_y.shape))
print("当前选择的模型是CRNN网络")
index = np.random.permutation(train_x.shape[0])
train_x = train_x[index, :]
train_y = train_y[index, :]  # 序列重排
num_classes = len(np.unique(valid_y))  # 类别数
train_y = to_categorical(train_y)
valid_y = to_categorical(valid_y)
input_dim = train_x.shape[1:]
print(input_dim)
model = CRNN(num_classes, input_dim)
# 配置训练方法
checkpoint = ModelCheckpoint(model_path + '/' + 'model_weight',
                             monitor='val_accuracy', save_weights_only=True, verbose=1, save_best_only=True)
model.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=0.001),
              loss=['categorical_crossentropy'],
              metrics=["accuracy"])
# 执行训练过程
history = model.fit(train_x, train_y,
                    batch_size=64, epochs=50, validation_data=(valid_x, valid_y), callbacks=[checkpoint])
print("当前模型在验证集上的准确率为：" + str(max(history.history['val_accuracy'])))
