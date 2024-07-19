from Util.dataloader import DataLoad
import numpy as np
from models import CRNN

# 需要测试的音频数据
data_path = r'data/test_after_process/德_鱼雷/德_鱼雷_空气鱼雷涡轮发动机20.wav'

"""展示测试样本在哪一类"""
model_path = r'.\save_model\CRNN'
label_dict = np.load(r'save_model\label_dict.npy', allow_pickle=True).item()

# pattern: 0表示n类，无子文件夹；1表示n类，有n个子文件夹 delta_dim: 表示差分的阶数
Data_loader = DataLoad(data_path, pattern=1, delta_dim=1)
test_x, label = Data_loader.test_data()

input_dim = test_x.shape[1:]
num_classes = len(label_dict.keys())
new_model = CRNN(num_classes, input_dim)
new_model.load_weights(model_path + '/' + 'model_weight')
Y_pred = new_model.predict(test_x)
Y_pred = np.argmax(Y_pred, 1)
Y_pred_result = np.unique(Y_pred)
for item in Y_pred_result:
    percent = float(sum(Y_pred == item))/float(len(Y_pred))
    print("该算法有"+ str(int(percent*100))+"%的概率是"+label_dict[item])