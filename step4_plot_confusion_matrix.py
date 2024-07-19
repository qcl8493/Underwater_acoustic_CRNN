import librosa
import numpy as np
from Util import DataLoad
from Util.dtw import dtw
from scipy.linalg import norm
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from models import Transfer_VGG, Resnet, CRNN

# 测试数据集和训练数据集
test_dirname = r'data\test'
train_dirname = r'data\train_after_process'

"""绘制混淆矩阵"""
'''以下部分为画KNN的混淆矩阵'''
temp = np.load('save_model/KNN/data_1.npz')
temp2 = np.load('save_model/KNN/data2_1.npz')
distances = temp.f.arr_0
label = temp2.f.arr_0
classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
# fit可以简单的认为是表格存储
classifier.fit(distances, label)
test_dirs = [f for f in os.listdir(test_dirname)]
train_dirs = [f for f in os.listdir(train_dirname)]

test_real_lable = []
test_pred_lable = []
l = 0
for dir1 in test_dirs:
    files1 = os.listdir(test_dirname + '/' + dir1)
    for i in range(len(files1)):
        y1, sr1 = librosa.load(test_dirname + '/' + dir1 + '/' + files1[i])
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
        distanceTest = []
        for dir2 in train_dirs:
            files2 = os.listdir(train_dirname + '/' + dir2)
            for j in range(len(files2)):
                y2, sr2 = librosa.load(train_dirname + "/" + dir2 + '/' + files2[j])
                mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
                dist, _, _, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
                distanceTest.append(dist)
        pre = classifier.predict([distanceTest])[0]
        test_pred_lable.append(pre)
        test_real_lable.append(l)
    l += 1
my_confusion_matrix = confusion_matrix(test_real_lable, test_pred_lable)
normalized_confusion_matrix = normalize(my_confusion_matrix, norm='l1')
df_cm = pd.DataFrame(normalized_confusion_matrix, test_dirs, train_dirs)
sn.set(font_scale=0.9)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cmap="Blues")
plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5], rotation=45)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('Confusion Matrix About Knn')
plt.savefig(r".\picture\confusion_matrix_knn_1", dpi=900, bbox_inches='tight')
plt.show()

# '''以下部分为画神经网络的混淆矩阵'''
# labels_dict = np.load(r'save_model\label_dict.npy', allow_pickle=True).item()
# labels = list(labels_dict.values())
# print(labels)
# models = ["transfer_vgg16"]
#
# for i in range(len(models)):
#     if models[i] == 'transfer_vgg16' or models[i] == 'Resnet':
#         Data_loader = DataLoad(test_dirname, pattern=1, delta_dim=2)
#     elif models[i] == "CRNN":
#         Data_loader = DataLoad(test_dirname, pattern=1, delta_dim=1)
#     train_x, valid_x, train_y, valid_y, label_dict = Data_loader.train_data()  # 借用训练数据导入程序，原理一样
#     test_x = np.concatenate([train_x, valid_x])
#     test_y = np.concatenate([train_y, valid_y])
#     num_classes = len(np.unique(test_y))
#     test_y = to_categorical(test_y)
#     input_dim = test_x.shape[1:]
#
#     if models[i] == 'transfer_vgg16':
#         model = Transfer_VGG(num_classes, input_dim)
#     elif models[i] == 'Resnet':
#         model = Resnet(num_classes, input_dim)
#     elif models[i] == 'CRNN':
#         model = CRNN(num_classes, input_dim)
#     model.load_weights(r'.\save_model\\' + models[i] + r'\model_weight')
#
#     pred = model.predict(test_x)
#     test_y_real = np.argmax(test_y, 1)
#     test_y_pred = np.argmax(pred, 1)
#     my_confusion_matrix = confusion_matrix(test_y_real, test_y_pred)
#     normalized_confusion_matrix = normalize(my_confusion_matrix, norm='l1')
#     df_cm = pd.DataFrame(normalized_confusion_matrix, labels, labels)
#     sn.set(font_scale=0.9)
#     sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cmap="Blues")
#     plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5], rotation=45)
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.title('Confusion Matrix About ' + models[i])
#     plt.savefig(r".\picture\confusion_matrix_" + models[i] , dpi=900, bbox_inches='tight')
#     plt.show()
