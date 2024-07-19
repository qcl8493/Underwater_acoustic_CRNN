import pyaudio
import numpy as np
import os
import time
import wave

# 设置参数
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
NUM_RECORDINGS = 100

# 创建音频输入流
audio_input = pyaudio.PyAudio().open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                     input=True, frames_per_buffer=CHUNK_SIZE)

# 初始化神经网络
# TODO: 添加你的神经网络代码
from models import Transfer_VGG
from models import CRNN
# model_path = r'.\save_model\transfer_vgg16'
model_path = r'.\save_model\CRNN'
label_dict = np.load(r'save_model\label_dict.npy', allow_pickle=True).item()
num_classes = len(label_dict.keys())


def record_and_recognize():
    # 连续录制并识别声音
    for i in range(NUM_RECORDINGS):
        # 读取声音数据
        frames = []
        for _ in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
            audio_data = audio_input.read(CHUNK_SIZE)
            frames.append(audio_data)

        # 将音频数据保存为WAV文件
        output_filename = f"output_{i}.wav"
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        print(f"录音 {i + 1} 已保存为 {output_filename}")

        # 对声音进行识别
        # TODO: 使用神经网络对声音进行识别
        from Util.dataloader import DataLoad
        # pattern: 0表示n类，无子文件夹；1表示n类，有n个子文件夹 delta_dim: 表示差分的阶数
        # Data_loader = DataLoad(output_filename, pattern=1, delta_dim=2)
        Data_loader = DataLoad(output_filename, pattern=1, delta_dim=1)
        test_x, _ = Data_loader.test_data()
        input_dim = test_x.shape[1:]
        # new_model = Transfer_VGG(num_classes, input_dim)
        new_model = CRNN(num_classes, input_dim)
        new_model.load_weights(model_path + '/' + 'model_weight')
        Y_pred = new_model.predict(test_x)
        print(Y_pred.shape)
        Y_pred = np.argmax(Y_pred, 1)
        Y_pred_result = np.unique(Y_pred)
        for item in Y_pred_result:
            percent = float(sum(Y_pred == item)) / float(len(Y_pred))
            print("该算法有" + str(int(percent * 100)) + "%的概率是" + label_dict[item])

        #删除音频文件
        os.remove(output_filename)
        global t
        print("耗时：",time.time() - t)
        t = time.time()
    # 关闭音频输入流
    audio_input.stop_stream()
    audio_input.close()

# # 创建一个线程来录制和识别声音
# recording_thread = threading.Thread(target=record_and_recognize)
#
# # 启动线程
# recording_thread.start()
#
# # 等待线程完成
# recording_thread.join()

#单线程
import time
t = time.time()
record_and_recognize()