import soundfile
import matplotlib.pyplot as plt
import os
import numpy as np
import librosa

# pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

"""
功能：数据的预处理，给数据进行降噪并分析结果
"""

# 信号的读取，降噪前信号路径和降噪后输出路径
orginDirname = r".\data\train"
outputDirname = r".\data\train_after_process"
# 是否画图展示
doPlot = False
# 是否计算信噪比
doCalSnr = False

def ssa(source):
    """ssa（奇异谱分析）降噪"""
    series = source
    series = series - np.mean(series)  # 中心化(非必须)
    # step1 嵌入
    windowLen = 20  # 嵌入窗口长度
    seriesLen = len(series)  # 序列长度
    K = seriesLen - windowLen + 1
    X = np.zeros((windowLen, K))
    for i in range(K):
        X[:, i] = series[i:i + windowLen]
    # step2: svd分解， U和sigma已经按升序排序
    U, sigma, VT = np.linalg.svd(X, full_matrices=False)
    for i in range(VT.shape[0]):
        VT[i, :] *= sigma[i]
    A = VT
    # 重组
    rec = np.zeros((windowLen, seriesLen))
    for i in range(windowLen):
        for j in range(windowLen - 1):
            for m in range(j + 1):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (j + 1)
        for j in range(windowLen - 1, seriesLen - windowLen + 1):
            for m in range(windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= windowLen
        for j in range(seriesLen - windowLen + 1, seriesLen):
            for m in range(j - seriesLen + windowLen, windowLen):
                rec[i, j] += A[i, j - m] * U[m, i]
            rec[i, j] /= (seriesLen - j)
    rrr = rec[0]  # 选择重构分量，这里选择了第一组分量
    return rrr

def snr_singlech(y1, y2):
    """
    计算信噪比
    :param y1: 降噪前信号序列
    :param y2: 降噪后信号序列
    :return: 信噪比大小
    """
    length = min(len(y2), len(y1))
    # 计算噪声语音
    est_noise = y1[:length] - y2[:length]
    # 计算信噪比
    SNR = 10 * np.log10((np.sum(y2 ** 2)) / (np.sum(est_noise ** 2)))
    print(f"信噪比：{SNR}")
    return SNR

dirs = [f for f in os.listdir(orginDirname) ]
for dir in dirs:
    if not os.path.exists(outputDirname+'/'+dir):
        os.makedirs(outputDirname+'/'+dir)
res = 0
# 初始化信噪比
SNR = 0
for dir in dirs:
    files = os.listdir(orginDirname+'/'+dir)
    for i in range(len(files)):
        print(f"进度：{i+1+res}/{len(files)+res}")
        source, Fs = librosa.load(orginDirname + '/' + dir + '/' +files[i])
        t = list(value / Fs for value in range(1, len(source) + 1))
        # SSA降噪
        result = ssa(source)
        if doPlot:
            # 结果可视化
            plt.subplot(1, 1, 1)
            plt.plot(t, source)
            plt.title('origin')
            plt.subplot(1, 1, 2)
            t = list(value / Fs for value in range(1, len(result) + 1))
            plt.plot(t, result)
            plt.title('after ssa')
            plt.show()
        #写入文件
        soundfile.write(outputDirname+'/'+dir+'/' + files[i], result, Fs)
        if doCalSnr:
            SNR += snr_singlech(source, result)
    res += len(files)
if doCalSnr:
    print(f"奇异谱降噪平均信噪比：{SNR / res}")
