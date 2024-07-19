# -*- coding: utf-8 -*-
# @Time    : 2023/8/26 17:34
# @Author  : zwl
import pyaudio
import wave
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # 设置录音时长，单位为秒
audio_input = pyaudio.PyAudio().open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                     input=True, frames_per_buffer=CHUNK_SIZE)
frames = []

for _ in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
    audio_data = audio_input.read(CHUNK_SIZE)
    frames.append(audio_data)

output_filename = "output.wav"
with wave.open(output_filename, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 16位采样，所以采样大小为2字节
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print("录音已保存为", output_filename)

audio_input.stop_stream()
audio_input.close()
pyaudio.PyAudio().terminate()
