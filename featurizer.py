#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/19 23:13
# @Author: ZhaoKe
# @File : featurizer.py
# @Software: PyCharm
import random
import numpy as np
import librosa
import torch
import torchaudio


class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        return self.amplitude_to_db(self.mel_transform(x))


def wav_slice_padding(old_signal, save_len=160000):
    new_signal = np.zeros(save_len)
    if old_signal.shape[0] < save_len:
        resi = save_len - old_signal.shape[0]
        # print("resi:", resi)
        new_signal[:old_signal.shape[0]] = old_signal
        new_signal[old_signal.shape[0]:] = old_signal[-resi:][::-1]
    elif old_signal.shape[0] > save_len:
        posi = random.randint(0, old_signal.shape[0] - save_len)
        new_signal = old_signal[posi:posi+save_len]
    return new_signal


def get_a_wavmel_sample(test_wav_path):
    # test_wav_path = f"G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_{mt}/{mt}/train/normal_id_00_00000016.wav"
    y, sr = librosa.core.load(test_wav_path, sr=16000)
    y = wav_slice_padding(y, 147000)
    w2m = Wave2Mel(16000)
    x_mel = w2m(torch.from_numpy(y.T))
    x_input = x_mel.unsqueeze(0).unsqueeze(0).to(torch.device("cuda")).transpose(2, 3)
    return y, x_input
