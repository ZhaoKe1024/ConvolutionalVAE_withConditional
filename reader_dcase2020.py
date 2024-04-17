#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/18 19:30
# @Author: ZhaoKe
# @File : reader_dcase2020.py
# @Software: PyCharm
import os
import glob
from tqdm import tqdm
import librosa
import torch
from torch.utils.data import Dataset
from featurizer import Wave2Mel, wav_slice_padding


class SpecAllReader(Dataset):
    def __init__(self, file_paths, mtype_list, mtid_list, y_true_list, configs, istrain=True, istest=False):
        self.files = file_paths
        self.mtids = mtid_list
        self.mtype_list = mtype_list
        self.y_true = y_true_list
        self.configs = configs
        self.w2m = Wave2Mel(16000)
        self.device = torch.device("cuda")
        self.wav_form = []
        self.mel_specs = []
        if file_paths:
            for fi in tqdm(file_paths, desc=f"build Set..."):
                self.mel_specs.append(self.load_wav_2mel(fi))
        self.istrain = istrain
        self.istest = istest

    def __getitem__(self, ind):
        if self.istrain:
            if not self.istest:
                # print("***")
                return self.wav_form[ind], self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind]
            else:
                # print("????")
                return self.wav_form[ind], self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
            # if self.istrain:
            #     return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.mtype_list[ind], self.mtids[ind]
            # else:
            #     return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
        else:
            # print("!!!")
            # print(self.mel_specs[ind].shape, self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind])
            return self.wav_form[ind], self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
            # return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.y_true[ind], self.files[ind]

    def __len__(self):
        return len(self.files)

    def load_wav_2mel(self, wav_path):
        # print(wav_path)
        y, sr = librosa.core.load(wav_path, sr=16000)
        # print("....\n", self.configs)
        y = wav_slice_padding(y, save_len=self.configs["feature"]["wav_length"])
        self.wav_form.append(torch.tensor(y, device=self.device).to(torch.float32))
        x_mel = self.w2m(torch.from_numpy(y.T))
        return torch.tensor(x_mel, device=self.device).transpose(0, 1).to(torch.float32)


def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        print("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
    else:
        print("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs
