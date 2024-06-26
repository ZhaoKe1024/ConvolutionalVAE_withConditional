#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/20 14:14
# @Author: ZhaoKe
# @File : trainer_cvae_2021.py
# @Software: PyCharm
import os
import time
import yaml
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from featurizer import get_a_wavmel_sample
from cvae_conv import ConvCVAE, vae_loss
from model_settings import get_model
from utils import setup_seed, load_ckpt
from reader_dcase2020 import SpecAllReader, select_dirs


def get_wavmel_settings(data_file, m2l_map, configs=None, mode="train", demo_test=False):
    loaders = []
    id_ma_map = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}
    ano_id_map = {"normal": 0, "anomal": 1}
    ma_id_map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
    print("---------------train dataset-------------")
    file_paths = []
    mtid_list = []
    mtype_list = []
    y_true_list = []
    with open(data_file, 'r') as fin:
        train_path_list = fin.readlines()
        if demo_test:
            train_path_list = random.choices(train_path_list, k=200)
        for item in train_path_list:
            parts = item.strip().split('\t')
            machine_type_id = int(parts[1])
            # if machine_type_id != 3:
            #     continue
            # if machine_type_id not in [3, 4]:
            #     continue
            file_paths.append(parts[0])
            mtype_list.append(machine_type_id)
            machine_id_id = parts[2]
            meta = ma_id_map[machine_type_id] + '-id_' + machine_id_id
            mtid_list.append(m2l_map[meta])
            y_true_list.append(int(item.strip().split('\t')[3]))
    if mode == "train":
        istrain, istest = True, False
    elif mode == "test":
        istrain, istest = True, True
    else:
        istrain, istest = False, True
    dataset = SpecAllReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                            y_true_list=y_true_list,
                            configs=configs,
                            istrain=istrain, istest=istest)
    loader = DataLoader(dataset, batch_size=configs["fit"]["batch_size"],
                        shuffle=True)
    return dataset, loader


class TrainerCVAE(object):
    def __init__(self, configs="./configs/cvae.yaml", istrain=True, demo_test=False):
        self.configs = None
        with open(configs) as stream:
            self.configs = yaml.safe_load(stream)
        # os.makedirs(self.configs["model_directory"], exist_ok=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        # load base_directory list
        self.train_dirs = select_dirs(param=self.configs, mode=True)
        print("---------dirs")
        for d in self.train_dirs:
            print(d)
        setup_seed(3407)
        self.num_epoch = self.configs["fit"]["epochs"]
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.demo_test = demo_test
        self.istrain = istrain
        self.klw = 0.00025
        self.latent_dim = 8
        if istrain:
            self.run_save_dir = self.configs[
                                    "run_save_dir"] + self.timestr + f'_cvae_ls8_klw0_00025/'
            if not self.demo_test:
                os.makedirs(self.run_save_dir, exist_ok=True)
                with open(self.run_save_dir + "setting_info.txt", 'w', encoding="utf_8") as fin:
                    fin.write(
                        "仍然全卷积，分开了Encoder和Decoder。")
                    fin.write("latent_dim=8, lr=1e-3 ~ 5e-5, kl_weight=0.00025。")
        # self.w2m = Wave2Mel(16000)
        with open("./datasets/d2020_metadata2label.json", 'r', encoding='utf_8') as fp:
            self.meta2label = json.load(fp)
        self.model = None
        self.train_dataset = None
        self.train_loader = None
        self.test_loader = None

    def train(self):
        self.model = get_model("cvae", configs=self.configs, istrain=True,
                               params={"latent_dim": self.latent_dim, "conditional": True, "num_labels": 6}).to(self.device)
        print("All model and loss are on device:", self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        self.loss_fn = vae_loss
        self.train_dataset, self.train_loader = get_wavmel_settings("../datasets/d2020_train_list.txt", self.meta2label,
                                                                    self.configs,
                                                                    mode="train",
                                                                    demo_test=self.demo_test)

        # 这四个存储所有epoch内的loss
        history1 = []

        for epoch in range(self.num_epoch):
            self.model.train()
            for x_idx, (_, x_mel, m_type, _, _) in enumerate(tqdm(self.train_loader, desc="Training")):
                x_mel = x_mel.unsqueeze(1) / 255.
                x_mel = x_mel.to(self.device)  # .transpose(2, 3)
                m_type = m_type.to(self.device)
                # print(x_mel.shape)
                optimizer.zero_grad()
                recon_mel, _, latent_mean, latent_logvar = self.model(input_mel=x_mel, conds=m_type)
                # print("recon shape")
                # print(recon_mel.shape, latent_mean.shape, latent_logvar.shape)
                recon_loss = self.loss_fn(recon_mel, x_mel, latent_mean, latent_logvar, kl_weight=self.klw)
                # mtid_pred_loss = class_loss(mtid_pred, mtid)
                # coarse_weight = 0.1125
                # sum_loss = coarse_weight * coarse_loss / coarse_loss.detach() + (
                #             1 - coarse_weight) * fine_loss / fine_loss.detach()
                # sum_loss.backward()
                recon_loss.backward()
                optimizer.step()

                history1.append(recon_loss.item())
                # history2.append(fine_loss.item())
                if x_idx % 90 == 0:
                    print(f"Epoch[{epoch}], recon loss:{recon_loss.item():.4f}")
                    # print(
                    #     f"Epoch[{epoch}], mtid pred loss:{coarse_loss.item():.4f}, mtid pred loss:{fine_loss.item():.4f}")

            plt.figure(0)
            plt.plot(range(len(history1[100:])), history1[100:], c="green", alpha=0.7)
            plt.savefig(self.run_save_dir + f'mtype_loss_iter_{epoch}.png')
            os.makedirs(self.run_save_dir + f"model_epoch_{epoch}/", exist_ok=True)
            tmp_model_path = "{model}model_{epoch}.pth".format(
                model=self.run_save_dir + f"model_epoch_{epoch}/",
                epoch=epoch)
            torch.save(self.model.state_dict(), tmp_model_path)
            self.test_recon(self.run_save_dir, epoch)
            if epoch >= self.configs["model"]["start_scheduler_epoch"]:
                scheduler.step()
        print("============== END TRAINING ==============")
        plt.close()

    def test_recon(self, resume_model, load_epoch, latent_dim=4):
        if self.model is None:
            self.model = ConvCVAE(input_channel=1, input_length=288,
                                  input_dim=128, latent_dim=latent_dim).to(self.device)
            load_ckpt(self.model, resume_model, load_epoch)
        self.model.eval()
        # print(self.model)
        # data_tf = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize([0.5], [0.5])]
        # )
        id_ma_map = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}
        m_types = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
        wav_path = [
            "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_fan/fan/train/",
            "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_pump/pump/train/",
            "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_slider/slider/train/",
            "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_valve/valve/train/",
            "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_ToyCar/ToyCar/train/",
            "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_ToyConveyor/ToyConveyor/train/"
        ]

        # m_types = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
        # wav_path = [
        #     f"F:/DATAS/DCASE2024Task2ASD/dev_{item}/{item}/train/"
        #     for item in m_types
        # ]

        wav_list = [
            wav_path[i] + random.choice(os.listdir(wav_path[i])) for i in range(len(wav_path))
        ]
        # print(x_input.shape)
        # latent_spec = encoder_model(input_mel=x_input, label_vec=None, ifcls=False)
        # print(latent_spec.shape)
        for i in range(len(wav_list)):
            # if i != 3:
            #     continue
            _, x_input = get_a_wavmel_sample(wav_list[i])
            recons_spec, _, _, _ = self.model(x_input, torch.tensor([id_ma_map[m_types[i]]]))
            # print(recons_spec.shape)

            # print(z.shape)

            plt.figure(0)
            plt.subplot(2, 1, 1)
            plt.imshow(np.asarray(x_input.transpose(2, 3).squeeze().data.cpu().numpy()))

            plt.subplot(2, 1, 2)
            plt.imshow(np.asarray(recons_spec.transpose(2, 3).squeeze().data.cpu().numpy()))
            plt.savefig(resume_model + f"testrecon_{load_epoch}_{m_types[i]}.png")
            plt.close()
            # plt.show()


if __name__ == '__main__':
    trainer = TrainerCVAE(configs="../configs/cvae.yaml", istrain=True, demo_test=False)
    trainer.train()

    # trainer = TrainerCVAE(configs="./configs/cvae.yaml", demo_test=True)
    # load_epoch = 9
    # # trainer.test_recon(resume_model=f"./run/VAE/202402191838_vae_klw0_00025_home/model_epoch_{load_epoch}",
    # #                    load_epoch=load_epoch)
    # trainer.test_recon(resume_model=f"./run/VAE/202402192327_vae_ls4_klw0_00025_home/model_epoch_{15}",
    #                    load_epoch=15)

    # model = ConvCVAE(input_channel=1, input_length=288, input_dim=128, latent_dim=4)
    # load_ckpt(model, "../run/VAE/202402192327_vae_ls4_klw0_00025_home/model_epoch_15", 15)
    # print(model)
