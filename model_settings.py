#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/18 19:52
# @Author: ZhaoKe
# @File : model_settings.py
# @Software: PyCharm
import torch

from asddskit.models.cvae_conv import ConvCVAE
from asddskit.models.vae_conv import ConvVAE
from asddskit.utils.utils import load_ckpt


def get_model(use_model, configs, istrain=True, params=None):
    model = None
    if use_model == "vae":
        model = ConvVAE(input_channel=1, input_length=configs["model"]["input_length"],
                        input_dim=configs["feature"]["n_mels"], conditional=False, num_labels=0)
    if use_model == "cvae":
        model = ConvCVAE(input_channel=1, input_length=configs["model"]["input_length"],
                         input_dim=configs["feature"]["n_mels"], **params)
    if istrain:
        if isinstance(model, list):
            for item in model:
                item.apply(weight_init)
        else:
            model.apply(weight_init)
        # amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    return model


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # if any(m.bias):
        # torch.nn.init.constant_(m.bias, 0.)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1.)
        # torch.nn.init.constant_(m.bias, 0.)


def get_pretrain_models(encoder_path, decoder_path, encoder_epoch, decoder_epoch, configs, use_gpu=True):
    if use_gpu:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    ed_models = get_model("conv_encoder_decoder", configs=configs, istrain=False)
    encoder = ed_models[0]
    decoder = ed_models[1]
    load_ckpt(encoder, encoder_path + f"/model_epoch_{encoder_epoch}", load_epoch=encoder_epoch)
    load_ckpt(decoder, decoder_path + f"/model_epoch_{decoder_epoch}", load_epoch=decoder_epoch)
    encoder.to(device).eval()
    decoder.to(device).eval()
    print("Load Pretrain AutoEncoder on Device", device, " Successfully!!")
    return encoder, decoder
