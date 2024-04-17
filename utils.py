#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/17 20:38
# @Author: ZhaoKe
# @File : utils.py
# @Software: PyCharm
import os
import random
import numpy as np
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_ckpt(model, resume_model, m_type=None, load_epoch=None):
    if m_type is None:
        state_dict = torch.load(os.path.join(resume_model, f'model_{load_epoch}.pth'))
    else:
        if load_epoch:
            state_dict = torch.load(os.path.join(resume_model, f'model_{m_type}_{load_epoch}.pth'))
        else:
            state_dict = torch.load(os.path.join(resume_model, f'model_{m_type}.pth'))
    model.load_state_dict(state_dict)
