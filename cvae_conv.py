#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/18 19:34
# @Author: ZhaoKe
# @File : cvae_conv.py
# @Software: PyCharm
import torch
import torch.nn as nn


class ConvCVAE(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128,
                 latent_dim=8, conditional=False, num_labels=0):  # , class_num=23, class_num1=6):
        super(ConvCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.latent_w = input_length // 8 - 3
        self.latent_h = input_dim // 8 - 3
        print("latent shape:", self.latent_dim, self.latent_w, self.latent_h)
        self.encoder = ConvVAEEncoder(input_channel=input_channel,
                                      input_length=input_length,
                                      input_dim=input_dim,
                                      latent_dim=latent_dim,
                                      conditional=conditional, num_labels=num_labels)
        self.decoder = ConvVAEDecoder(input_channel=input_channel,
                                      input_length=input_length,
                                      input_dim=input_dim,
                                      latent_dim=latent_dim, max_cha=self.encoder.max_cha,
                                      conditional=conditional, num_labels=num_labels)

    def forward(self, input_mel, conds=None, dec=True):
        _, means, logvar = self.encoder(input_mel, conds)
        # print("shape of latent map: ", latent_map.squeeze().shape)  # [64, 256, 33, 13]
        # print("shape of means logvar:", means.shape, logvar.shape)  # [bs, 8, 33, 13] [bs, 8, 33, 13]
        z = self.reparameterize(means, logvar)
        # print("shape of z: ", z.squeeze().shape)  # [64, 8, 33, 13]
        if not dec:
            return means, logvar
        else:
            recon_mel = self.decoder(z, conds)
            # print("shape of recon:", recon_mel.shape)  # [64, 1, 288, 128]
            return recon_mel, z, means, logvar

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def label_embedding(self, z, conds):
        b, c, h, w = z.shape
        # print("zshape before concat:", z.shape)
        # print(conds.cpu().numpy().tolist())
        if conds.ndim == 1:
            conds_x = torch.zeros(size=(b, self.num_labels, h, w), device=self.decoder.device)
            # print("shape of physical parameters: ", conds_x.shape)
            for i, idx in enumerate(conds.cpu().numpy()):
                # print(i, idx)
                conds_x[i, idx, :, :] = 1  # 1 意味着复制 0 意味着没有
        else:
            conds_x = torch.zeros(size=(b,))
            print("ndim=2?")
            # pass
        z = torch.concat((z, conds_x), dim=1)

    def generate_sample(self, class_idx, device="cpu", latent_vec=None):
        if type(class_idx) is int:
            class_idx = torch.tensor(class_idx)
        class_idx = class_idx.to(device)
        if class_idx.ndim == 0:
            class_idx = class_idx.unsqueeze(0)
            bs = 1
        else:
            bs = class_idx.shape[0]
        # print("getting conds:", class_idx)
        # conds_x = torch.zeros(size=(batch_size, self.num_labels, h, w), device=self.decoder.device)
        # # print("shape of physical parameters: ", conds_x.shape)
        # for i, idx in enumerate(class_idx.cpu().numpy()):
        #     # print(i, idx)
        #     conds_x[i, idx, :, :] = 1  # 1 意味着复制 0 意味着没有
        if latent_vec is None:
            latent_vec = torch.randn((bs, self.latent_dim, self.latent_w, self.latent_h)).to(device)
        else:
            if latent_vec.ndim == 3:
                latent_vec = latent_vec.unsqueeze(0)
            # latent_vec = latent_vec.to(device)
        # z = torch.concat((z, conds_x), dim=1)

        # y = self.label_embedding(class_idx)
        res = self.decoder(latent_vec, class_idx)
        # if not batch_size:
        #     res = res.squeeze(0)
        return res

MSE_loss = nn.MSELoss(reduction="mean")


def vae_loss(X, X_hat, mean, logvar, kl_weight=0.0001):
    reconstruction_loss = MSE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    # print(reconstruction_loss.item(), KL_divergence.item())
    return reconstruction_loss + kl_weight * KL_divergence


class ConvVAEEncoder(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128,
                 latent_dim=16, conditional=False, num_labels=0):  # , class_num=23, class_num1=6):
        super(ConvVAEEncoder, self).__init__()
        self.conditional = conditional
        self.num_labels = num_labels
        self.input_dim = input_channel
        self.max_cha = 256
        es = [input_channel, 32, 64, 128, self.max_cha]  # , 128]
        self.encoder_layers = nn.Sequential()
        kernel_size, stride, padding = 4, 2, 1
        for i in range(len(es) - 2):
            self.encoder_layers.append(
                nn.Conv2d(es[i], es[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(
                nn.LayerNorm((es[i + 1], input_length // (2 ** (i + 1)), input_dim // (2 ** (i + 1)))))
            self.encoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.encoder_layers.append(nn.Conv2d(es[-2], es[-1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        self.z_len = input_length // 8 - 3
        self.z_dim = input_dim // 8 - 3

        self.mean_linear = nn.Conv2d(self.max_cha + num_labels, latent_dim, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.var_linear = nn.Conv2d(self.max_cha + num_labels, latent_dim, kernel_size=1, stride=1, padding=0,
                                    bias=False)

    def forward(self, input_mel, conds):
        z = self.encoder_layers(input_mel)
        # print("z: ", z.squeeze().shape)
        if self.conditional:
            b, c, h, w = z.shape
            # print("zshape before concat:", z.shape)
            # print(conds.cpu().numpy().tolist())
            if conds.ndim == 1:
                conds_x = torch.zeros(size=(b, self.num_labels, h, w), device=input_mel.device)
                # print("shape of physical parameters: ", conds_x.shape)
                for i, idx in enumerate(conds.cpu().numpy()):
                    # print(i, idx)
                    conds_x[i, idx, :, :] = 1  # 1 意味着复制 0 意味着没有
            else:
                conds_x = torch.zeros(size=(b,))
                print("ndim=2?")
                # pass
            z = torch.concat((z, conds_x), dim=1)
            # print("zshape after concat:", z.shape)
        means = self.mean_linear(z)
        logvar = self.var_linear(z)
        return z, means, logvar


class ConvVAEDecoder(nn.Module):
    def __init__(self, max_cha, input_channel=1, input_length=288, input_dim=128, latent_dim=16
                 , conditional=False, num_labels=0):
        super(ConvVAEDecoder, self).__init__()
        self.decoder_projection = nn.Conv2d(latent_dim + num_labels, max_cha, kernel_size=1, stride=1, padding=0,
                                            bias=False)
        self.decoder_layers = ConvDecoder(input_channel=input_channel, input_length=input_length, input_dim=input_dim, max_cha=max_cha)
        self.conditional = conditional
        self.num_labels = num_labels

    def forward(self, latent_mel, conds):
        if self.conditional:
            b, c, h, w = latent_mel.shape
            # print("decoder zshape before concat:", latent_mel.shape)
            if conds.ndim == 1:
                conds_x = torch.zeros(size=(b, self.num_labels, h, w), device=latent_mel.device)
                for i in range(len(latent_mel)):
                    conds_x[i, conds[i], :, :] = 1
            else:
                conds_x = torch.zeros(size=(b,))
                print("ndim=2?")
                # pass
            latent_mel = torch.concat((latent_mel, conds_x), dim=1)
            # print("decoder zshape after concat:", latent_mel.shape)
        recon_input = self.decoder_projection(latent_mel)
        # print("proj after:", recon_input.shape)
        recon_mel = self.decoder_layers(recon_input)
        return recon_mel


class ConvDecoder(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128, max_cha=0):
        super(ConvDecoder, self).__init__()
        self.max_cha = max_cha if max_cha > 0 else 256
        kernel_size, stride, padding = 4, 2, 1
        z_len = input_length // 8
        z_dim = input_dim // 8
        ds = [self.max_cha, 128, 64, 32, input_channel]

        self.decoder_layers = nn.Sequential()
        self.decoder_layers.append(
            nn.ConvTranspose2d(ds[0], ds[1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        self.decoder_layers.append(nn.LayerNorm((ds[1], z_len, z_dim)))
        self.decoder_layers.append(nn.ReLU(inplace=True))
        for i in range(1, len(ds) - 2):
            self.decoder_layers.append(
                nn.ConvTranspose2d(ds[i], ds[i + 1], kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False))
            self.decoder_layers.append(nn.LayerNorm((ds[i + 1], z_len * 2 ** i, z_dim * 2 ** i)))
            self.decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder_layers.append(
            nn.ConvTranspose2d(ds[-2], ds[-1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.decoder_layers.append(nn.Tanh())

    def forward(self, latent_map):
        """
        :param latent_map: shape (33, 13)
        :return:
        """
        d = self.decoder_layers(latent_map)
        return d


if __name__ == '__main__':
    # test c encoder
    batch_size, physics_num = 32, 10
    input_mels = torch.rand(batch_size, 1, 288, 128)
    input_conds = torch.randint(0, physics_num, size=(batch_size,))

    # emodel = ConvVAEEncoder(conditional=True, num_labels=physics_num)  # .to("cuda")
    # embed_vec, embed_mean, embed_lgovar = emodel(input_mels, input_conds)
    # print("shape of embedding:", embed_vec.shape)
    # # test c decoder
    # dmodel = ConvVAEDecoder(conditional=True, num_labels=physics_num, max_cha=emodel.max_cha)
    # recon_mel = dmodel(embed_vec, conds=input_conds)
    # print("shape of recon mel:", recon_mel.shape)

    model = ConvCVAE(conditional=True, num_labels=physics_num)
    recon_x, z, me, lo = model(input_mels, conds=torch.randint(0, physics_num, size=(batch_size,)))
    print(recon_x.shape, z.shape, me.shape, lo.shape)
    # input_mel = torch.rand(32, 1, 288, 128)  # .to("cuda")
    # input_mean = torch.rand(64, 128)
    # input_logvar = torch.rand(64, 128)
    # print(vae_loss_1(recon_mel, input_mel, input_mean, input_logvar))
    # recon_x = dmodel(z, conds=torch.randint(0, 10, size=(32,)))
    # loss = vae_loss_fn(recon_x, input_mel, me, lo)
    # print(loss)
    # loss.backward()
    # print(recon_x.shape)
