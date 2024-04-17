[English](./README.md) | [简体中文](README_cn.md)
# ConvolutionalVAE_withConditional

This Repository provides the implementation of Convolutional VAE and conditional variational autoencoder.


# VAE

```text
self.model = get_model("cvae", configs=self.configs, istrain=True,
                               params={"latent_dim": self.latent_dim, "conditional": False, "num_labels": 6}).to(self.device)
```

# Conditional VAE

```text
self.model = get_model("cvae", configs=self.configs, istrain=True,
                               params={"latent_dim": self.latent_dim, "conditional": True, "num_labels": self.class_num}).to(self.device)
```

# Run

```text
python trainer_cvae.py
```
### Dataset：DCASE2020 contraining 20000 waveform
Each data is a 10s audio waveform, converted to MelSpectra with a shape of (128, 288) and model input of (batch_size, 1, 288, 128).
