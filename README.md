# ConvolutionalVAE_withConditional

This Repository provides Convolutional VAE (Variational AutoEncoder) implements, and Convolutional Conditional VAE.


# VAE

```text
self.model = get_model("cvae", configs=self.configs, istrain=True,
                               params={"latent_dim": self.latent_dim, "conditional": False, "num_labels": 6}).to(self.device)
```

Run

python trainer_cvae.py

# Conditional VAE

```text
self.model = get_model("cvae", configs=self.configs, istrain=True,
                               params={"latent_dim": self.latent_dim, "conditional": True, "num_labels": self.class_num}).to(self.device)
```