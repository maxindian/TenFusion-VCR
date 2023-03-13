import os
import paddle
import copy
import paddle.nn as nn
from matplotlib import pyplot as plt

from tqdm import tqdm
from paddle import optimizer
from modules import UNet_conditional, EMA
import logging
import numpy as np
from paddle.io import DataLoader

from data_process import TrainDataFlowers

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = paddle.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return paddle.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = paddle.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = paddle.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = paddle.randn(shape=x.shape)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return paddle.randint(low=1, high=self.noise_steps, shape=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with paddle.no_grad():
            x = paddle.randn((n, 3, self.img_size, self.img_size))
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = paddle.to_tensor([i] * x.shape[0]).astype("int64")
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    cfg_scale = paddle.to_tensor(cfg_scale).astype("float32")
                    predicted_noise = paddle.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = paddle.randn(shape=x.shape)
                else:
                    noise = paddle.zeros_like(x)
                x = 1 / paddle.sqrt(alpha) * (x - ((1 - alpha) / (paddle.sqrt(1 - alpha_hat))) * predicted_noise) + paddle.sqrt(beta) * noise
        model.train()
        x = (x.clip(-1, 1) + 1) / 2
        x = (x * 255)
        return x


def train(args):
    # setup_logging(args.run_name)
    device = args.device
    dataloader = args.dataloader
    model = UNet_conditional(num_classes=args.num_classes)
    opt = optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    # print("ema_model", ema_model)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)  # 损失函数

            opt.clear_grad()
            loss.backward()
            opt.step()

            ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 30 == 0:     # 保存模型，可视化训练结果。
            paddle.save(model.state_dict(), f"models/ddpm_cond{epoch}.pdparams")

            labels = paddle.arange(5).astype("int64")
            # 一共采样10张图片
            # 从左到右依次为-->向日葵，玫瑰，郁金香，蒲公英，雏菊
            sampled_images1 = diffusion.sample(model, n=len(labels), labels=labels)
            sampled_images2 = diffusion.sample(model, n=len(labels), labels=labels)
            # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            for i in range(5):
                img = sampled_images1[i].transpose([1, 2, 0])
                img = np.array(img).astype("uint8")
                plt.subplot(2,5,i+1)
                plt.imshow(img)
            for i in range(5):
                img = sampled_images2[i].transpose([1, 2, 0])
                img = np.array(img).astype("uint8")
                plt.subplot(2,5,i+1+5)
                plt.imshow(img)
            plt.show()


def launch():
    import argparse
    dataset = TrainDataFlowers()
    dataloader = DataLoader(dataset, batch_size=24, shuffle=True)
    # 参数设置
    class ARGS:
        def __init__(self):
            self.run_name = "DDPM_Uncondtional"
            self.epochs = 300
            self.batch_size = 48
            self.image_size = 64
            self.device = "cuda"
            self.lr = 1.5e-4
            self.num_classes = 5
            self.dataloader = dataloader


    args = ARGS()
    train(args)


if __name__ == '__main__':
    # 训练
    launch()
    pass