import paddle
from modules import UNet    # 模型
from ddpm_train import Diffusion
from matplotlib import pyplot as plt

from modules import UNet_conditional
import numpy as np

# 无条件预测
model = UNet()
model.set_state_dict(paddle.load("car_models/ddpm_uncond140.pdparams"))   # 加载模型文件
diffusion = Diffusion(img_size=64, device="cuda")

sampled_images = diffusion.sample(model, n=8)

# 采样图片
for i in range(8):
    img = sampled_images[i].transpose([1, 2, 0])
    img = np.array(img).astype("uint8")
    plt.subplot(2, 4,i+1)
    plt.imshow(img)
plt.show()

# 有条件预测
model = UNet_conditional(num_classes=5)
model.set_state_dict(paddle.load("models/ddpm_cond270.pdparams"))   # 加载模型文件
diffusion = Diffusion(img_size=64, device="cuda")

# 向日葵，玫瑰，郁金香，蒲公英，雏菊分别对应标签0，1，2，3，4
labels = paddle.to_tensor([0, 0, 0, 0, 0]).astype("int64")
# 标签引导强度
cfg_scale = 7
sampled_images = diffusion.sample(model, n=len(labels), labels=labels, cfg_scale=cfg_scale)
for i in range(5):
    img = sampled_images[i].transpose([1, 2, 0])
    img = np.array(img).astype("uint8")
    plt.subplot(1,5,i+1)
    plt.imshow(img)
plt.show()