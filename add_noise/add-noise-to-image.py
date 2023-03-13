import cv2
import numpy as np 

# 读取图像

img = cv2.imread("00002.jpg")
# 设置噪声强度
noise_strength = 1      # 标准差
beta_t = 0.0001         #初始数值
alpha_t = 1 - beta_t    #初始数值

T = 30 #加噪步数
temp = (0.02 - 0.0001) / T 

# 生成高斯噪声
noise = np.random.normal(loc=0, scale=noise_strength*255, size=img.shape)

noisy_img = pow(alpha_t, 0.5) * img + pow(1-alpha_t, 0.5) * noise 
noisy_img_n = noisy_img.astype(np.uint8)

for item in range(6):
	beta_t = beta_t + temp
	alpha_t = 1 - beta_t 
	noisy_img = pow(alpha_t, 0.5) * noisy_img_n + pow(1-alpha_t, 0.5) * noise 
	noisy_img_n = noisy_img.astype(np.uint8)
	cv2.imshow('Noisy Image {}'.format(item), noisy_img_n)

# 等待按键事件
cv2.waitKey(0)

# 关闭窗口
cv2.destroyAllWindows()
