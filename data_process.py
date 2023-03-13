import os

# 解压汽车数据集(无条件)
# if not os.path.exists("work/cars"):
#     !mkdir work/cars
# ! unzip -oq data/stanford_cars.zip -d works/cars

# 解压花朵数据集(有条件)
# import os
# if not os.path.exists("work/flowers"):
#     !mkdir work/flowers
# !unzip -oq data/data173680/flowers.zip -d work/flowers

import paddle
import paddle.vision
import matplotlib.pyplot as plt
from PIL import Image
import paddle.nn as nn
import paddle.vision as V
from paddle.io import Dataset, DataLoader
from tqdm import tqdm


# 定义展示图片函数
def show_images(imgs_paths=[],cols=4):
    num_samples = len(imgs_paths)
    plt.figure(figsize=(15,15))
    i = 0
    for img_path in imgs_paths:
        img = Image.open(img_path)
        plt.subplot(int(num_samples/cols + 1), cols, i + 1)
        plt.imshow(img)
        i += 1

# imgs_paths = [
#     "work/cars/cars_train/05930.jpg", "work/cars/cars_train/06816.jpg", "work/cars/cars_train/02885.jpg", "work/cars/cars_train/07471.jpg",
#     "work/cars/cars_train/06600.jpg", "work/cars/cars_train/06020.jpg", "work/cars/cars_train/04818.jpg", "work/cars/cars_train/06088.jpg"
# ]
# show_images(imgs_paths)

# 这里我们不需要用到图像标签，可以直接用paddle.vision里面提供的数据集接口
def get_data(args):
    transforms = V.transforms.Compose([
        V.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        V.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        V.transforms.ToTensor(),
        V.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = V.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def load_flower_dataset():
    train_sunflower = os.listdir("work/flowers/pic/train/sunflower")            # 0——向日葵
    valid_sunflower = os.listdir("work/flowers/pic/validation/sunflower")       # 0——向日葵
    train_rose      = os.listdir("work/flowers/pic/train/rose")                 # 1——玫瑰
    valid_rose      = os.listdir("work/flowers/pic/validation/rose")            # 1——玫瑰
    train_tulip     = os.listdir("work/flowers/pic/train/tulip")                # 2——郁金香
    valid_tulip     = os.listdir("work/flowers/pic/validation/tulip")           # 2——郁金香
    train_dandelion = os.listdir("work/flowers/pic/train/dandelion")            # 3——蒲公英
    valid_dandelion = os.listdir("work/flowers/pic/validation/dandelion")       # 3——蒲公英
    train_daisy     = os.listdir("work/flowers/pic/train/daisy")                # 4——雏菊
    valid_daisy     = os.listdir("work/flowers/pic/validation/daisy")           # 4——雏菊

    with open("flowers_data.txt", 'w') as f:
    for image in train_sunflower:
        f.write("work/flowers/pic/train/sunflower/" + image + ";" + "0" + "\n")
    for image in valid_sunflower:
        f.write("work/flowers/pic/validation/sunflower/" + image + ";" + "0" + "\n")
    for image in train_rose:
        f.write("work/flowers/pic/train/rose/" + image + ";" + "1" + "\n")
    for image in valid_rose:
        f.write("work/flowers/pic/validation/rose/" + image + ";" + "1" + "\n")
    for image in train_tulip:
        f.write("work/flowers/pic/train/tulip/" + image + ";" + "2" + "\n")
    for image in valid_tulip:
        f.write("work/flowers/pic/validation/tulip/" + image + ";" + "2" + "\n")
    for image in train_dandelion:
        f.write("work/flowers/pic/train/dandelion/" + image + ";" + "3" + "\n")
    for image in valid_dandelion:
        f.write("work/flowers/pic/validation/dandelion/" + image + ";" + "3" + "\n")
    for image in train_daisy:
        f.write("work/flowers/pic/train/daisy/" + image + ";" + "4" + "\n")
    for image in valid_daisy:
        f.write("work/flowers/pic/validation/daisy/" + image + ";" + "4" + "\n")


# 数据变换
transforms = V.transforms.Compose([
        V.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        V.transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        V.transforms.ToTensor(),
        V.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

class TrainDataFlowers(Dataset):
    def __init__(self, txt_path="flowers_data.txt"):
        with open(txt_path, "r") as f:
            data = f.readlines()
        self.image_paths = data[:-1]    # 最后一行是空行，舍弃
    
    def __getitem__(self, index):
        image_path, label = self.image_paths[index].split(";")
        image = Image.open(image_path)
        image = transforms(image)

        label = int(label)
        
        return image, label
    
    def __len__(self):
        return len(self.image_paths)



if __name__ == "__main__": # 测试数据集是否可用
    # dataset = TrainDataFlowers()
    # dataloader = DataLoader(dataset, batch_size=24, shuffle=True)
    # pbar = tqdm(dataloader)
    # for i, (images, labels) in enumerate(pbar):
    #     pass
    print("ok")