import torch
from torchvision import datasets, transforms
import cv2
import numpy as np

import matplotlib.pyplot as plt

# mnist_train = datasets.MNIST(root='~/datasets', train=True, download=True, transform=transforms.ToTensor())
# mnist_test = datasets.MNIST(root='~/datasets', train=False, download=True, transform=transforms.ToTensor())

cifar10_train = datasets.CIFAR10(root='~/datasets/cifar10', train=True, download=True, transform=transforms.ToTensor())
cifar10_test = datasets.CIFAR10(root='~/datasets/cifar10', train=False, download=True, transform=transforms.ToTensor())


def add_noise(img, intensity=0.5):
    noise = np.random.randn(*img.shape) * intensity
    noisy_img = img + noise
    np.clip(noisy_img, 0, 255, out=noisy_img)
    return noisy_img.astype(np.uint8)

def add_shape_with_opacity(img, shape, opacity=0.5, color=(255, 0, 0), texture=None):
    overlay = img.copy()
    
    if shape == 'triangle':
        points = np.array([[15, 5], [25, 20], [5, 20]], np.int32)
        cv2.fillPoly(overlay, [points], color)
    elif shape == 'square':
        cv2.rectangle(overlay, (5, 5), (20, 20), color, -1)
    elif shape == 'circle':
        cv2.circle(overlay, (15, 15), 10, color, -1)
    elif shape == 'ellipse':
        cv2.ellipse(overlay, (15, 15), (10, 15), 0, 0, 360, color, -1)
    elif shape == 'star':
        points = np.array([[15, 0], [19, 10], [30, 10], [21, 17], 
                           [24, 30], [15, 22], [6, 30], [9, 17], 
                           [0, 10], [11, 10]], np.int32)
        cv2.fillPoly(overlay, [points], color)
    elif shape == 'diamond':
        points = np.array([[15, 5], [25, 15], [15, 25], [5, 15]], np.int32)
        cv2.fillPoly(overlay, [points], color)
    elif shape == 'heart':
        # 绘制两个半圆
        cv2.ellipse(overlay, (10, 10), (5, 5), 0, 0, 180, color, -1)
        cv2.ellipse(overlay, (20, 10), (5, 5), 0, 0, 180, color, -1)
        # 绘制下方的三角形
        points = np.array([[5, 10], [15, 20], [25, 10]], np.int32)
        cv2.fillPoly(overlay, [points], color)    
    elif shape == 'cross':
        cv2.line(overlay, (10, 10), (20, 20), color, 5)
        cv2.line(overlay, (20, 10), (10, 20), color, 5)
    elif shape == 'pentagon':
        points = np.array([[15, 0], [30, 10], [24, 30], [6, 30], [0, 10]], np.int32)
        cv2.fillPoly(overlay, [points], color)
    elif shape == 'hexagon':
        points = np.array([[10, 5], [20, 5], [25, 15], [20, 25], [10, 25], [5, 15]], np.int32)
        cv2.fillPoly(overlay, [points], color)
    if texture == 'noise':
        overlay = add_noise(overlay)

    return cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)



# 加载MNIST数据集的一个样本
# mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

image, label = cifar10_train[0]  # 获取第一个样本

# 将图像转换为OpenCV格式
image_cv = image.numpy().squeeze() * 255
image_cv = image_cv.astype(np.uint8)

# 添加图形和透明度
shapes = ['triangle', 'square', 'circle', 'ellipse', 'star', 'diamond', 'heart', 'cross', 'pentagon', 'hexagon']

for shape in shapes:
    image_with_shape = add_shape_with_opacity(image_cv, shape, 0.8, color=(225, 255, 225), texture='noise')

    # 可视化结果
    plt.figure(figsize=(12, 6))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image_cv, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # 添加图形后的图像
    plt.subplot(1, 2, 2)
    plt.imshow(image_with_shape, cmap='gray')
    plt.title("Image with Shape")
    plt.axis('off')
    plt.savefig(f'generated/cifar10_{shape}.png')

# plt.show()


