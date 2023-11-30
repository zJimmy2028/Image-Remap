import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
star = time.time()


def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def cartesian_to_polar(x, y, width):
    theta = 2 * np.pi * x / width
    r = y
    return r, theta


def transform(image):
    height, width = image.shape
    radius = height
    d = 2 * radius

    x = np.arange(width)
    y = np.arange(height)
    xp, yp = np.meshgrid(x, y)  # 创建网格坐标矩阵

    r, theta = cartesian_to_polar(xp, yp, width)
    new_x, new_y = polar_to_cartesian(r, theta)
    new_x = (new_x + radius).astype(int)
    new_y = (new_y + radius).astype(int)

    map_x = np.zeros((d, d), dtype=np.float32)
    map_y = np.zeros((d, d), dtype=np.float32)
    map_x[new_y, new_x] = xp
    map_y[new_y, new_x] = yp

    remapped_img = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return remapped_img


def transform_remap(image):
    height, width = image.shape
    radius = height
    d = 2 * radius

    x = np.arange(width)
    y = np.arange(height)
    xp, yp = np.meshgrid(x, y)  # 创建网格坐标矩阵

    polar_img = np.zeros((d, d), dtype=np.uint8)

    r, theta = cartesian_to_polar(xp, yp, width)  # 转换为极坐标
    new_x, new_y = polar_to_cartesian(r, theta)  # 转换为笛卡尔坐标
    new_x = np.round(new_x + radius).astype(int)  # 四舍五入并转换为整数
    new_y = np.round(new_y + radius).astype(int)

    valid_indices = (new_x >= 0) & (new_x < d) & (new_y >= 0) & (new_y < d)  # 过滤有效的索引
    polar_img[new_y[valid_indices], new_x[valid_indices]] = image[yp[valid_indices], xp[valid_indices]]  # 更新极坐标图像

    return polar_img


if __name__ == '__main__':
    img = cv2.imread('chessboard.jpg', cv2.IMREAD_GRAYSCALE)
    tra_img = transform_remap(img)
    print('img:', img.shape, '\ttra_img:', tra_img.shape)
    plt.imshow(tra_img, cmap='gray')
    end = time.time()
    print(end-star)
    plt.show()

