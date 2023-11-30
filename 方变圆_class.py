import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
star = time.time()


class ImageTransformer:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.height, self.width = self.image.shape[:2]

        # 分割区域
        self.region = 1250
        self.upper_region = self.image[:self.region, :]
        self.lower_region = self.image[self.region:, :]

    def resize_lower_region(self):
        h_l = self.lower_region.shape[0]
        w_l = self.width
        h = int(h_l / np.cos(np.radians(41)))
        return cv2.resize(self.image[self.region:, :], (w_l, h))

    def resize_upper_region(self):
        h_u = self.upper_region.shape[0]
        w_u = self.width
        h = int(h_u / np.sin(np.radians(41)))
        return cv2.resize(self.image[:self.region, :], (w_u, h))

    def polar_to_cartesian(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def cartesian_to_polar(self, x, y, cartesian_width):
        theta = 2 * np.pi * x / cartesian_width
        r = y
        return r, theta

    def transform(self, image):
        h_t, w_t = image.shape
        radius = h_t
        d = 2 * radius

        x = np.arange(w_t)
        y = np.arange(h_t)
        xp, yp = np.meshgrid(x, y)  # 创建网格坐标矩阵

        polar_img = np.zeros((d, d), dtype=np.uint8)

        r, theta = self.cartesian_to_polar(xp, yp, w_t)  # 转换为极坐标
        new_x, new_y = self.polar_to_cartesian(r, theta)  # 转换为笛卡尔坐标
        new_x = np.floor(new_x + radius).astype(int)
        new_y = np.floor(new_y + radius).astype(int)

        valid_indices = (new_x >= 0) & (new_x < d) & (new_y >= 0) & (new_y < d)  # 过滤有效的索引
        polar_img[new_y[valid_indices], new_x[valid_indices]] = image[yp[valid_indices], xp[valid_indices]]  # 更新极坐标图像

        return polar_img

    def transform_remap(self, image):
        h_t, w_t = image.shape
        radius = h_t
        d = 2 * radius

        x = np.arange(w_t)
        y = np.arange(h_t)
        xp, yp = np.meshgrid(x, y)

        r, theta = self.cartesian_to_polar(xp, yp, w_t)
        new_x, new_y = self.polar_to_cartesian(r, theta)

        new_x = (new_x + radius).astype(int)
        new_y = (new_y + radius).astype(int)

        map_x = np.zeros((d, d), dtype=np.float32)
        map_y = np.zeros((d, d), dtype=np.float32)
        map_x[new_y, new_x] = xp
        map_y[new_y, new_x] = yp

        remapped_img = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        return remapped_img

    def center_matrix(self, image):
        pad_width = self.width - image.shape[1]
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad
        c_matrix = np.pad(image, ((0, 0), (left_pad, right_pad)), mode='constant')
        return c_matrix

    def transform_image(self):
        # 角度矫正
        resize_upper = self.resize_upper_region()
        resize_lower = self.resize_lower_region()
        print('resize_lower:', resize_lower.shape, '\tresize_upper:', resize_upper.shape)

        # 顶面映射为圆形
        trans_upper = self.transform(resize_upper)

        # 将圆形放在中间
        center_upper = self.center_matrix(trans_upper)

        # 输出图像
        result = np.vstack((center_upper, resize_lower))

        return result


if __name__ == '__main__':
    img = ImageTransformer('123.jpg')
    transformed_image = img.transform_image()
    print('transformed_image', transformed_image.shape)
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))

    # 计时
    end = time.time()
    print('time:', end - star)

    plt.show()
