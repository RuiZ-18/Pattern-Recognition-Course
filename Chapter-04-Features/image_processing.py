# 定义一个类，用于对图像预处理和后处理，预处理主要包含中值滤波、高斯滤波、均值滤波、双边滤波，后处理包含开运算和闭运算
import cv2
import numpy as np


class ImageProcess:
    def __init__(self, image):
        self.image = image
        self.img = None

    def median_blur(self, ksize=5):
        self.img = cv2.medianBlur(self.image, ksize)

    def gaussian_blur(self, ksize=(5, 5)):
        self.img = cv2.GaussianBlur(self.image, ksize, 0)

    def average_blur(self, ksize=(5, 5)):
        self.img = cv2.blur(self.image, ksize)

    def bilateral_filter(self, d=9, sigmaColor=75, sigmaSpace=75):
        self.img = cv2.bilateralFilter(self.image, d, sigmaColor, sigmaSpace)

    def mean_shift_filter(self, sp=10, sr=50):
        self.img = cv2.pyrMeanShiftFiltering(self.image, sp, sr)

    def opening(self, filename, ksize=(10, 10)):
        kernel = np.ones(ksize, np.uint8)
        self.img = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(filename, self.img)

    def closing(self, filename, ksize=(10, 10)):
        kernel = np.ones(ksize, np.uint8)
        self.img = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(filename, self.img)

    def opening_closing(self, filename, ksize=(10, 10)):
        kernel = np.ones(ksize, np.uint8)
        self.img = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        kernel = np.ones(ksize, np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(filename, self.img)

    def closing_opening(self, filename, ksize=(10, 10)):
        kernel = np.ones(ksize, np.uint8)
        self.img = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones(ksize, np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(filename, self.img)
