from utils import manual_sample, auto_sample, bayes
import os
import shutil
import cv2
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

output_dir = "output"
data_dir = "data"
img_name_list = ["0618.png", "0854.png", "1066.png"]
img_name = img_name_list[0]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, img_name[:-4])
img = cv2.imread(os.path.join(data_dir, img_name))
img_h, img_w, img_c = img.shape

# train_data, train_label = manual_sample(img, data_dir, img_name, output_dir)
output_dir_ = "bayes2"
sample_interval_list = [x for x in range(10, 55, 10)]
for sample_interval in sample_interval_list:
    # sample_interval = 10
    print(f'current sample interval: {sample_interval}')

    train_data, train_label = auto_sample(data_dir, img_name, output_dir_, sample_interval)

    P_1 = sum(train_label == 1) * 1.0 / train_label.shape[0]
    P_2 = 1 - P_1
    print(f'P_1: {P_1}, P_2: {P_2}')

    # 将train_label转为一维向量
    train_label = train_label.reshape(train_label.shape[0], )
    RGB1 = train_data[train_label == 1]
    RGB2 = train_data[~(train_label == 1)]
    RGB1_m = np.mean(RGB1, axis=0)
    RGB2_m = np.mean(RGB2, axis=0)
    RGB1_cov = np.cov(RGB1.T) / (RGB1.shape[0] - 1)
    RGB2_cov = np.cov(RGB2.T) / (RGB2.shape[0] - 1)

    resImage, grayImage, grayImage2 = bayes(img, P_1, P_2, RGB1_m, RGB2_m, RGB1_cov, RGB2_cov)
    from matplotlib import pyplot as plt
    import matplotlib

    # matplotlib.use('TkAgg')
    #
    # plt.imshow(resImage, 'grey')
    # plt.show()
    save_name1 = output_dir_ + "/" + img_name[:-4] + "_{:02d}".format(
                sample_interval) + "/" + img_name[:-4] + "_res" + img_name[-4:]
    cv2.imwrite(save_name1, resImage)

    save_name2 = output_dir_ + "/" + img_name[:-4] + "_{:02d}".format(
                sample_interval) + "/" + img_name[:-4] + "_gray" + img_name[-4:]
    cv2.imwrite(save_name2, grayImage)

    save_name3 = output_dir_ + "/" + img_name[:-4] + "_{:02d}".format(
                sample_interval) + "/" + img_name[:-4] + "_gray2" + img_name[-4:]
    cv2.imwrite(save_name3, grayImage2)
