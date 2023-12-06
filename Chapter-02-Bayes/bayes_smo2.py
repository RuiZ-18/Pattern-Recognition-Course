from utils import manual_sample, auto_sample, bayes, bayes_smo, bayes_smo2
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
output_dir_ = "output_auto_sample_smo2"
sample_interval_list = [x for x in range(30, 55, 10)]
# sample_interval_list = [x for x in range(10, 11, 5)]
for sample_interval in sample_interval_list:
    # sample_interval = 10
    print(f'current sample interval: {sample_interval}')

    train_data, train_label = auto_sample(data_dir, img_name, output_dir_, sample_interval)
    print(train_data.shape[0])
    resImage = bayes_smo2(img, train_data, train_label)

    # from matplotlib import pyplot as plt
    # import matplotlib

    # matplotlib.use('TkAgg')
    # plt.imshow(resImage, 'grey')
    # plt.show()
    save_name = output_dir_ + "/" + img_name[:-4] + "_{:02d}".format(
                sample_interval) + "/" + img_name[:-4] + "_res" + img_name[-4:]
    cv2.imwrite(save_name, resImage)





