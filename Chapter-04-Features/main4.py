import cv2
import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import auto_sample, sample, evaluator
from feature_extractor import FeatureExtractor
from classifier import Classifier
from manifold import ManifoldPlot
from image_processing import ImageProcess
import warnings
from cuml import TSNE
warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')


data_dir = "data"
output_dir = 'output'
img_name_list = ["0618.png", "0854.png", "1066.png"]
img_name = img_name_list[0]
img = cv2.imread(os.path.join(data_dir, img_name))

img_mask = cv2.imread(os.path.join(data_dir, img_name[:-4] + "_mask.png"), flags=0)
data = img.reshape(-1, 3)

sample_interval = 50
train_data, train_label = auto_sample(data_dir, img_name, output_dir, sample_interval)

# 1. 图像预处理，包括中值滤波、高斯滤波、均值滤波、双边滤波、均值漂移滤波，并保存图像到output/1_image_processing/目录下
image_process = ImageProcess(img)
filter_list = ['median_blur', 'gaussian_blur', 'average_blur', 'bilateral_filter', 'mean_shift_filter']
for filter_item in filter_list:
    image_process.__getattribute__(filter_item)()
    filter_img_dir = os.path.join(output_dir, "1_image_processing")
    filter_img_name = filter_item + "_" + img_name
    cv2.imwrite(os.path.join(filter_img_dir, filter_img_name), image_process.img)

    # 滤波后新的图像进行采样
    train_data, train_label = sample(data_dir, img_name, filter_img_dir, filter_img_name, sample_interval)
    # 2.特征提取，包括原始特征、pca特征、kpca特征、lda特征、字典学习特征
    feature_extractor = FeatureExtractor(train_data, train_label, data)
    feature_extractor_list = ['original_feature', 'pca_feature', 'kpca_feature', 'lda_feature',
                              'dictionary_learning_feature']

    # 设置feature_extractor_params，用于后续遍历

    feature_extractor_params = dict(
        original_feature=[{}],
        pca_feature=[dict({'n_components': i}) for i in range(1, 3)],
        kpca_feature=[
            dict({'n_components': i, 'kernel': j}) for i in range(1, 3) for j in ['rbf', 'poly', 'sigmoid', 'cosine']
        ],
        lda_feature=[dict({'n_components': i}) for i in range(1, 2)],
        dictionary_learning_feature=[dict({'n_components': i}) for i in range(1, 3)]
    )


    # 特征提取，需要把feature_extractor_params传入

    for feature_extractor_item in feature_extractor_list:
        for feature_extractor_param in feature_extractor_params[feature_extractor_item]:
            feature_extractor.__getattribute__(feature_extractor_item)(**feature_extractor_param)

            # 3.流形学习，包括tsne、isomap、lle
            # 使用cuml的tsne加速
            tsne = TSNE(n_components=2, perplexity=30.0)
            tsne_train_data = tsne.fit_transform(train_data)
            plt.scatter(tsne_train_data[:, 0], tsne_train_data[:, 1], c=train_label, alpha=0.1)
            plt.show()
