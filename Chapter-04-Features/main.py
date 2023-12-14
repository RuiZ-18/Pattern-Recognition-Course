import cv2
import os
import numpy as np
from utils import auto_sample, evaluator
from feature_extractor import pca_feature, kpca_feature, lda_feature, dictionary_learning_feature
from classifier import svm_clf, random_forest_clf, knn_clf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


data_dir = "data"
output_dir = 'output'
img_name_list = ["0618.png", "0854.png", "1066.png"]
img = cv2.imread(os.path.join(data_dir, img_name_list[0]))
img_mask = cv2.imread(os.path.join(data_dir, img_name_list[0][:-4] + "_mask.png"), flags=0)
data = img.reshape(-1, 3)

sample_interval = 30
train_data, train_label = auto_sample(data_dir, img_name_list[0], output_dir, sample_interval)


# # 用svm进行分类，核函数采用poly
# res = svm_clf(train_data, train_label, data, kernel='poly')
# cv2.imshow("res", res)
# cv2.waitKey(0)
# print(evaluator(res, img_mask))


# 用pca提取特征
# train_data, data = pca_feature(train_data, data, n_components=2)

# 用kpca提取特征
# train_data, data = kpca_feature(train_data, data, n_components=2, kernel='poly')

# 用lda提取特征
# train_data, data = lda_feature(train_data, train_label, data, n_components=1)

# 字典学习方法
train_data, data = dictionary_learning_feature(train_data, data, n_components=1)

# # 用svm进行分类
# res = svm_clf(train_data, train_label, data, kernel='poly')
# cv2.imshow("res", res)
# cv2.waitKey(0)
# print(evaluator(res, img_mask))


# 使用随机树
# res = random_forest_clf(train_data, train_label, data)
# cv2.imshow("res", res)
# cv2.waitKey(0)
# print(evaluator(res, img_mask))


# 使用knn
res = knn_clf(train_data, train_label, data)
cv2.imshow("res", res)
cv2.waitKey(0)
print(evaluator(res, img_mask))
