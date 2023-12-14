import cv2
import os
import numpy as np
from utils import auto_sample, evaluator
from feature_extractor import FeatureExtractor
from classifier import Classifier
from manifold import ManifoldPlot
# from tsnecuda import TSNE
import matplotlib.pyplot as plt

data_dir = "data"
output_dir = 'output'
img_name_list = ["0618.png", "0854.png", "1066.png"]
img_name = img_name_list[2]
img = cv2.imread(os.path.join(data_dir, img_name))
# 对图像img做中值滤波
img = cv2.medianBlur(img, 5)
# 对图像做高斯滤波
# img = cv2.GaussianBlur(img, (5, 5), 0)
# 对图像做均值滤波
img = cv2.blur(img, (10, 10))
# 对图像做双边滤波
# img = cv2.bilateralFilter(img, 9, 75, 75)
# 对图像做均值迁移滤波
# img = cv2.pyrMeanShiftFiltering(img, 10, 50)
# 对图像做开运算
# kernel = np.ones((5, 5), np.uint8)
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 对图像做闭运算
# kernel = np.ones((5, 5), np.uint8)
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


img_mask = cv2.imread(os.path.join(data_dir, img_name[:-4] + "_mask.png"), flags=0)
data = img.reshape(-1, 3)

sample_interval = 10
train_data, train_label = auto_sample(data_dir, img_name, output_dir, sample_interval)

# 提取hog特征
# hog = cv2.HOGDescriptor()
# train_data = np.array([hog.compute(train_data[i].reshape(30, 30, 3)) for i in range(len(train_data))])
# train_data = train_data.reshape(-1, 324)
# data = np.array([hog.compute(data[i].reshape(30, 30, 3)) for i in range(len(data))])
# data = data.reshape(-1, 324)


feature_extractor = FeatureExtractor(train_data, train_label, data)
feature_extractor.original_feature()
feature_extractor.kpca_feature(n_components=5, kernel='poly')

classifier = Classifier(feature_extractor)
classifier.knn_clf()
res = classifier.res
# cv2.imshow("res", res)
# 对res做闭运算


# 开运算，消除小物体
for _ in range(1):
    kernel = np.ones((5, 5), np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
#
# 闭运算，消除小空洞
for _ in range(2):
    kernel = np.ones((10, 10), np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

plt.imshow(res)
plt.show()
# cv2.waitKey(0)
print(evaluator(res, img_mask))

# # TODO: 使用tsne-cuda加速
# # 使用tsne-cuda加速
# tsne = TSNE(n_components=2)
# data_tsne = tsne.fit_transform(feature_extractor.data)
# print(data_tsne.shape)
# # print(data_tsne)
# # 展示降维后的数据
# import matplotlib.pyplot as plt
#
# # res将255转换为1
# # res = res.astype(np.uint8)
# res = res // 255
# plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=res.reshape(-1), alpha=0.10)
# print(data_tsne[:, 0])
# plt.show()

# manifold_plot = ManifoldPlot(train_data, train_label, data)
# manifold_plot.tsne_plot()
# manifold_plot.isomap_plot()
# manifold_plot.lle_plot()
