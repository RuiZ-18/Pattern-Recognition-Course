import cv2
import os
import numpy as np
from utils import auto_sample, evaluator
from feature_extractor import FeatureExtractor
from classifier import Classifier
from manifold import ManifoldPlot

data_dir = "data"
output_dir = 'output'
img_name_list = ["0618.png", "0854.png", "1066.png"]
img_name = img_name_list[0]
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


img_mask = cv2.imread(os.path.join(data_dir, img_name[:-4] + "_mask.png"), flags=0)
data = img.reshape(-1, 3)

sample_interval = 30
train_data, train_label = auto_sample(data_dir, img_name, output_dir, sample_interval)


feature_extractor = FeatureExtractor(train_data, train_label, data)
feature_extractor.original_feature()
feature_extractor.pca_feature(n_components=2)

classifier = Classifier(feature_extractor)
classifier.knn_clf()
res = classifier.res
cv2.imshow("res", res)
cv2.waitKey(0)
print(evaluator(res, img_mask))

manifold_plot = ManifoldPlot(train_data, train_label, data)
manifold_plot.tsne_plot()
manifold_plot.isomap_plot()
manifold_plot.lle_plot()