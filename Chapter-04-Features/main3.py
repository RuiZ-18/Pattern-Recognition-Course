import cv2
import os
import numpy as np
from utils import auto_sample, sample, evaluator
from feature_extractor import FeatureExtractor
from classifier import Classifier
from manifold import ManifoldPlot
from image_processing import ImageProcess
import warnings

warnings.filterwarnings("ignore")
# from tsnecuda import TSNE
import matplotlib.pyplot as plt

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
        dictoinary_learning_feature=[dict({'n_components': i}) for i in range(1, 3)]
    )

    # 特征提取，需要把feature_extractor_params传入
    for feature_extractor_item in feature_extractor_list:
        for feature_extractor_param in feature_extractor_params[feature_extractor_item]:
            feature_extractor.__getattribute__(feature_extractor_item)(**feature_extractor_param)

            # 3.流形学习，包括tsne、isomap、lle
            manifold_plot = ManifoldPlot(train_data, train_label, data)
            manifold_plot_list = ['tsne_plot', 'isomap_plot', 'lle_plot']
            manifold_plot_list = ['isomap_plot', 'lle_plot']
            manifold_plot_params = dict(
                tsne_plot=[dict({'filename': os.path.join(output_dir, "2_manifold", filter_item,
                                                          "tsne_plot_perplexity_" + str(j) + "_" + img_name),
                                 'n_components': 2, 'perplexity': j}) for j in range(10, 100, 10)],
                isomap_plot=[dict({'filename': os.path.join(output_dir, "2_manifold", filter_item,
                                                            "isomap_plot_n_neighbors_" + str(j) + "_" + img_name),
                                   'n_components': 2, 'n_neighbors': j}) for j in range(1, 10)],
                lle_plot=[dict({'filename': os.path.join(output_dir, "2_manifold", filter_item,
                                                         "lle_plot_n_neighbors_" + str(j) + "_" + img_name),
                                'n_components': 2, 'n_neighbors': j}) for j in range(1, 10)]
            )

            for manifold_plot_item in manifold_plot_list:
                for manifold_plot_param in manifold_plot_params[manifold_plot_item]:
                    manifold_plot.__getattribute__(manifold_plot_item)(**manifold_plot_param)

            # 4.分类器，包括svm、随机森林、knn
            classifier = Classifier(feature_extractor)
            classifier_list = ['svm_clf', 'random_forest_clf', 'knn_clf']
            classifier_params = dict(
                svm_clf=[dict({'filename': os.path.join(output_dir, "3_classifier", filter_item,
                                                        "svm_" + i + "_" + img_name),
                               'kernel': i}) for i in ['rbf', 'poly', 'sigmoid']],
                random_forest_clf=[dict({'filename': os.path.join(output_dir, "3_classifier", filter_item,
                                                                  "random_forest_n_estimators" + str(
                                                                      i) + "_" + img_name),
                                         'n_estimators': i}) for i in range(1, 10)],
                knn_clf=[dict({'filename': os.path.join(output_dir, "3_classifier", filter_item,
                                                        "random_forest_n_neighbors" + str(
                                                            i) + "_" + img_name),
                               'n_neighbors': i}) for i in range(1, 10)]
            )
            for classifier_item in classifier_list:
                for classifier_param in classifier_params[classifier_item]:
                    classifier.__getattribute__(classifier_item)(**classifier_param)
                    res = classifier.res
                    print(evaluator(res, img_mask))

# print(feature_extractor_item)

# for feature_extractor_item in feature_extractor_list:
#     feature_extractor.__getattribute__(feature_extractor_item)()
# print(feature_extractor_item)

# feature_extractor = FeatureExtractor(train_data, train_label, data)
# feature_extractor.original_feature()
# feature_extractor.kpca_feature(n_components=5, kernel='poly')


# classifier = Classifier(feature_extractor)
# classifier.knn_clf()
# res = classifier.res


# cv2.imshow("res", res)
# 对res做闭运算


# # 开运算，消除小物体
# for _ in range(1):
#     kernel = np.ones((5, 5), np.uint8)
#     res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
# #
# # 闭运算，消除小空洞
# for _ in range(2):
#     kernel = np.ones((10, 10), np.uint8)
#     res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

# plt.imshow(res)
# plt.show()
# # cv2.waitKey(0)
# print(evaluator(res, img_mask))

# manifold_plot = ManifoldPlot(train_data, train_label, data)
# manifold_plot.tsne_plot()
# manifold_plot.isomap_plot()
# manifold_plot.lle_plot()
