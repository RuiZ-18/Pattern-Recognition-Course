import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from image_processing import ImageProcess


# 读取output/0618classifiers/mean_shift_filter/original_feature下的三个图片
img_name_list = ["0618.png", "0854.png", "1066.png"]
# img_name = img_name_list[0]
output_dir = 'output'
image_process_dir = 'mean_shift_filter'
original_feature_dir = 'original_feature'
clf_name = ['svm', 'random_forest', 'knn']

output_dir_new = os.path.join(output_dir, "4_post_processing")
# 如果存在output_dir_new，就删除
if os.path.exists(output_dir_new):
    os.system("rm -rf {}".format(output_dir_new))
os.mkdir(output_dir_new)

for img_name in img_name_list:
    # 在output_dir_new下创建文件夹img_name[:-4]
    output_dir_new_img = os.path.join(output_dir_new, img_name[:-4])
    if os.path.exists(output_dir_new_img):
        os.system("rm -rf {}".format(output_dir_new_img))
    os.mkdir(output_dir_new_img)

    for i in range(3):
        # 在output_dir_new_img下创建三个文件夹，分别存放svm、random_forest、knn的后处理结果
        output_dir_new_clf = os.path.join(output_dir_new_img, clf_name[i])
        if os.path.exists(output_dir_new_clf):
            os.system("rm -rf {}".format(output_dir_new_clf))
        os.mkdir(output_dir_new_clf)

        # 读取output/0618classifiers/mean_shift_filter/original_feature下的三个图片
        original_feature_img_name = 'feature_1_' + clf_name[i] + '.png'
        original_feature_img = cv2.imread(os.path.join(output_dir, img_name[:-4] + 'classifiers', image_process_dir, original_feature_dir, original_feature_img_name))
        # 数据后处理
        image_process = ImageProcess(original_feature_img)
        image_process.opening(os.path.join(output_dir_new_clf, 'opening.png'))
        image_process.closing(os.path.join(output_dir_new_clf, 'closing.png'))
        image_process.opening_closing(os.path.join(output_dir_new_clf, 'opening_closing.png'))
        image_process.closing_opening(os.path.join(output_dir_new_clf, 'closing_opening.png'))





