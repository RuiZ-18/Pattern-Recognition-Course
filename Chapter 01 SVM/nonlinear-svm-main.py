from utils import linear_svm, mouse_callback
import os
import shutil
import cv2
import numpy as np
from sklearn.svm import SVC


output_dir = "output_non_linear"
data_dir = "data"
img_name_list = ["0618.png", "0854.png", "1066.png"]
img_name = img_name_list[0]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, img_name[:-4])
img = cv2.imread(os.path.join(data_dir, img_name))
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

shutil.copy(os.path.join(data_dir, img_name), output_dir + "/" + img_name)
img_h, img_w, img_c = img.shape
sample_img = img.copy()

positive_coordinates = []
negative_coordinates = []
param = {"positive": positive_coordinates, "negative": negative_coordinates, "img": sample_img}

cv2.namedWindow("select samples")
cv2.setMouseCallback("select samples", lambda event, x, y, flags, param: mouse_callback(event, x, y, flags, param), param)
while 1:
    cv2.imshow("select samples", sample_img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()
print("positive_sample", positive_coordinates)
print("negative_sample", negative_coordinates)
sample_indices = []
train_label = []
for coor in positive_coordinates:
    sample_indices.append(coor[0] + coor[1]*img_w)
    train_label.append(1)
for coor in negative_coordinates:
    sample_indices.append(coor[0] + coor[1]*img_w)
    # train_label.append(-1)
    train_label.append(0)
cv2.imwrite(output_dir + "/" + img_name[:-4] + "_sample_ori_" + "{:02d}".format(len(sample_indices)) + img_name[-4:], sample_img)

data = np.array(img.copy(), dtype=int).reshape(img_h * img_w, img_c)
# label = np.array(label_img, dtype=int).reshape(img_h * img_w)
train_data = data[sample_indices]

# 高斯核函数
C = [1.0, 10.0, 100.0, 1000.0]
for cc in (C):
    svm_classifier = SVC(kernel='rbf', gamma='auto', C=cc)
    svm_classifier.fit(train_data, train_label)
    output = svm_classifier.predict(data)
    output_img = output.copy().reshape((img_h, img_w))
    output_img[output_img > 0] = 255
    # output_img[output_img < 0] = 0liy
    output_img = np.array(output_img, dtype=np.uint8)
    save_name = output_dir + "/" + img_name[:-4] + "svm_seg_nonlinear" + "{:02d}_{:.1f}".format(
        len(sample_indices), cc) + img_name[-4:]
    cv2.imwrite(save_name, output_img)
    # while 1:
    #     cv2.imshow("svm_seg_nonlinear", output_img)
    #     if cv2.waitKey(0) & 0xFF == 27:
    #         break
    # cv2.destroyAllWindows()

# # 高斯
# svm_classifier = SVC(kernel='rbf', gamma='auto')
# svm_classifier.fit(train_data, train_label)
# output = svm_classifier.predict(data)
# output_img = output.copy().reshape((img_h, img_w))
# output_img[output_img > 0] = 255
# # output_img[output_img < 0] = 0
# output_img = np.array(output_img, dtype=np.uint8)
# save_name = output_dir + "/" + img_name[:-4] + "svm_seg_nonlinear" + "{:02d}".format(
#     len(sample_indices)) + img_name[-4:]
# cv2.imwrite(save_name, output_img)
# while 1:
#     cv2.imshow("svm_seg_nonlinear", output_img)
#     if cv2.waitKey(0) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()