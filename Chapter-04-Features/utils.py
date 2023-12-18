import cv2
import os
import numpy as np
import shutil


def auto_sample(data_dir, img_name, output_dir, sample_interval_list):
    for sample_interval in [sample_interval_list]:
        print(f'sample_interval: {sample_interval}')
        img_label = np.load(os.path.join(data_dir, img_name[:-4] + "_label.npy"))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_dir = os.path.join(output_dir, img_name[:-4] + "_{}".format(sample_interval))
        img = cv2.imread(os.path.join(data_dir, img_name))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        shutil.copy(os.path.join(data_dir, img_name), output_dir + "/" + img_name)
        img_h, img_w, img_c = img.shape
        sample_img = img.copy()

        positive_coordinates = []
        negative_coordinates = []
        for i in range(img_h):
            for j in range(img_w):
                if i % sample_interval == 0 and j % sample_interval == 0:
                    if img_label[i, j] == 1:
                        positive_coordinates.append((i, j))
                        cv2.circle(sample_img, (j, i), 1, (0, 0, 255), thickness=5)
                    else:
                        negative_coordinates.append((i, j))
                        cv2.circle(sample_img, (j, i), 1, (255, 0, 0), thickness=5)

        cv2.destroyAllWindows()
        print("positive_sample", positive_coordinates)
        print("negative_sample", negative_coordinates)
        sample_indices = []
        train_label = []
        for coor in positive_coordinates:
            sample_indices.append(coor[0]*img_w + coor[1])
            train_label.append(1)
        for coor in negative_coordinates:
            sample_indices.append(coor[0]*img_w + coor[1])
            # train_label.append(-1)
            train_label.append(0)
        cv2.imwrite(
            output_dir + "/" + img_name[:-4] + "_sample" + img_name[-4:], sample_img)
        data = np.array(img.copy(), dtype=int).reshape(img_h * img_w, img_c)
        # label = np.array(label_img, dtype=int).reshape(img_h * img_w)
        train_data = data[sample_indices]
        train_label = np.array(train_label)
        return train_data, train_label


# 采样函数，与auto_sample函数采样部分功能相同，但是不需要读取mask，不需要保存图像，返回值同样为train_data和train_label
def sample(data_dir, img_name, output_dir_filter, filter_img_name, sample_interval):
    img_label = np.load(os.path.join(data_dir, img_name[:-4] + "_label.npy"))
    label = img_label.reshape(-1)
    img = cv2.imread(os.path.join(output_dir_filter, filter_img_name))
    img_h, img_w, img_c = img.shape
    sample_img = img.copy()

    positive_coordinates = []
    negative_coordinates = []
    for i in range(img_h):
        for j in range(img_w):
            if i % sample_interval == 0 and j % sample_interval == 0:
                if img_label[i, j] == 1:
                    positive_coordinates.append((i, j))
                else:
                    negative_coordinates.append((i, j))
    print("positive_sample", positive_coordinates)
    print("negative_sample", negative_coordinates)
    sample_indices = []
    train_label = []
    for coor in positive_coordinates:
        sample_indices.append(coor[0]*img_w + coor[1])
        train_label.append(1)
    for coor in negative_coordinates:
        sample_indices.append(coor[0]*img_w + coor[1])
        train_label.append(0)
    data = np.array(img.copy(), dtype=int).reshape(img_h * img_w, img_c)
    # label = np.array(label_img, dtype=int).reshape(img_h * img_w)
    train_data = data[sample_indices]
    train_label = np.array(train_label)
    return train_data, train_label, data, label


def evaluator(img, img_mask):
    # img_mask为图像真值，img为预测的二值图像，计算准确率，精确率，召回率，F1值，返回一个字典
    img = img.astype(np.uint8)
    img_mask = img_mask.astype(np.uint8)
    img_h, img_w = img.shape
    img = img.reshape(-1)
    img_mask = img_mask.reshape(-1)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(img_h*img_w):
        if img_mask[i] == 255:
            if img[i] == 255:
                TP += 1
            else:
                FN += 1
        else:
            if img[i] == 255:
                FP += 1
            else:
                TN += 1
    # 如果TP+FP=0，说明预测的图像中没有白色像素，此时精确率为1，召回率为0
    if TP+FP == 0:
        precision = 1
        recall = 0
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "F1": F1}
