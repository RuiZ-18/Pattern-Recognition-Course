import numpy as np
from qpsolvers import solve_qp
from cvxopt import matrix, solvers
import cv2
import os
import shutil
from tqdm import tqdm
from scipy.stats import multivariate_normal


def mouse_callback(event, x, y, flags, param):
    img = param["img"]
    if event == cv2.EVENT_LBUTTONDOWN:
        positive = param["positive"]
        xy = "%d,%d,%d" % (x, y, 1)
        print(x, y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=5)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_TRIPLEX,
                    0.5, (0, 0, 0), thickness=1)
        cv2.imshow("select samples", img)
        positive.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        negative = param["negative"]
        xy = "%d,%d,%d" % (x, y, -1)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=5)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_TRIPLEX,
                    0.5, (0, 0, 0), thickness=1)
        cv2.imshow("select samples", img)
        negative.append([x, y])


def manual_sample(img, data_dir, img_name, output_dir):
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
    cv2.setMouseCallback("select samples", lambda event, x, y, flags, param: mouse_callback(event, x, y, flags, param),
                         param)
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
        sample_indices.append(coor[0] + coor[1] * img_w)
        train_label.append(1)
    for coor in negative_coordinates:
        sample_indices.append(coor[0] + coor[1] * img_w)
        train_label.append(-1)

    cv2.imwrite(output_dir + "/" + img_name[:-4] + "_sample_" + "{:02d}".format(len(sample_indices)) + img_name[-4:],
                sample_img)

    train_label = np.array(train_label).reshape((len(train_label), 1))
    # sample_indices = [7665, 75300, 39300, 29593]
    # train_label = np.array([1, 1, 1, -1])

    data = np.array(img.copy(), dtype=int).reshape(img_h * img_w, img_c)
    # label = np.array(label_img, dtype=int).reshape(img_h * img_w)
    train_data = data[sample_indices]

    return train_data, train_label


def auto_sample(data_dir, img_name, output_dir_, sample_interval_list):
    for sample_interval in [sample_interval_list]:
        print(sample_interval)
        img_label = np.load(os.path.join(data_dir, img_name[:-4] + "_label.npy"))
        if not os.path.exists(output_dir_):
            os.mkdir(output_dir_)

        output_dir = os.path.join(output_dir_, img_name[:-4] + "_{}".format(sample_interval))
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
                    # print(i, j)
                    if img_label[i, j] == 1:
                        positive_coordinates.append((j, i))
                        cv2.circle(sample_img, (j, i), 1, (0, 0, 255), thickness=5)
                    else:
                        negative_coordinates.append((j, i))
                        cv2.circle(sample_img, (j, i), 1, (255, 0, 0), thickness=5)

        cv2.destroyAllWindows()
        print("positive_sample", positive_coordinates)
        print("negative_sample", negative_coordinates)
        sample_indices = []
        train_label = []
        for coor in positive_coordinates:
            sample_indices.append(coor[0] + coor[1] * img_w)
            train_label.append(1)
        for coor in negative_coordinates:
            sample_indices.append(coor[0] + coor[1] * img_w)
            # train_label.append(-1)
            train_label.append(0)
        cv2.imwrite(
            output_dir + "/" + img_name[:-4] + "_sample" + img_name[-4:],
            sample_img)
        data = np.array(img.copy(), dtype=int).reshape(img_h * img_w, img_c)
        # label = np.array(label_img, dtype=int).reshape(img_h * img_w)
        train_data = data[sample_indices]
        train_label = np.array(train_label)
        return train_data, train_label


def bayes(img, P_1, P_2, RGB1_m, RGB2_m, RGB1_cov, RGB2_cov):
    img_h, img_w = img.shape[0], img.shape[1]
    resImage = np.zeros((img_h, img_w))
    grayImage = np.zeros((img_h, img_w))
    grayImage2 = np.zeros((img_h, img_w))
    set_flag = 0
    for i in tqdm(range(2, img_h - 1)):
        for j in range(2, img_w - 1):
            set_flag = 0
            num_pos = 0
            num_neg = 0
            for k in range(i - 1, i + 2):
                for s in range(j - 1, j + 2):
                    # if 1:
                    p1 = P_1 * multivariate_normal.pdf(img[k, s, :].reshape(1, 3), RGB1_m, RGB1_cov)
                    p2 = P_2 * multivariate_normal.pdf(img[k, s, :].reshape(1, 3), RGB2_m, RGB2_cov)
                    if p1 > p2:
                        num_pos += 1
                    else:
                        num_neg += 1
                if num_pos > 4:
                    for k in range(i - 1, i + 2):
                        for s in range(j - 1, j + 2):
                            resImage[k, s] = 255
                            try:
                                grayImage[k, s] = int(255 * p2 / (p1 + p2))
                                grayImage2[k, s] = int(num_pos * 255 / 9)
                            except:
                                grayImage[k, s] = 1
                                grayImage2[k, s] = 1
                            set_flag = 1
                elif num_neg > 4:
                    for k in range(i - 2, i + 2):
                        for s in range(j - 1, j + 1):
                            resImage[k, s] = 0
                            try:
                                grayImage[k, s] = int(255 * p2 / (p1 + p2))
                                grayImage2[k, s] = int(num_pos * 255 / 9)
                            except:
                                grayImage[k, s] = 0
                                grayImage2[k, s] = 0
                            set_flag = 1
                if set_flag:
                    break
            # 3 * 3 blocks
            # if num_pos > 4:
            #     for k in range(i - 1, i + 2):
            #         for s in range(j - 1, j + 2):
            #             resImage[k, s] = 255
            # elif num_neg > 4:
            #     for k in range(i - 2, i + 2):
            #         for s in range(j - 1, j + 1):
            #             resImage[k, s] = 0
    return resImage, grayImage, grayImage2


def bayes_smo(img, train_data, train_label):
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
    img_h, img_w = img.shape[0], img.shape[1]
    resImage = np.zeros((img_h, img_w))
    for i in tqdm(range(2, img_h - 1)):
        # set_flag = 0
        for j in range(2, img_w - 1):
            num_pos = 0
            num_neg = 0
            set_flag = 0
            for k in range(i - 1, i + 2):
                for s in range(j - 1, j + 2):
                    # if 1:
                    if (P_1 * multivariate_normal.pdf(img[k, s, :].reshape(1, 3), RGB1_m, RGB1_cov) >
                            P_2 * multivariate_normal.pdf(img[k, s, :].reshape(1, 3), RGB2_m, RGB2_cov)):
                        num_pos += 1
                    else:
                        num_neg += 1
                # 3 * 3 blocks
                if num_pos > 4:
                    for k in range(i - 1, i + 2):
                        for s in range(j - 1, j + 2):
                            resImage[k, s] = 255
                            set_flag = 1
                elif num_neg > 4:
                    for k in range(i - 2, i + 2):
                        for s in range(j - 1, j + 1):
                            resImage[k, s] = 0
                            set_flag = 2
                if set_flag == 1:
                    # train_data
                    # 更新cov，m

                    if train_data.shape[0] > 1500:
                        break
                    train_data = np.vstack((train_data, img[i - 1, j - 1]))
                    train_label = np.hstack((train_label, int(resImage[i - 1, j - 1] / 255)))
                    RGB1 = train_data[train_label == 1]
                    RGB2 = train_data[~(train_label == 1)]
                    RGB1_m = np.mean(RGB1, axis=0)
                    RGB2_m = np.mean(RGB2, axis=0)
                    RGB1_cov = np.cov(RGB1.T) / (RGB1.shape[0] - 1)
                    RGB2_cov = np.cov(RGB2.T) / (RGB2.shape[0] - 1)

                    # img_h, img_w = img.shape[0], img.shape[1]
                    break

                elif set_flag == 2:
                    break
            # 3 * 3 blocks
            # if num_pos > 4:
            #     for k in range(i - 1, i + 2):
            #         for s in range(j - 1, j + 2):
            #             resImage[k, s] = 255
            # elif num_neg > 4:
            #     for k in range(i - 2, i + 2):
            #         for s in range(j - 1, j + 1):
            #             resImage[k, s] = 0
    print(train_data.shape[0])
    return resImage


def bayes_smo2(img, train_data, train_label):
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
    img_h, img_w = img.shape[0], img.shape[1]
    resImage = np.zeros((img_h, img_w))
    for i in tqdm(range(2, img_h - 1)):
        # set_flag = 0
        for j in range(2, img_w - 1):
            num_pos = 0
            num_neg = 0
            set_flag = 0
            for k in range(i - 1, i + 2):
                for s in range(j - 1, j + 2):
                    # if 1:
                    if (P_1 * multivariate_normal.pdf(img[k, s, :].reshape(1, 3), RGB1_m, RGB1_cov) >
                            P_2 * multivariate_normal.pdf(img[k, s, :].reshape(1, 3), RGB2_m, RGB2_cov)):
                        num_pos += 1
                    else:
                        num_neg += 1
                # 3 * 3 blocks
                # if num_pos > 4:
                #     for k in range(i - 1, i + 2):
                #         for s in range(j - 1, j + 2):
                #             resImage[k, s] = 255
                #             set_flag = 1
                # elif num_neg > 4:
                #     for k in range(i - 2, i + 2):
                #         for s in range(j - 1, j + 1):
                #             resImage[k, s] = 0
                #             set_flag = 1
                # if set_flag:
                #     # train_data
                #     # 更新cov，m
                #
                #     if train_data.shape[0] > 1500:
                #         break
                #     train_data = np.vstack((train_data, img[i-1, j-1]))
                #     train_label = np.hstack((train_label, int(resImage[i-1, j-1]/255)))
                #     RGB1 = train_data[train_label == 1]
                #     RGB2 = train_data[~(train_label == 1)]
                #     RGB1_m = np.mean(RGB1, axis=0)
                #     RGB2_m = np.mean(RGB2, axis=0)
                #     RGB1_cov = np.cov(RGB1.T) / (RGB1.shape[0] - 1)
                #     RGB2_cov = np.cov(RGB2.T) / (RGB2.shape[0] - 1)
                #
                #     # img_h, img_w = img.shape[0], img.shape[1]
                #     break
            # 3 * 3 blocks
            if num_pos > 4:
                for k in range(i - 1, i + 2):
                    for s in range(j - 1, j + 2):
                        resImage[k, s] = 255
            elif num_neg > 4:
                for k in range(i - 2, i + 2):
                    for s in range(j - 1, j + 1):
                        resImage[k, s] = 0
            if train_data.shape[0] > 200 or num_pos < 8:
                continue
            train_data = np.vstack((train_data, img[i - 1, j - 1]))
            train_label = np.hstack((train_label, int(resImage[i - 1, j - 1] / 255)))
            RGB1 = train_data[train_label == 1]
            RGB2 = train_data[~(train_label == 1)]
            RGB1_m = np.mean(RGB1, axis=0)
            RGB2_m = np.mean(RGB2, axis=0)
            RGB1_cov = np.cov(RGB1.T) / (RGB1.shape[0] - 1)
            RGB2_cov = np.cov(RGB2.T) / (RGB2.shape[0] - 1)
    print(train_data.shape[0])
    return resImage


def acc_cal(img, img_mask):
    total = img.shape[0] * img.shape[1]
    same = (img == img_mask).sum()
    return same * 1.0 / total
