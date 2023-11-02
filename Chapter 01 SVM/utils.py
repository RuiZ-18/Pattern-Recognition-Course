import numpy as np
from qpsolvers import solve_qp
from cvxopt import matrix, solvers
import cv2


def linear_svm(train_data, train_label):
    train_data_num, train_data_dim = train_data.shape
    # 矩阵P
    P = np.zeros((train_data_dim + 1, train_data_dim + 1))
    for i in range(train_data_dim):
        for j in range(train_data_dim):
            if i == j:
                P[i, j] = 1
    # 矩阵q
    q = np.zeros((train_data_dim + 1, 1))
    # 矩阵G
    G = np.zeros((train_data_num, train_data_dim + 1))
    for i in range(train_data_num):
        G[i, :] = np.hstack((-train_label[i] * train_data[i, :], -train_label[i]))
    # 矩阵h
    h = - np.ones((train_data_num, 1))
    # 计算结果
    X = solve_qp(P, q, G, h, solver='cvxopt')
    w, b0 = X[:train_data_dim], X[-1]
    return w, b0


def linear_svm_dual(train_data, train_label):
    train_data_num, train_data_dim = train_data.shape

    P = np.outer(train_label, train_label) * np.dot(train_data, train_data.T)
    q = - np.ones((train_data_num, 1))
    # G = np.vstack([-np.eye(train_data_num), np.eye(train_data_num)])
    G = - np.eye(train_data_num)
    h = np.zeros((train_data_num, 1))
    A = train_label.astype(float).reshape(1, -1)
    b = np.double(0)
    lb = np.zeros(train_data_num)
    ub = np.full(train_data_num, np.inf)
    a0 = np.zeros(train_data_num)
    # X = solve_qp(P=P, q=q, G=G, A=Aeq, b=beq, lb=lb, ub=ub, solver='cvxopt')
    # X = solve_qp(P, q, G, h, A, b, solver='cvxopt')
    Pm = matrix(P, tc='d')
    qm = matrix(q, tc='d')
    Gm = matrix(G, tc='d')
    hm = matrix(h, tc='d')
    Am = matrix(A, tc='d')
    bm = matrix(b, tc='d')
    result = solvers.qp(Pm, qm, Gm, hm, Am, bm)

    alphas = np.array(result['x'])

    # 提取支持向量
    threshold = 1e-5  # 定义一个阈值来判断是否为支持向量
    sv_indices = np.where(alphas > threshold)[0]
    support_vectors = train_data[sv_indices]
    support_vector_labels = train_label[sv_indices]

    # 计算权重向量w和截距b
    w = np.sum(alphas * train_label.reshape(-1, 1) * train_data, axis=0)
    b = np.mean(train_label - np.dot(train_data, w))

    return w, b



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
