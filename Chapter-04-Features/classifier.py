import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import cv2


# 写一个类，把所有的分类器都封装起来
class Classifier:
    def __init__(self, feature_extractor, img_h=250, img_w=500):
        self.train_data = feature_extractor.train_data
        self.train_label = feature_extractor.train_label
        self.data = feature_extractor.data
        self.img_h = img_h
        self.img_w = img_w
        self.res = None

    def svm_clf(self, filename, kernel='poly'):
        clf = svm.SVC(kernel=kernel)
        clf.fit(self.train_data, self.train_label)
        self.res = (clf.predict(self.data).reshape(self.img_h, self.img_w) * 255).astype(np.uint8)
        cv2.imwrite(filename, self.res)

    def random_forest_clf(self, filename, n_estimators=10):
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(self.train_data, self.train_label)
        self.res = (clf.predict(self.data).reshape(self.img_h, self.img_w) * 255).astype(np.uint8)
        cv2.imwrite(filename, self.res)

    def knn_clf(self, filename, n_neighbors=5):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(self.train_data, self.train_label)
        self.res = (clf.predict(self.data).reshape(self.img_h, self.img_w) * 255).astype(np.uint8)
        cv2.imwrite(filename, self.res)
