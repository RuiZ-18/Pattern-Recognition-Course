import numpy as np
from sklearn import svm
from feature_extractor import FeatureExtractor


def svm_clf(train_data, train_label, data, kernel='poly', img_h=250, img_w=500):
    clf = svm.SVC(kernel=kernel)
    clf.fit(train_data, train_label)
    res = clf.predict(data).reshape(img_h, img_w) * 255
    res = res.astype(np.uint8)
    return res


# 使用随机树
def random_forest_clf(train_data, train_label, data, img_h=250, img_w=500):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(train_data, train_label)
    res = clf.predict(data).reshape(img_h, img_w) * 255
    res = res.astype(np.uint8)
    return res


# 使用knn
def knn_clf(train_data, train_label, data, img_h=250, img_w=500):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(train_data, train_label)
    res = clf.predict(data).reshape(img_h, img_w) * 255
    res = res.astype(np.uint8)
    return res


# 写一个类，把所有的分类器都封装起来
class Classifier:
    def __init__(self, feature_extractor, img_h=250, img_w=500):
        self.train_data = feature_extractor.train_data
        self.train_label = feature_extractor.train_label
        self.data = feature_extractor.data
        self.img_h = img_h
        self.img_w = img_w
        self.res = None

    def svm_clf(self, kernel='poly'):
        clf = svm.SVC(kernel=kernel)
        clf.fit(self.train_data, self.train_label)
        self.res = (clf.predict(self.data).reshape(self.img_h, self.img_w) * 255).astype(np.uint8)

    def random_forest_clf(self):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(self.train_data, self.train_label)
        self.res = (clf.predict(self.data).reshape(self.img_h, self.img_w) * 255).astype(np.uint8)

    def knn_clf(self):
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(self.train_data, self.train_label)
        self.res = (clf.predict(self.data).reshape(self.img_h, self.img_w) * 255).astype(np.uint8)
