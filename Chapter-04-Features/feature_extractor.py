from sklearn.decomposition import PCA, KernelPCA, MiniBatchDictionaryLearning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# pca
def pca_feature(train_data, data, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    data = pca.transform(data)
    return train_data, data


# kpca
def kpca_feature(train_data, data, n_components=1, kernel='rbf'):
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    kpca.fit(train_data)
    train_data = kpca.transform(train_data)
    data = kpca.transform(data)
    return train_data, data


# lda
# ValueError: n_components cannot be larger than min(n_features, n_classes - 1).
def lda_feature(train_data, train_label, data, n_components=1):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(train_data, train_label)
    train_data = lda.transform(train_data)
    data = lda.transform(data)
    return train_data, data


# 字典学习
def dictionary_learning_feature(train_data, data, n_components=1):
    dl = MiniBatchDictionaryLearning(n_components=n_components)
    dl.fit(train_data)
    train_data = dl.transform(train_data)
    data = dl.transform(data)
    return train_data, data


# 定义一个类，把所有的特征提取方法都封装起来
class FeatureExtractor:
    def __init__(self, train_data, train_label, data):
        self.td = train_data
        self.d = data
        self.train_data = None
        self.data = None
        self.train_label = train_label

    def original_feature(self):
        self.train_data = self.td
        self.data = self.d

    def pca_feature(self, n_components=2):
        pca = PCA(n_components=n_components)
        pca.fit(self.td)
        self.train_data = pca.transform(self.td)
        self.data = pca.transform(self.d)

    def kpca_feature(self, n_components=1, kernel='rbf'):
        kpca = KernelPCA(n_components=n_components, kernel=kernel)
        kpca.fit(self.td)
        self.train_data = kpca.transform(self.td)
        self.data = kpca.transform(self.d)

    def lda_feature(self, n_components=1):
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(self.td, self.train_label)
        self.train_data = lda.transform(self.td)
        self.data = lda.transform(self.d)

    def dictionary_learning_feature(self, n_components=1):
        dl = MiniBatchDictionaryLearning(n_components=n_components)
        dl.fit(self.td)
        self.train_data = dl.transform(self.td)
        self.data = dl.transform(self.d)
