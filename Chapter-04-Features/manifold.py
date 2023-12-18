import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding


def tsne_plot(train_data, train_label, n_components=2, perplexity=10):
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_train_data = tsne.fit_transform(train_data)
    for i in range(tsne_train_data.shape[0]):
        if train_label[i] == 1:
            plt.scatter(tsne_train_data[i, 0], tsne_train_data[i, 1], c='r')
        else:
            plt.scatter(tsne_train_data[i, 0], tsne_train_data[i, 1], c='b')
    plt.show()


def isomap_plot(train_data, train_label, n_components=2, n_neighbors=5):
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    isomap_train_data = isomap.fit_transform(train_data)
    for i in range(isomap_train_data.shape[0]):
        if train_label[i] == 1:
            plt.scatter(isomap_train_data[i, 0], isomap_train_data[i, 1], c='r')
        else:
            plt.scatter(isomap_train_data[i, 0], isomap_train_data[i, 1], c='b')
    plt.show()


def lle_plot(train_data, train_label, n_components=2, n_neighbors=5):
    lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    lle_train_data = lle.fit_transform(train_data)
    for i in range(lle_train_data.shape[0]):
        if train_label[i] == 1:
            plt.scatter(lle_train_data[i, 0], lle_train_data[i, 1], c='r')
        else:
            plt.scatter(lle_train_data[i, 0], lle_train_data[i, 1], c='b')
    plt.show()


# 定义一个类，把所有的降维方法都封装起来
def save_fig(fig_name):
    plt.savefig(fig_name)
    plt.close()


class ManifoldPlot:
    def __init__(self, train_data, train_label, data):
        self.train_data = train_data
        self.train_label = train_label
        self.data = data

    def tsne_plot(self, filename, n_components=2, perplexity=10):
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        tsne_train_data = tsne.fit_transform(self.train_data)
        plt.scatter(tsne_train_data[:, 0], tsne_train_data[:, 1], c=self.train_label, alpha=0.1)
        plt.savefig(filename)

    def isomap_plot(self, filename, n_components=2, n_neighbors=5):
        isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        isomap_train_data = isomap.fit_transform(self.train_data)
        plt.scatter(isomap_train_data[:, 0], isomap_train_data[:, 1], c=self.train_label, alpha=0.1)
        plt.savefig(filename)
        # plt.show()

    def lle_plot(self, filename, n_components=2, n_neighbors=5):
        lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
        lle_train_data = lle.fit_transform(self.train_data)
        plt.scatter(lle_train_data[:, 0], lle_train_data[:, 1], c=self.train_label, alpha=0.1)
        plt.savefig(filename)


