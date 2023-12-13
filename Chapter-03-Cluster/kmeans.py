import os
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")
import numpy as np

matplotlib.use('TkAgg')

output_dir = "output"
data_dir = "data"
img_name_list = ["0618.png", "0854.png", "1066.png"]

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

img_list = []
for img_name in img_name_list:
    img = cv2.imread(os.path.join(data_dir, img_name))
    img_list.append(img)

def get_label(img, n_clusters, init='random'):
    img_h, img_w, img_c = img.shape
    img_data = img.reshape(img_h * img_w, img_c)
    cluster = KMeans(n_clusters=n_clusters, init=init).fit(img_data)
    label = cluster.labels_
    label = label.reshape(img_h, img_w)
    return label

for n_clusters in tqdm(range(2, 10)):

    # cluster = DBSCAN(eps=0.8, min_samples=2).fit(img_data)
    # cluster = KMeans(n_clusters=n_clusters, init='random').fit(img_data)
    # cluster = KMeans(n_clusters=n_clusters, init='k-means++').fit(img_data)
    # label = cluster.labels_
    #
    # label = label.reshape(img_h, img_w)



    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
    plt.xticks([])
    plt.yticks([])

    init = 'random'
    x = get_label(img_list[0], n_clusters, init)
    axs[0].imshow(x, 'viridis')
    # axs[0].imshow(get_label(img_list[0], n_clusters, init), 'viridis')
    axs[1].imshow(get_label(img_list[1], n_clusters, init), 'viridis')
    axs[2].imshow(get_label(img_list[2], n_clusters, init), 'viridis')

    plt.savefig(f"{output_dir}/kmeans/{n_clusters:02d}_{init}.png")
    # plt.show()
    # break

