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


def get_label(img, eps, min_samples):
    img_h, img_w, img_c = img.shape
    img_data = img.reshape(img_h * img_w, img_c)
    # cluster = KMeans(n_clusters=n_clusters, init=init).fit(img_data)
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(img_data)
    label = cluster.labels_
    label = label.reshape(img_h, img_w)
    return label


# for eps in tqdm([x for x in np.arange(0.9, 1.0, 0.1)]):
for eps in tqdm([x for x in np.arange(0.9, 1.5, 0.1)]):
    for min_samples in [x for x in range(1, 10)]:
        fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
        plt.xticks([])
        plt.yticks([])

        axs[0].imshow(get_label(img_list[0], eps, min_samples), 'viridis')
        axs[1].imshow(get_label(img_list[1], eps, min_samples), 'viridis')
        axs[2].imshow(get_label(img_list[2], eps, min_samples), 'viridis')

        plt.savefig(f"{output_dir}/dbscan/{eps:.2f}_{min_samples:02d}.png")
        # plt.show()
        # break
