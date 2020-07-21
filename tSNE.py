from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

def get_data():
    digits = datasets.load_digits(n_class=6)  # 0-5
    data = digits.data  # data.shape=[1083,64]
    label = digits.target
    n_samples, n_features = data.shape  # 1083, 64
    return data, label, n_samples, n_features

def plot_embedding(result, label, title):  # [1083,2] [1083]
    x_min, x_max = np.min(result, 0), np.max(result, 0)
    data = (result - x_min) / (x_max - x_min)  # [0-1] scale
    plt.figure()
    for i in range(data.shape[0]):  # 1083
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.title(title)
    plt.show()

def main():
    data, label, n_samples, n_features = get_data()
    tsne = TSNE(n_components=2, init='pca', random_state=0)  # n_components: 64 -> 2；
    t0 = time()
    result = tsne.fit_transform(data)  # [1083,64]-->[1083,2]
    plot_embedding(result, label, 't-SNE embedding of the digits (time %.2fs)' % (time() - t0))


if __name__ == '__main__':
    main()