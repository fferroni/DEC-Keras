from keras_dec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np

def preproc(X):
    # 1/d * ||x_i||2**2 = 1.0
    return (X.T / X.mean(1)).T

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.asarray([x.flatten() for x in X_train], dtype='float32')
X_test = np.asarray([x.flatten() for x in X_test], dtype='float32')

X_train = preproc(X_train)
X_test = preproc(X_test)

c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
c.initialize(X_train, epochs=60)
c.cluster(X_train, y=y_train, iter_max = 20000)
