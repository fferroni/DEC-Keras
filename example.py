from keras_dec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np

def preproc(X):
    # 1/d * ||x_i||2**2 = 1.0
    return (X.T / X.mean(1)).T

def get_mnist():
    np.random.seed(1234) # set seed for deterministic ordering
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis = 0)
    Y = np.concatenate((y_train, y_test), axis = 0)
    X = x_all.reshape(-1,x_all.shape[1]*x_all.shape[2])
    
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32)*0.02
    Y = Y[p]
    return X, Y


#X_train = np.asarray([x.flatten() for x in X_train], dtype='float32')
#X_test = np.asarray([x.flatten() for x in X_test], dtype='float32')

#X_train = preproc(X_train)
#X_test = preproc(X_test)

X, Y  = get_mnist()

c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=500000)
c.cluster(X, y=Y)
