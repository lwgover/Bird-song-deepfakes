from keras.datasets import mnist
import numpy as np
def get_stupid_mnist():
    (X_train_full, y_train_full), (X_test, y_test)= mnist.load_data()
    X = list(X_train_full) + list(X_test)
    X = tuple(map(lambda a:a.reshape((28,14,2)),X))
    return X
   
if __name__ == "__main__":
    get_stupid_mnist()
    