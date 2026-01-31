import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist(sample_size=5000, random_state=42):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # Flatten images: 28x28 → 784
    X = X.reshape(X.shape[0], -1)

    # Normalize to [0,1]
    X = X / 255.0

    # Sample subset (IMPORTANT)
    np.random.seed(random_state)
    indices = np.random.choice(len(X), sample_size, replace=False)

    return X[indices], y[indices]
import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist(sample_size=5000, random_state=42):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # Flatten images: 28x28 → 784
    X = X.reshape(X.shape[0], -1)

    # Normalize to [0,1]
    X = X / 255.0

    # Sample subset (IMPORTANT)
    np.random.seed(random_state)
    indices = np.random.choice(len(X), sample_size, replace=False)

    return X[indices], y[indices]
