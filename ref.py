import numpy as np
import pandas as pd


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def cross_entropy(x, y):
    return -np.sum(y * np.log(x))


def forward(W, x):
    z = W @ x
    a = softmax(z)
    return a


def backward(x, y, a):
    dz = a - y
    dW = np.expand_dims(dz, 1) @ np.expand_dims(x, 0)
    return dW


if __name__ == "__main__":
    data = pd.read_csv("data/mnist_train.csv").to_numpy()
    train_y = data[:, 0]
    train_X = data / 255
    train_X[:, 0] = 1
    train_y = np.eye(10)[train_y]

    data = pd.read_csv("data/mnist_test.csv").to_numpy()
    test_y = data[:, 0]
    test_X = data / 255
    test_X[:, 0] = 1
    test_y = np.eye(10)[test_y]

    in_dim = 28 * 28 + 1
    out_dim = 10

    np.random.seed(12)
    W = np.random.randn(out_dim, in_dim) * np.sqrt(2/(784 + 10))

    lr = 0.01
    epochs = 10
    batch_size = 64
    num_batch = len(train_X) // batch_size

    for epoch in range(epochs):
        mean_loss = 0.
        mean_acc = 0.

        for b in range(num_batch):
            dW_sum = np.zeros_like(W)
            for i in range(batch_size):
                idx = b * batch_size + i
                x = train_X[idx]
                y = train_y[idx]
                a = forward(W, x)

                dW_sum += backward(x, y, a)
                mean_loss += cross_entropy(a, y) / batch_size
                pred = np.argmax(a)
                trg = np.argmax(y)
                mean_acc += np.sum(pred == trg) / batch_size
            W -= lr / batch_size * dW_sum

        print(f'Epoch {epoch+1}')
        print(f'Train/loss    : {mean_loss / num_batch:.4f}')
        print(f'Train/accuracy: {mean_acc / num_batch:.4f}')

        test_acc = 0.0

        for i in range(len(test_X)):
            x = test_X[i]
            y = test_y[i]
            a = forward(W, x)

            pred = np.argmax(a)
            trg = np.argmax(y)
            test_acc += np.sum(pred == trg)
        print(f'Test/accuracy : {test_acc / len(test_X):.4f}')
