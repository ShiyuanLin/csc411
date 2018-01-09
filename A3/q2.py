import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)


class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch


class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        self.vel = (self.beta * self.vel) - (self.lr * grad)

        params += self.vel

        return params


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)

    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss

        result = []
        for i in range(X.shape[0]):
            result.append(max(1 - y[i] * np.dot(X[i], self.w), 0))
        return np.array(result)

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        hinge_loss = self.hinge_loss(X, y)
        hinge_loss_grad = []
        for i in range(hinge_loss.shape[0]):
            if (hinge_loss[i] == 0):
                hinge_loss_grad.append(np.zeros(X[0].shape))
            else:
                hinge_loss_grad.append(-np.dot(y[i], X[i]))
        loss_grad = np.array(hinge_loss_grad)
        for i in range(np.shape(loss_grad)[0]):
            loss_grad[i] += self.w
        return (self.c/len(X)) * np.sum(loss_grad, axis=0)

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        result = []
        for x in X:
            if (np.dot(self.w, x) > 0):
                result.append(1)
            else:
                result.append(-1)
        return np.array(result)


def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets


def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''

    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]


    for _ in range(steps):
        # Optimize and update the history
        w = optimizer.update_params(w, func_grad(w))
        w_history.append(w)
    return w_history


def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    svm = SVM(penalty, train_data.shape[1])
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)
    for _ in range(iters):
        batch_x, batch_y = batch_sampler.get_batch()
        grad = svm.grad(batch_x, batch_y)
        svm.w = optimizer.update_params(svm.w, grad)
    return svm


if __name__ == '__main__':

    # 2.1
    gd_zero = GDOptimizer(1.0, 0.0)
    w_zero = optimize_test_function(gd_zero)
    gd_nine = GDOptimizer(1.0, 0.9)
    w_nine = optimize_test_function(gd_nine)

    line_up = plt.plot(w_zero, '.')
    line_down = plt.plot(w_nine, '.')
    plt.legend(['Beta: 0.0', 'Beta: 0.9'])
    plt.xlabel('time-step')
    plt.ylabel('w')
    plt.show()

    # 2.3
    train_data, train_targets, test_data, test_targets = load_data()

    optimizer = GDOptimizer(0.05, 0)
    svm = optimize_svm(train_data, train_targets, 1.0, optimizer,100, 500)
    train_pred = svm.classify(train_data)
    train_acc = (train_pred == train_targets).mean()
    test_pred = svm.classify(test_data)
    test_acc = (test_pred == test_targets).mean()
    print("The train loss is {} when beta is 0".format(svm.hinge_loss(train_data, train_targets).mean()))
    print("The test loss is {} when beta is 0".format(svm.hinge_loss(test_data, test_targets).mean()))
    print("The train accuracy is {} when beta is 0".format(train_acc))
    print("The test accuracy is {} when beta is 0".format(test_acc))
    plt.imshow(svm.w.reshape((28,28)),cmap='gray')
    plt.show()

    optimizer = GDOptimizer(0.05, 0.1)
    svm = optimize_svm(train_data, train_targets, 1.0, optimizer, 100, 500)
    train_pred = svm.classify(train_data)
    train_acc = (train_pred == train_targets).mean()
    test_pred = svm.classify(test_data)
    test_acc = (test_pred == test_targets).mean()
    print("The train loss is {} when beta is 0.1".format(svm.hinge_loss(train_data, train_targets).mean()))
    print("The test loss is {} when beta is 0.1".format(svm.hinge_loss(test_data, test_targets).mean()))
    print("The train accuracy is {} when beta is 0.1".format(train_acc))
    print("The test accuracy is {} when beta is 0.1".format(test_acc))
    plt.imshow(svm.w.reshape((28, 28)), cmap='gray')
    plt.show()