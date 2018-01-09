import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


BATCHES = 50
K = 500

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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''

    return np.divide(-np.dot((2*(y - np.dot(X,w))), X), X.shape[0])
    # for i in range(X.shape[0]):
    #     if (i==0):
    #         err = (y[i] - np.dot(X[i], w))* X[i]
    #     else:
    #         np.add((y[i] - np.dot(X[i], w))* X[i], err)
    # print(np.divide(err, X.shape[0]))
    # np.divide(err, -0.5)
    # return np.divide(err, X.shape[0])

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    X_b, y_b = batch_sampler.get_batch()
    batch_grad = lin_reg_gradient(X_b, y_b, w)
    # batch_grads = np.empty((0, X.shape[1]))

    for i in range(K):
        X_b, y_b = batch_sampler.get_batch()
        if (i==0) :
            batch_grads =  np.add(lin_reg_gradient(X_b, y_b, w), w)
        else:
            np.add(lin_reg_gradient(X_b, y_b, batch_grads), batch_grads)
        # print(lin_reg_gradient(X_b, y_b, w))
        #batch_grads.append(lin_reg_gradient(X_b, y_b, w))
    #comp_grad = np.mean(batch_grads)
    np.divide(batch_grads, K)
    true_grad = lin_reg_gradient(X, y, w)
    print("Cosine similarity:", cosine_similarity(batch_grads, true_grad))
    distance = (batch_grads-true_grad) ** 2
    distance = distance.sum(axis=0)
    # distance = np.sqrt(distance)
    print("Square matrix distance:", distance)
    print(batch_grads)

    print(batch_grads.shape)
    #print("Compted gradient:", comp_grad)

    # q3-6
    log_ms= []
    log_var = []
    batch_gradsf = []
    for m in range(1, 400):
        log_ms.append(np.log(m))
        for i in range(K):
            X_b, y_b = batch_sampler.get_batch(m)
            batch_grad = lin_reg_gradient(X_b, y_b, w)
            batch_gradsf.append(batch_grad[0])
        np_batch_gradsf = np.array(batch_gradsf)
        log_var.append(np.log(np.mean(np_batch_gradsf ** 2) - np.mean(np_batch_gradsf) ** 2))
    plt.plot(log_ms,log_var, ".")
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
