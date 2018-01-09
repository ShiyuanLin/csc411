'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(7000):
        label_class = train_labels[i]
        # print(train_data[i])
        means[int(label_class)] = means[int(label_class)] + train_data[i]
    # print("------------------------------------------------------")
    # print(means)
    for i in range(10):
        means[i] = means[i] / 7000
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    # Compute covariances
    for i in range(10):
        for j in range(64):
            for k in range(64):
                i_indx = []
                for indx in range(len(train_data)):
                    if (train_labels[indx] == i):
                        i_indx.append(indx)
                N = len(i_indx)
                X_j = [train_data[indx][j] for indx in i_indx]
                X_k = [train_data[indx][k] for indx in i_indx]
                cov_j_k = (1 / N) * sum([(X_j[n] - means[i][j]) * (X_k[n] - means[i][k]) for n in range(N)])
                covariances[i][j][k] = cov_j_k
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    cov_diags = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        # ...
        cov_diags.append(np.reshape(cov_diag, (-1, 8)))

    all_concat = np.concatenate(np.log(cov_diags), 1)
    plt.imshow(all_concat, cmap="gray")
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    result = []
    for digit in digits:
        log_likelihood = []
        for k in range(10):
            exp = np.exp((-1/2) * np.linalg.multi_dot([(digit - means[k]).T,
                                                       np.linalg.inv(covariances[k] + 0.01 * np.identity(64)),
                                                       (digit - means[k])]))
            log_likelihood.append(np.log(((2 * np.pi)**(-64/2)) *
                                         (np.linalg.det(covariances[k] + 0.01 * np.identity(64))**(-1/2)) *
                                         exp))
        result.append(log_likelihood)
    return result

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    result = []
    gen_likelihood = generative_likelihood(digits, means, covariances)
    for gen in gen_likelihood:
        log_gen = np.log(sum(np.exp(gen_k) for gen_k in gen)/10)
        cond_likelihood = []
        for gen_k in gen:
            cond_likelihood.append(gen_k + np.log(1/10) - log_gen)
        result.append(cond_likelihood)
    return result

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    total = 0
    for indx, value in enumerate(cond_likelihood):
        total += value[int(labels[indx])]
    return total/len(digits)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    return [value.index(max(value)) for value in cond_likelihood]

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    # print(means)
    covariances = compute_sigma_mles(train_data, train_labels)
    plot_cov_diagonal(covariances)

    # Evaluation
    # Question 2
    avg_train_conditional_likelihood = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_test_conditional_likelihood = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("average conditional log-likelihood for train set: ", avg_train_conditional_likelihood)
    print("average conditional log-likelihood for test set: ", avg_test_conditional_likelihood)

    # Question 3
    classify_train_data = classify_data(train_data, means, covariances)
    classify_test_data = classify_data(test_data, means, covariances)
    train_accuracy = 0
    for index, k in enumerate(classify_train_data):
        if train_labels[index] == k:
            train_accuracy += 1
    train_accuracy = train_accuracy / len(train_labels)
    test_accuracy = 0
    for index, k in enumerate(classify_test_data):
        if test_labels[index] == k:
            test_accuracy += 1
    test_accuracy = test_accuracy / len(test_labels)
    print("accuracy on train data is ", train_accuracy)
    print("accuracy on test data is ", test_accuracy)


if __name__ == '__main__':
    main()