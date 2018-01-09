'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    a = 2
    b = 2
    for k in range(10):
        for i in range(64):
            N_c = 0
            for n, X_n in enumerate(train_data):
                if (train_labels[n] == k):
                    N_c += X_n[i]
            N = 0
            for n, X_n in enumerate(train_labels):
                if (train_labels[n] == k):
                    N += 1
            eta[k][i] = (N_c + a - 1) / (N + a + b -2)
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    imgs = []
    for i in range(10):
        img_i = class_images[i]
        imgs.append(np.reshape(img_i, (-1, 8)))
        # ...
    all_concat = np.concatenate(imgs, 1)
    plt.imshow(all_concat, cmap="gray")
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for k in range(10):
        for i in range(64):
            generated_data[k][i] = np.random.binomial(1, eta[k][i])
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    result = []
    for digit in bin_digits:
        log_likelihood = []
        for k in range(10):
            like = 1
            for i in range(64):
                like = like * (eta[k][i]**digit[i]) * ((1-eta[k][i])**(1-digit[i]))
            log_likelihood.append(np.log(like))
        result.append(log_likelihood)
    return result

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gen_log_likelihood = generative_likelihood(bin_digits, eta)
    result = []
    for gen in gen_log_likelihood:
        log_gen = np.log(sum([np.exp(gen_k) for gen_k in gen]) / 10)
        con_log_likelihoods = []
        for gen_k in gen:
            con_log_likelihoods.append(gen_k + np.log(1/10) - log_gen)
        result.append(con_log_likelihoods)
    return result

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    total = 0
    for indx, value in enumerate(cond_likelihood):
        total += value[int(labels[indx])]
    return total / len(bin_digits)

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    return [value.index(max(value)) for value in cond_likelihood]

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    # Question 3
    plot_images(eta)
    # Question 4
    generate_new_data(eta)
    # Question 5
    avg_train_conditional_likelihood = avg_conditional_likelihood(train_data, train_labels, eta)
    avg_test_conditional_likelihood = avg_conditional_likelihood(test_data, test_labels, eta)
    print("average conditional log-likelihood for train set: ", avg_train_conditional_likelihood)
    print("average conditional log-likelihood for test set: ", avg_test_conditional_likelihood)
    # Question 6
    classify_train_data = classify_data(train_data, eta)
    classify_test_data = classify_data(test_data, eta)
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
