from kmeans import KMeans, ShowMeans
from util import LoadData, LoadDataQ4
import sys
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

if sys.version_info.major == 3:
    raw_input = input


def mogEM(x, K, iters, randConst=1, minVary=0):
    """
    Fits a Mixture of K Diagonal Gaussians on x.

    Inputs:
      x: data with one data vector in each column.
      K: Number of Gaussians.
      iters: Number of EM iterations.
      randConst: scalar to control the initial mixing coefficients
      minVary: minimum variance of each Gaussian.

    Returns:
      p: probabilities of clusters (or mixing coefficients).
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.
      logLikelihood: log-likelihood of data after every iteration.
    """
    N, T = x.shape

    # Initialize the parameters
    p = randConst + np.random.rand(K, 1)
    p = p / np.sum(p)   # mixing coefficients
    mn = np.mean(x, axis=1).reshape(-1, 1)
    vr = np.var(x, axis=1).reshape(-1, 1)

    # Q4.3 
    mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
    #mu = KMeans(x, 7, 5)
    
    vary = vr * np.ones((1, K)) * 2
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    logLikelihood = np.zeros((iters, 1))

    # Do iters iterations of EM
    for i in xrange(iters):
        # Do the E step
        respTot = np.zeros((K, 1))
        respX = np.zeros((N, K))
        respDist = np.zeros((N, K))
        ivary = 1 / vary
        logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - \
            0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
        logPcAndx = np.zeros((K, T))
        for k in xrange(K):
            dis = (x - mu[:, k].reshape(-1, 1))**2
            logPcAndx[k, :] = logNorm[k] - 0.5 * \
                np.sum(ivary[:, k].reshape(-1, 1) * dis, axis=0)

        mx = np.max(logPcAndx, axis=0).reshape(1, -1)
        PcAndx = np.exp(logPcAndx - mx)
        Px = np.sum(PcAndx, axis=0).reshape(1, -1)
        PcGivenx = PcAndx / Px
        logLikelihood[i] = np.sum(np.log(Px) + mx)

        print 'Iter %d logLikelihood %.5f' % (i + 1, logLikelihood[i])

        # Plot log likelihood of data
        plt.figure(0)
        plt.clf()
        plt.plot(np.arange(i), logLikelihood[:i], 'r-')
        plt.title('Log-likelihood of data versus # iterations of EM for %s'%(K))
        plt.xlabel('Iterations of EM')
        plt.ylabel('Log-likelihood')
        plt.draw()
        plt.savefig('LL_K=%s.png' %( K) )
        
        # Do the M step
        # update mixing coefficients
        respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
        p = respTot

        # update mean
        respX = np.zeros((N, K))
        for k in xrange(K):
            respX[:, k] = np.mean(x * PcGivenx[k, :].reshape(1, -1), axis=1)

        mu = respX / respTot.T

        # update variance
        respDist = np.zeros((N, K))
        for k in xrange(K):
            respDist[:, k] = np.mean(
                (x - mu[:, k].reshape(-1, 1))**2 * PcGivenx[k, :].reshape(1, -1), axis=1)

        vary = respDist / respTot.T
        vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    return p, mu, vary, logLikelihood


def mogLogLikelihood(p, mu, vary, x):
    """ Computes log-likelihood of data under the specified MoG model

    Inputs:
      x: data with one data vector in each column.
      p: probabilities of clusters.
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.

    Returns:
      logLikelihood: log-likelihood of data after every iteration.
    """
    K = p.shape[0]
    N, T = x.shape
    ivary = 1 / vary
    logLikelihood = np.zeros(T)
    for t in xrange(T):
        # Compute log P(c)p(x|c) and then log p(x)
        logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
            - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
            - 0.5 * \
            np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)
                   ** 2, axis=0).reshape(-1, 1)

        mx = np.max(logPcAndx, axis=0)
        logLikelihood[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx

    return logLikelihood


def q2():
    # Question 4.2 and 4.3
    K = 7
    iters = 10
    minVary = 0.01
    randConst = 100.0

    # load data
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData(
        '../toronto_face.npz')

    # Train a MoG model with 7 components on all training data, i.e., inputs_train,
    # with both original initialization and kmeans initialization.

    p, mu, vary, log_likelihood = mogEM(inputs_train, K, iters, randConst, minVary)
    ShowMeans(mu, 1)
    ShowMeans(vary, 2)
    print  p

#########
def q4():
    
    # Question 4.4
    iters = 10
    minVary = 0.01
    randConst = 10.0

    numComponents = np.array([7, 14, 21, 28, 35])
    T = numComponents.shape[0]
    errorTrain = np.zeros(T)
    errorTest = np.zeros(T)
    errorValidation = np.zeros(T)

    # extract data of class 1-Anger, 4-Happy
    dataQ4 = LoadDataQ4('../toronto_face.npz')
    # images
    x_train_anger = dataQ4['x_train_anger']
    x_train_happy = dataQ4['x_train_happy']
    x_train = np.concatenate([x_train_anger, x_train_happy], axis=1)
    x_valid = np.concatenate(
        [dataQ4['x_valid_anger'], dataQ4['x_valid_happy']], axis=1)
    x_test = np.concatenate(
        [dataQ4['x_test_anger'], dataQ4['x_test_happy']], axis=1)

    # label
    y_train = np.concatenate(
        [dataQ4['y_train_anger'], dataQ4['y_train_happy']])
    y_valid = np.concatenate(
        [dataQ4['y_valid_anger'], dataQ4['y_valid_happy']])
    y_test = np.concatenate([dataQ4['y_test_anger'], dataQ4['y_test_happy']])

    # Hints: this is p(d), use it based on Bayes Theorem
    num_anger_train = x_train_anger.shape[1]
    num_happy_train = x_train_happy.shape[1]
    log_likelihood_class = np.log(
        [num_anger_train, num_happy_train]) - np.log(num_anger_train + num_happy_train)

    for t in xrange(T):
        K = numComponents[t]
               
        p_anger, mu_anger, var_angry, ll_train_anger = mogEM(x_train_anger, K, iters, randConst, minVary)
        p_happy, mu_happy, var_happy, ll_train_happy = mogEM(x_train_happy, K, iters, randConst, minVary)

        #log-liklihood        
        ll_train_anger = mogLogLikelihood(p_anger, mu_anger, var_angry, x_train) + log_likelihood_class[0]
        ll_train_happy = mogLogLikelihood(p_happy, mu_happy, var_happy, x_train) + log_likelihood_class[1]
        ll_valid_anger = mogLogLikelihood(p_anger, mu_anger, var_angry, x_valid) + log_likelihood_class[0]
        ll_valid_happy = mogLogLikelihood(p_happy, mu_happy, var_happy, x_valid) + log_likelihood_class[1]
        ll_test_anger = mogLogLikelihood(p_anger, mu_anger, var_angry, x_test) + log_likelihood_class[0]
        ll_test_happy = mogLogLikelihood(p_happy, mu_happy, var_happy, x_test) + log_likelihood_class[1]
 
        #predictions
        pred_train = (ll_train_anger < ll_train_happy).astype(float)
        pred_valid = (ll_valid_anger <ll_valid_happy).astype(float)
        pred_test = (ll_test_anger < ll_test_happy).astype(float)
       
        #error
        errorTrain[t] = np.sum((pred_train != y_train).astype(float)) / y_train.shape[0]
        errorValidation[t] = np.sum((pred_valid != y_valid).astype(float)) / y_valid.shape[0]
        errorTest[t] = np.sum((pred_test != y_test).astype(float)) / y_test.shape[0]

        
    # Plot the error rate
    plt.figure(0)
    plt.clf()    
    plt.plot(numComponents, errorTrain, 'b', label='Train')
    plt.plot(numComponents, errorValidation, 'g', label='Valid')
    plt.plot(numComponents, errorTest, 'r', label='Test')
    plt.xlabel('Mixture Component Number (K)')
    plt.ylabel('Error Rate of Classification')
    plt.title('Error Rate vs Number of Mixture Components', fontsize =14, fontweight='bold')
    plt.legend()
    plt.draw()
    plt.savefig('errorplots.png')
    plt.pause(0.0001)
    
if __name__ == '__main__':
    
    #q2()
    q4()

    raw_input('Press Enter to continue.')
