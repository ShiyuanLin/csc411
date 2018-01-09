'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
from sklearn.model_selection import KFold
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        dist = self.l2_distance(test_point).tolist()
        min_dict = {}
        min_labels = []
        for i in range(k):
            indx = dist.index(min(dist))
            if self.train_labels[indx] not in min_dict:
                min_dict[self.train_labels[indx]] = dist[indx]
            else:
                min_dict[self.train_labels[indx]] += dist[indx]
            min_labels.append(self.train_labels[indx])
            dist[indx] = max(dist)
        record = {}
        for label in min_labels:
            if label not in record:
                record[label] = 1
            else:
                record[label] += 1
        digit = int(max(record, key=record.get))
        # confl = []
        # # Get conflict label
        # for i in record:
        #     if (i != digit) and (record[i] == record[digit]):
        #         confl.append(i)
        # # Get label which has the minimum distance
        # min_dist = max(dist)
        # for i in confl:
        #     if (min_dict[i]/record[i]) < min_dist:
        #         min_dist = min_dict[i]/record[i]
        #         digit = i
        return digit


def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    kf = KFold(n_splits=10)
    acc = []
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        avg_acc = []
        for train_index, test_index in kf.split(train_data):
            train = train_data[train_index]
            test = train_data[test_index]
            train_label = train_labels[train_index]
            test_label = train_labels[test_index]
            knn = KNearestNeighbor(train, train_label)
            avg_acc.append(classification_accuracy(knn, k, test, test_label))
        acc.append(sum(avg_acc)/len(avg_acc))
    # print(acc)
    for i, value in enumerate(acc):
        print("k = ", i+1, ", accuracy is ", value)
    return len(acc) - acc[::-1].index(max(acc))

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    predict_labels = []
    for data in eval_data:
        predict_labels.append(knn.query_knn(data, k))
    count = 0
    for i in range(len(eval_labels)):
        if predict_labels[i] == eval_labels[i]:
            count += 1
    return (count/len(eval_labels))

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    predicted_label = knn.query_knn(test_data[0], 1)
    # plt.imshow(predicted_label, cmap='gray')
    # plt.show()
    # print(predicted_label)
    print("test data, k=1: ",classification_accuracy(knn, 1, test_data, test_labels))
    print("test data, k=15: ",classification_accuracy(knn, 15, test_data, test_labels))
    print("train data, k=1: ",classification_accuracy(knn, 1, train_data, train_labels))
    print("train data, k=15: ",classification_accuracy(knn, 15, train_data, train_labels))

    print("Optimal k is ", cross_validation(train_data, train_labels))

if __name__ == '__main__':
    main()