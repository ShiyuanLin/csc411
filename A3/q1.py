'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    # tf-idf data set
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names


def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    # print('Most common word in traiing set via TF-IDF is "{}"'.format(feature_names[tf_idf_train.sum(axis=0).argmax()]))
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names


def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


def logistic_regression_model(train_data, train_labels, test_data, test_labels):
    model = LogisticRegression()

    lists = np.arange(1, 20, 1)
    param_dist = dict(C=lists)
    grid = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=None, n_jobs=1)
    grid.fit(train_data, train_labels)

    model = grid.best_estimator_

    # model.fit(train_data, train_labels)

    train_pred = model.predict(train_data)
    print('Logistic Regression train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(test_data)
    print('Logistic Regression test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


def svm_model(train_data, train_labels, test_data, test_labels):
    model = svm.SVC()

    lists = np.arange(1, 20, 1)
    param_dist = dict(C=lists)
    grid = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=None, n_jobs=-1)
    grid.fit(train_data, train_labels)

    model = grid.best_estimator_

    # model.fit(train_data, train_labels)

    train_pred = model.predict(train_data)
    print('SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(test_data)
    print('SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


def knn_model(train_data, train_labels, test_data, test_labels):
    model = KNeighborsClassifier()

    k_range = range(1, 20)
    param_dist = dict(n_neighbors = k_range)
    grid = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=4, n_jobs=-1)
    grid.fit(train_data, train_labels)

    model = grid.best_estimator_
    # model.fit(train_data, train_labels)

    train_pred = model.predict(train_data)
    print('KNN train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(test_data)
    print('KNN test accuracy = {}'.format((test_pred == test_labels).mean()))


if __name__ == '__main__':
    train_data, test_data = load_data()
    # train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    # bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)

    train_tf, test_tf, feature_names = tf_idf_features(train_data, test_data)
    # bnb_model_t = bnb_baseline(train_tf, train_data.target, test_tf, test_data.target)

    # logistic_regression_model(train_bow, train_data.target, test_bow, test_data.target)
    lr_model = logistic_regression_model(train_tf, train_data.target, test_tf, test_data.target)

    # svm_model(train_bow, train_data.target, test_bow, test_data.target)
    svm_model(train_tf, train_data.target, test_tf, test_data.target)

    # knn_model(train_bow, train_data.target, test_bow, test_data.target)
    knn_model(train_tf, train_data.target, test_tf, test_data.target)

    test_pred = lr_model.predict(test_tf)
    # scores = f1_score(test_data.target, test_pred, average=None)
    # argSort = scores.argsort()
    # scores = scores[argSort]
    # matrix_data = confusion_matrix(test_data.target, test_pred)
    # print('raw matrix: "{}"'.format(matrix_data))
    cm = confusion_matrix(test_data.target, test_pred)
    prec, _, _, _ = precision_recall_fscore_support(test_data.target, test_pred)
    prec_sort = prec.argsort()
    print('{}'.format(prec))
    print('{}'.format(prec_sort))
    print('2 most confused classes are "{}" and "{}"'.format(test_data.target_names[prec_sort[0]], test_data.target_names[prec_sort[1]]))
