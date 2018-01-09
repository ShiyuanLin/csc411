from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.plot(X[:, i],y, 'bo')
        plt.title(features[i])

    # plt.title(features)
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    # raise NotImplementedError()
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)

    # Add a bias term
    bias = []
    for i in range(X.shape[0]):
        bias_row = [1] + X[i].tolist()
        bias.append(bias_row)
    X = np.array(bias)

    #TODO: Split data into train and test
    random_i = np.random.choice(len(X), len(X), replace=False)
    # print(X.shape)
    # print(y.shape)
    X_test = X[random_i[:(len(random_i)//5)]]
    y_test = y[random_i[:(len(random_i)//5)]]
    X = X[random_i[(len(random_i)//5):]]
    y = y[random_i[(len(random_i)//5):]]
    # print(len(random_i))
    # print(X_test.shape)
    # print(y_test.shape)
    # print(X.shape)
    # print(y.shape)


    # Fit regression model
    w = fit_regression(X, y)
    print("fitted values:", w)
    print("CRIM: {} | ZN: {}  | INDUS: {}\nCHAS: {}   | NOX: {}  | RM: {} \nAGE: {} | DIS: {} | RAD: {}\n"
          "TAXl: {} | PTRATIO: {} | B:{} \nLSTAT:{}".format(
        w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13]))
    print("BIAS:", w[0])
    # Compute fitted values, MSE, etc.
    fit_y = np.dot(X_test, w)
    mse = np.square(fit_y - y_test).mean()
    rmse = np.sqrt(mse)
    mad = abs(fit_y - y_test).mean()
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAD:", mad)


if __name__ == "__main__":
    main()

