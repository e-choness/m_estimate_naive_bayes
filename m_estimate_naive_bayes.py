from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


def get_data():
    iris = load_iris()
    x = iris.data
    y = iris.target
    n_feature = x.shape[1]
    x_coded = None
    le = LabelEncoder()

    # encode all the floats into categorical
    for f in range(n_feature):
        if x_coded is None:
            x_coded = le.fit_transform(x[:, f]).reshape(-1, 1)
        else:
            x_coded = np.concatenate((x_coded, le.fit_transform(x[:, f]).reshape(-1, 1)), axis=1)

    # get training and testing samples, set random_state to 123 will yield 100% accuracy
    # If leave it be it will randomly split training and testing samples
    xtrain, xtest, ytrain, ytest = train_test_split(x_coded, y, test_size=0.2)

    return xtrain, xtest, ytrain, ytest


class M_Estimate_NB:
    def __init__(self):
        self.n_feature = None
        self.class_unique = None
        self.prior = None
        # store likelihood
        self.likelihood = None

    def fit(self, xtrain, ytrain):
        self.class_unique = np.unique(ytrain)
        self.n_feature = xtrain.shape[1]

        # calculate probabilities of each class
        y_bin = np.bincount(ytrain)
        self.prior = y_bin / np.sum(y_bin)

        # calculate and store probabilities of each unique values in terms of each class, this is for prediction
        self.likelihood = []
        for i in range(self.n_feature):
            feature = xtrain[:, i]

            feature_prob = self.class_prob(feature, ytrain)
            self.likelihood.append(feature_prob)

        # print(self.likelihood)
        # print(self.prior)

    def class_prob(self, feature, y):
        listofzeros = [0] * (max(feature) + 1)
        # This is nc+mp/n+m, since m is the number of possible values of the feature
        # and p is 1/the number of possible values of the feature
        # mp = 1, so each unique value count + 1 equals the equation above
        listofones = np.array(listofzeros) + 1
        feature_prob = []

        for c in self.class_unique:
            x_c = feature[y == c]
            u_list = self.pdf(x_c, listofones)
            feature_prob.append(u_list)
        return np.array(feature_prob)

    # count all unique values and calculate the probabilities
    def pdf(self, x, listofones):
        u_list = listofones.copy()
        for i in x:
            u_list[i] += 1
        prob = u_list / sum(u_list)
        return prob.tolist()

    # predict result
    def predict(self, xtest):

        predictions = []
        for x in xtest:
            prediction = self.find_class(x)
            predictions.append(prediction)
        # print(predictions)
        return predictions

    def find_class(self, x):
        c_probs = []
        for c in self.class_unique:
            c_p = self.prior[c]
            pos = self.get_posterior(x, c)
            c_probs.append(pos * c_p)
        # print(np.argmax(c_probs))
        return np.argmax(c_probs)

    def get_posterior(self, x, c):
        probs = []
        for idx, x_f in enumerate(x):
            c_l = self.likelihood[idx][c]
            f_p = c_l[x_f]
            probs.append(f_p)

        # print(np.prod(probs))
        return np.prod(probs)

    def evaluate(self, xtest, ytest):
        predictions = self.predict(xtest)
        count = 0
        n_sample = len(ytest)
        for i in range(n_sample):
            if predictions[i] == ytest[i]:
                count += 1

        accuracy = count / n_sample
        print("Testing accuracy:", accuracy)


xtrain, xtest, ytrain, ytest = get_data()

menb = M_Estimate_NB()
menb.fit(xtrain, ytrain)

print("predictions:", menb.predict(xtest))
print("ground truth:", ytest)
# Since the testing sample sometimes contain unseen instances, the classifier can break in this case.
# Normally the testing result ranges from 0.83 to 1.0
menb.evaluate(xtest, ytest)
