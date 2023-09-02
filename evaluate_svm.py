import cvxopt
import cvxopt.solvers
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score
cvxopt.solvers.options['show_progress'] = False


class SVM():
    def __init__(self, kernel, polyconst=1, gamma=10, degree=2):
        self.kernel = kernel
        self.polyconst = float(1)
        self.gamma = float(gamma)
        self.degree = degree
        self.kf = {
            "linear": self.linear,
            "rbf": self.rbf,
            "poly": self.polynomial
        }
        self._support_vectors = None
        self._alphas = None
        self.intercept = None
        self._n_support = None
        self.weights = None
        self._support_labels = None
        self._indices = None

    def linear(self, x, y):
        return np.dot(x.T, y)

    def polynomial(self, x, y):
        return (np.dot(x.T, y) + self.polyconst) ** self.degree

    def rbf(self, x, y):
        return np.exp(-1.0 * self.gamma * np.dot(np.subtract(x, y).T, np.subtract(x, y)))

    def transform(self, X):
        K = np.zeros([X.shape[0], X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = self.kf[self.kernel](X[i], X[j])
        return K

    def fit(self, data, labels):
        num_data, num_features = data.shape
        labels = np.array(labels).astype(np.double)
        # print(labels.shape)
        K = self.transform(data)
        P = cvxopt.matrix(np.outer(labels, labels) * K)
        q = cvxopt.matrix(np.ones(num_data) * -1)
        A = cvxopt.matrix(labels, (1, num_data))
        b = cvxopt.matrix(0.0)
        #C = 0.01  # You can modify this value as needed
        #G = cvxopt.matrix(np.vstack((np.diag(np.ones(num_data) * -1), np.diag(np.ones(num_data)))), tc='d')
        #h = cvxopt.matrix(np.hstack((np.zeros(num_data), np.ones(num_data) * C)), tc='d')
        G = cvxopt.matrix(np.diag(np.ones(num_data) * -1))
        h = cvxopt.matrix(np.zeros(num_data))

        alphas = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
        is_sv = alphas > 1e-5
        self._support_vectors = data[is_sv]
        self._n_support = np.sum(is_sv)
        self._alphas = alphas[is_sv]
        self._support_labels = labels[is_sv]
        self._indices = np.arange(num_data)[is_sv]
        self.intercept = 0
        for i in range(self._alphas.shape[0]):
            self.intercept += self._support_labels[i]
            self.intercept -= np.sum(self._alphas * self._support_labels * K[self._indices[i], is_sv])
        self.intercept /= self._alphas.shape[0]
        #print(self._alphas.shape)
        self.weights = np.sum(data * labels.reshape(num_data, 1) * self._alphas.reshape(num_data, 1), axis=0,
                              keepdims=True) if self.kernel == "linear" else None
        print("Training done!")

    def signum(self, X):
        return np.where(X > 0, 1, -1)

    def project(self, X):
        if self.kernel == "linear":
            score = np.dot(X, self.weights) + self.intercept
        else:
            score = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                s = 0
                for alpha, label, sv in zip(self._alphas, self._support_labels, self._support_vectors):
                    s += alpha * label * self.kf[self.kernel](X[i], sv)
                score[i] = s
            score = score + self.intercept
        return score

    def predict(self, X):
        return self.signum(self.project(X))


def load_labels(path):
    feature_vectors = []
    labels = []
    files = os.listdir(path)
    #files.sort()
    #folders = ["aluminium", "copper", "PCB"]
    for name in files:
        for i in range(1, 7):
            if name == 'aluminium':
                labels.append(0)
            elif name == 'copper':
                labels.append(1)
            else:
                labels.append(2)
    return labels


def split_data(feature_vectors, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    file_path = 'D:/Dissertation/shape_features/shape/'
    labels = load_labels(file_path)
    #print(labels)
    data = np.loadtxt('D:/Dissertation/shape_features/features.txt')
    # num_data, num_features = data.shape
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    svm = SVM(kernel='rbf')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    # Calculate precision and recall for each category (0, 1, and 2)
    precision_matrix = []
    recall_matrix = []
    for category in range(3):
        true_labels = (np.array(y_test) == category).astype(int)  # Convert to integers
        predicted_labels = (np.array(y_pred) == category).astype(int)  # Convert to integers
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        precision_matrix.append(precision)
        recall_matrix.append(recall)

        # Print and/or store the precision and recall matrices
    for category in range(3):
        print(
            f"Category {category}: Precision = {precision_matrix[category]:.2f}, Recall = {recall_matrix[category]:.2f}")

        # Calculate and print overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Overall Accuracy: {accuracy:.2f}")
    # accuracy = np.mean(y_pred == y_test)
    # print(f"Accuracy: {accuracy:.2f}")
