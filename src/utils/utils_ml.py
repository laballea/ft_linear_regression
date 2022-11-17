import numpy as np


def data_spliter(x, y, proportion):
    """
    split data into a train set and a test set, respecting to the given proportion
    return (x_train, x_test, y_train, y_test)
    """
    if (not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(proportion, float)):
        print("spliter invalid type")
        return None
    if (x.shape[0] != y.shape[0]):
        print("spliter invalid shape")
        return None
    arr = np.concatenate((x, y), axis=1)
    N = len(y)
    X = arr[:, :x.shape[1]]
    Y = arr[:, x.shape[1]]
    sample = int(proportion*N)
    np.random.shuffle(arr)
    x_train, x_test, y_train, y_test = np.array(X[:sample, :]), np.array(X[sample:, :]), np.array(Y[:sample, ]).reshape(-1, 1), np.array(Y[sample:, ]).reshape(-1, 1)
    return (x_train, x_test, y_train, y_test)


def cross_validation(x, y, K):
    """
    split data into n parts
    """
    if (not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray)):
        print("spliter invalid type")
        return None
    if (x.shape[0] != y.shape[0]):
        print("spliter invalid shape")
        return None
    arr = np.concatenate((x, y), axis=1)
    N = len(y)
    np.random.shuffle(arr)
    for n in range(K):
        sample = int((1 / K) * N)
        test = arr[(sample * n):(sample * (n + 1))]
        train = np.concatenate([arr[0:(sample * n)], arr[(sample * (n + 1)):N]])
        x_train, y_train, x_test, y_test = train[:, :x.shape[1]], train[:, x.shape[1]].reshape(-1, 1), test[:, :x.shape[1]], test[:, x.shape[1]].reshape(-1, 1),
        yield (x_train, y_train, x_test, y_test)


def add_polynomial_features(x, power):
    try:
        if (not isinstance(x, np.ndarray) or (not isinstance(power, int) and not isinstance(power, list))):
            print("Invalid type")
            return None
        if (isinstance(power, list) and len(power) != x.shape[1]):
            return None
        result = x.copy()
        if not isinstance(power, list):
            for po in range(2, power + 1):
                for col in x.T:
                    result = np.concatenate((result, (col**po).reshape(-1, 1)), axis=1)
        else:
            for col, power_el in zip(x.T, power):
                for po in range(2, power_el + 1):
                    result = np.concatenate((result, (col**po).reshape(-1, 1)), axis=1)
        return np.array(result)
    except Exception as inst:
        print(inst)
        return None


def intercept_(x):
    """
    add one columns to x
    """
    try:
        if (not isinstance(x, np.ndarray)):
            print("intercept_ invalid type")
            return None
        return np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
    except Exception as inst:
        print(inst)
        return None


def normalize(x, value=None):
    """
    normalize matrix with minmax method
    """
    if not isinstance(x, np.ndarray):
        print("normalize Invalid type.")
        return None
    result = []
    if (value == None):
        for row in x.T:
            min_r = min(row)
            max_r = max(row)
            result.append([(el - min_r) / (max_r - min_r) for el in row])
        return np.array(result).T
    else:
        min_r = float(min(x))
        max_r = float(max(x))
        return (value - min_r) / (max_r - min_r)

def denormalize(x, norm_x):
    """
    normalize matrix with minmax method
    """
    if not isinstance(x, np.ndarray):
        print("normalize Invalid type.")
        return None
    result = []
    for row_x, row_norm_x in zip(x.T, norm_x.T):
        min_r = min(row_x)
        max_r = max(row_x)
        result.append([el * (max_r - min_r) + min_r for el in row_norm_x])
    return np.array(result).T


from cmath import inf
import numpy as np



class Normalizer():
    def __init__(self, X = None):
        if X is not None:
            self.X = X
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
        else:
            self.mean_ = None
            self.std_ = None
        
    def z_score(self, X):
        try:
            X_tr = np.copy(X)
            X_tr -= self.mean_
            X_tr /= self.std_
            return X_tr
        except Exception as inst:
            print(inst)
            return 0

    def unz_score(self, X_tr):
        try:
            X = np.copy(X_tr)
            X *= self.std_
            X += self.mean_
            return X
        except Exception:
            return np.array([[0.0]])

    def minmax(self, x):
        if not isinstance(x, np.ndarray):
            print("normalize Invalid type.")
            return None
        result = []
        for row_x, row_base in zip(x.T, self.X.T):
            min_r = min(row_base)
            max_r = max(row_base)
            result.append([(el - min_r) / (max_r - min_r) for el in row_x])
        return np.array(result).T

    def unminmax(self, x):
        """
        normalize matrix with minmax method
        """
        if not isinstance(x, np.ndarray):
            print("normalize Invalid type.")
            return None
        result = []
        for row_x, row_base in zip(x.T, self.X.T):
            min_r = min(row_base)
            max_r = max(row_base)
            result.append([el * (max_r - min_r) + min_r for el in row_x])
        return np.array(result).T