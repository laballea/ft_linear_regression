import sys
import getopt
import yaml
import pandas as pd
import numpy as np
from utils.utils_ml import normalize
from utils.utils_ml import denormalize
from utils.utils_ml import data_spliter
from utils.mylinearregression import MyLinearRegression as myLR
from matplotlib import pyplot as plt


def init_models(yml_models, data):
    yml_models = {}
    yml_models["model"] = {
        "theta": np.ones(2).reshape(-1, 1).tolist(),
        "rmse": None,
        "historic": [],
        "total_it": 0,
        "alpha": 1e-1,
    }
    with open("model.yml", 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)


def train(yml_models, data, alpha, rate):
    X = np.array(data["km"]).reshape(-1, 1)  # get x data
    Y = np.array(data["price"]).reshape(-1, 1)  # get y data
    x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.8)  # split into a training and test set
    x_test = normalize(x_test)  # normalize
    y_test = normalize(y_test)
    x_train = normalize(x_train)
    y_train = normalize(y_train)

    model = yml_models["model"]  #  get model
    theta = np.array(model["theta"]).reshape(-1, 1)  # get model's theta
    my_lr = myLR(theta, alpha, rate)  # init linear regression
    historic = my_lr.fit_(x_train, y_train, historic_bl=True)  # train model
    end_rmse = my_lr.rmse_(y_test, my_lr.predict_(x_test))  # evaluate it
    model["rmse"] = end_rmse  #  save data into file
    model["theta"] = my_lr.theta.tolist()
    model["total_it"] = int(model["total_it"]) + rate
    model["historic"] = model["historic"] + historic
    with open("model.yml", 'w') as outfile:
        yaml.dump(yml_models, outfile, default_flow_style=None)

def display(yml_models, data):
    X = np.array(data["km"]).reshape(-1, 1)  # get x data
    Y = np.array(data["price"]).reshape(-1, 1)  # get y data
    fig = plt.figure()
    plt.scatter(X, Y, label="data")


    model = yml_models["model"]
    theta = np.array(model["theta"]).reshape(-1, 1) 
    my_lr = myLR(theta)
    prediction = denormalize(Y, my_lr.predict_(normalize(X)))
    plt.plot(X, prediction, c="red", label="prediction")
    plt.xlabel('km')
    plt.ylabel('price')
    plt.title(my_lr.mse_(normalize(Y), my_lr.predict_(normalize(X))))
    plt.show()

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "a:r:", ["reset", "train", "display"])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    with open("model.yml", "r") as stream:
        try:
            yml_models = yaml.safe_load(stream)
            data = pd.read_csv("data/data.csv")
        except yaml.YAMLError as exc:
            print(exc)
    alpha, rate = 0.1, 1000
    for opt, arg in opts:
        if (opt == "-a"):
            alpha = float(arg)
        elif (opt == "-r"):
            rate = int(arg)
    for opt, arg in opts:
        if opt == '--reset':
            init_models(yml_models, data)
        elif opt == '--train':
            train(yml_models, data, alpha=alpha, rate=rate)
        elif opt == '--display':
            display(yml_models, data)


if __name__ == "__main__":
    main(sys.argv[1:])