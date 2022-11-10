import sys
import getopt
import yaml
import pandas as pd
import numpy as np
from utils.mylinearregression import MyLinearRegression as myLR
from utils.utils_ml import normalize
from utils.utils_ml import denormalize


def predict(yml_models, data, arg):
    X = np.array(data["km"]).reshape(-1, 1)  # get x data
    Y = np.array(data["price"]).reshape(-1, 1)  # get y data
    model = yml_models["model"]
    theta = np.array(model["theta"]).reshape(-1, 1)
    my_lr = myLR(theta)

    prediction = int(denormalize(Y, my_lr.predict_(np.array(normalize(X, arg)).reshape(-1, 1))))
    print("{}km vehicle could be sell for {}$".format(arg, prediction))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", [])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    with open("model.yml", "r") as stream:
        try:
            yml_models = yaml.safe_load(stream)
            data = pd.read_csv("data/data.csv")
        except yaml.YAMLError as exc:
            print(exc)
    for arg in args:
        try:
            arg = int(arg)
            predict(yml_models, data, arg)
        except Exception as inst:
            raise (inst)


if __name__ == "__main__":
    main(sys.argv[1:])