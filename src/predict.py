import sys
import getopt
import yaml
import pandas as pd
import numpy as np
from utils.mylinearregression import MyLinearRegression as myLR
from utils.utils_ml import Normalizer

green = '\033[92m' # vert
blue = '\033[94m' # blue
yellow = '\033[93m' # jaune
red = '\033[91m' # rouge
reset = '\033[0m' #gris, couleur normale

def predict(yml_models, data, arg):
    X = np.array(data["km"]).reshape(-1, 1)  # get x data
    Y = np.array(data["price"]).reshape(-1, 1)  # get y data
    X_n = Normalizer(X)
    Y_n = Normalizer(Y)
    model = yml_models["model"]
    theta = np.array(model["theta"]).reshape(-1, 1)
    prediction = theta[0] + theta[1] * X_n.minmax(arg)
    print("{}km vehicle could be sell for\nprediction => {:.2f}\nunnormalized prediction => {}$".format(float(arg), float(prediction), int(Y_n.unminmax(prediction))))

def control(mileage):
    try:
        return np.array(mileage, dtype=np.float64).reshape(-1, 1)
    except Exception:
        print(f"Please, enter an {blue}int{reset} !")
        return None

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", [])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    with open("model.yml", "r") as stream:
        try:
            yml_models = yaml.safe_load(stream)
            data = pd.read_csv("data/data.csv", dtype=np.float64)
        except yaml.YAMLError as exc:
            print(exc)
    for arg in args:
        try:
            arg = int(arg)
            predict(yml_models, data, control(arg))
        except Exception as inst:
            raise (inst)


if __name__ == "__main__":
    main(sys.argv[1:])