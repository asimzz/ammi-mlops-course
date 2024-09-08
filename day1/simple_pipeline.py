import numpy as np
import pandas as pd

from dataset import load_iris_dataset, split_dataset
from model import train, accuracy_score



if __name__ == "__main__":
    print("---------Loading Iris Dataset---------")
    iris_dataset = load_iris_dataset()
    print(iris_dataset.head())
    
    X = iris_dataset.drop(["target"], axis=1)
    y = iris_dataset["target"]
    
    X_train, X_test, y_train, y_test = split_dataset(X,y)
    
    print("---------Evaluation---------")
    classifier = train(X_train, y_train)
    
    accuracy = accuracy_score(classifier,X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f} %")
    
    