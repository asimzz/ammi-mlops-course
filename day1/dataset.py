from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def load_iris_dataset():
    dataset = load_iris()
    dataset = pd.DataFrame(data= np.c_[dataset["data"], dataset['target']],
                     columns= dataset['feature_names'] + ['target'])
    return dataset


def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.2)
    