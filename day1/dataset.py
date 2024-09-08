from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

def load_iris_dataset():
    dataset = load_iris()
    dataset = pd.DataFrame(data= np.c_[dataset["data"], dataset['target']],
                     columns= dataset['feature_names'] + ['target'])
    return dataset