
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from dmba import classificationSummary, gainsChart, liftChart
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular


data_csv = pd.read_csv('winemag-data-130k-v2.csv')
print(data_csv)


