from ast import Index
from json import encoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import model2 as model2
import pandas as pd
import numpy as np
from jedi.api.refactoring import inline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from IPython.display import display
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from bokeh.models import Y
from numpy import int64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn import metrics
from dmba import classificationSummary, gainsChart, liftChart, regressionSummary
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lime import lime_tabular
from dmba import plotDecisionTree
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import plot_roc_curve
import graphviz
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc
import six
import sys
sys.modules['sklearn.externals.six'] = six

from sklearn.metrics import roc_curve, auc

### 1 Develop an understanding of the data mining project

### 2 Obtain the dataset to be used in the analysis
wine_df = pd.read_csv('winemag-data-130k-v2.csv')

## Create duplicate wine_df dataframe
wine_duplicate = wine_df.copy()

## print(wine_duplicate.describe())
#                   id         points          price
# count  129971.000000  129971.000000  120975.000000
# mean    64985.000000      88.447138      35.363389
# std     37519.540256       3.039730      41.022218
# min         0.000000      80.000000       4.000000
# 25%     32492.500000      86.000000      17.000000
# 50%     64985.000000      88.000000      25.000000
# 75%     97477.500000      91.000000      42.000000
# max    129970.000000     100.000000    3300.000000


# Missing column percentage
# def perc_missing(df):
#     # '''prints out columns with missing values with its %'''
#     for col in df.columns:
#         pct = df[col].isna().mean() * 100
#         if pct != 0:
#             print('{} => {}%'.format(col, round(pct, 2)))


# perc_missing(wine_duplicate)


# region_1 is empty than replace region_1 with region_2 vale if the vale is 0
np.where(wine_duplicate["region_1"] == 'NaN', wine_duplicate['region_2'], wine_duplicate['region_1'])
np.where(wine_duplicate["taster_name"] == 'NaN', wine_duplicate['taster_twitter_handle'], wine_duplicate['taster_name'])

# delete if price is above 250
wine_duplicate = wine_duplicate[wine_duplicate['price'].between(15, 55)]
wine_duplicate = wine_duplicate[wine_duplicate['points'].between(90, 105)]
np.where(wine_duplicate["price"] == 'NaN', wine_duplicate['taster_twitter_handle'], wine_duplicate['taster_name'])
# drop null row
wine_duplicate = wine_duplicate[wine_duplicate['country'].notna()]
wine_duplicate = wine_duplicate[wine_duplicate['price'].notna()]
wine_duplicate = wine_duplicate[wine_duplicate['region_1'].notna()]

# converting all null to zero (0)
wine_duplicate['country'].fillna(0, inplace=True)
wine_duplicate['designation'].fillna(0, inplace=True)
wine_duplicate['price'].fillna(0, inplace=True)
wine_duplicate['province'].fillna(0, inplace=True)
wine_duplicate['region_1'].fillna(0, inplace=True)
wine_duplicate['region_2'].fillna(0, inplace=True)
wine_duplicate['taster_name'].fillna(0, inplace=True)
wine_duplicate['taster_twitter_handle'].fillna(0, inplace=True)


# plt.hist(wine_duplicate['price'])
# plt.show()
# print(wine_duplicate.count())

#1 4-10-->VALUE
#2 10-15-->POPULAR_PREMIUM
#3 15-20-->PREMIUM
#4 20-30-->SUPPER_PREMIUM
#5 30-50-->ULTRA_PREMIUM
#6 50-100-->LUXURY
#6 100-200-->SUPER_LUXURY
#8 200-250-->ICON
wine_duplicate['price_category'] = np.where(wine_duplicate.price <= 10, 'VALUE',
                                            np.where(wine_duplicate.price <= 15, 'POPULAR_PREMIUM',
                                            np.where(wine_duplicate.price <= 20, 'PREMIUM',
                                            np.where(wine_duplicate.price <= 30, 'SUPPER_PREMIUM',
                                            np.where(wine_duplicate.price <= 50, 'ULTRA_PREMIUM',
                                            np.where(wine_duplicate.price <= 100, 'LUXURY',
                                            np.where(wine_duplicate.price <= 200, 'SUPER_LUXURY', 'ICON')))))))

# print(wine_duplicate[["price", "price_category"]].head(100))
dummy_price_category = pd.get_dummies(wine_duplicate["price_category"])
wine_duplicate = pd.concat([wine_duplicate, dummy_price_category], axis=1)
# print(dummy_price_category.head())
dummy_country = pd.get_dummies(wine_duplicate["country"])
wine_duplicate = pd.concat([wine_duplicate, dummy_country], axis=1)
cols = wine_duplicate.columns.tolist()
wine_duplicate = wine_duplicate.reindex(columns=cols).fillna(0)
# print(wine_duplicate.describe())

# print(wine_duplicate.groupby("taster_twitter_handle")["id"].count())

# print(wine_duplicate.groupby("taster_twitter_handle")["id"].count())
# perc_missing(wine_duplicate)
# # Drop unnecessary columns that are not important
colsToDrop = ['price', 'price_category', 'country', 'description', 'designation', 'region_2', 'taster_name', 'taster_twitter_handle']
wine_duplicate.drop(colsToDrop, axis=1, inplace=True)

# Rename region_1
wine_duplicate.rename(columns={'region_1': 'region'}, inplace=True)

# print(wine_duplicate.count())
# id                 101017
# points             101017
# province           101017
# region             101017
# title              101017
# variety            101017
# winery             101017
# LUXURY             101017
# POPULAR_PREMIUM    101017
# PREMIUM            101017
# SUPER_LUXURY       101017
# SUPPER_PREMIUM     101017
# ULTRA_PREMIUM      101017
# VALUE              101017
# Argentina          101017
# Australia          101017
# Canada             101017
# France             101017
# Italy              101017
# Spain              101017
# US                 101017


# Data Type change
# wine_duplicate['country'] = wine_duplicate['country'].astype(float)
wine_duplicate['points'] = wine_duplicate['points'].astype(float)
wine_duplicate['id'] = wine_duplicate['id'].astype(int64)
# wine_duplicate['Argentina'] = wine_duplicate['Argentina'].astype(int64)
# wine_duplicate['Australia'] = wine_duplicate['Australia'].astype(int64)
# wine_duplicate['Canada'] = wine_duplicate['Canada'].astype(int64)
# wine_duplicate['France'] = wine_duplicate['France'].astype(int64)
# wine_duplicate['Italy'] = wine_duplicate['Italy'].astype(int64)
# wine_duplicate['Spain'] = wine_duplicate['Spain'].astype(int64)
# wine_duplicate['US'] = wine_duplicate['US'].astype(int64)
# wine_duplicate['VALUE'] = wine_duplicate['VALUE'].astype(int64)
# wine_duplicate['POPULAR_PREMIUM'] = wine_duplicate['POPULAR_PREMIUM'].astype(int64)
# wine_duplicate['PREMIUM'] = wine_duplicate['PREMIUM'].astype(int64)
# wine_duplicate['SUPPER_PREMIUM'] = wine_duplicate['SUPPER_PREMIUM'].astype(int64)
# wine_duplicate['ULTRA_PREMIUM'] = wine_duplicate['ULTRA_PREMIUM'].astype(int64)
# wine_duplicate['LUXURY'] = wine_duplicate['LUXURY'].astype(int64)
# wine_duplicate['SUPER_LUXURY'] = wine_duplicate['SUPER_LUXURY'].astype(int64)
# wine_duplicate['ICON'] = wine_duplicate['ICON'].astype(int64)

# print(wine_duplicate.dtypes)
# id                   int64
# points             float64
# province            object
# region              object
# title               object
# variety             object
# winery              object
# ICON                 int64
# LUXURY               int64
# POPULAR_PREMIUM      int64
# PREMIUM              int64
# SUPER_LUXURY         int64
# SUPPER_PREMIUM       int64
# ULTRA_PREMIUM        int64
# VALUE                int64
# Argentina            int64
# Australia            int64
# Canada               int64
# France               int64
# Italy                int64
# Spain                int64
# US                   int64
# dtype: object


# show histogram
# for col in wine_duplicate.columns:
#     hist = wine_duplicate[col].hist(bins=100)
#     print("Plot for column \"{}\"".format(col))
#     plt.show()




## 4 Reduce data dimension (if necessary)

Index(['id', 'points', 'LUXURY', 'POPULAR_PREMIUM', 'PREMIUM', 'SUPPER_PREMIUM', 'ULTRA_PREMIUM', 'Argentina', 'Australia', 'Canada', 'France', 'Italy', 'Spain', 'US'],
      dtype='object')
predictors = ['LUXURY', 'POPULAR_PREMIUM', 'PREMIUM', 'SUPPER_PREMIUM', 'ULTRA_PREMIUM', 'Argentina', 'Australia', 'Canada', 'France', 'Italy', 'Spain', 'US']
outcome = 'points'

#6 Partition the data( for supervised tasks)
train, validate = train_test_split(wine_duplicate, test_size=0.25, random_state=1)
# print("Training : ", train.shape)
# print("Validation :  ", validate.shape)

# training (50
train, temp = train_test_split(wine_duplicate, test_size=0.3, random_state=1)
validate, test = train_test_split(temp, test_size=0.3, random_state=1)
# print("Training : ", train.shape)
# print("Validation : ", validate.shape)
# print("Test : ", test.shape)

# Training :  (60840, 9)
# Validation :   (40560, 9)
# Training :  (50700, 9)
# Validation :  (30420, 9)
# Test :  (20280, 9)

X = train[predictors]
y = train[outcome]
# (50700, 8)
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=1)
# ## ## Logistic Regression
# logit_reg = LogisticRegression(penalty="l2", C=1e42, solver='liblinear', class_weight='balanced')
logit_reg = LogisticRegression()

logit_reg.fit(train_X, train_y)

# score = logit_reg.score(train_X, train_y)
# print(score)

print(pd.DataFrame({'coeff': logit_reg.coef_[0]}, index=X.columns))
regressionSummary(train_y, logit_reg.predict(train_X))
# # Model Metrics
classificationSummary(train_y, logit_reg.predict(train_X))
classificationSummary(valid_y, logit_reg.predict(valid_X))
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
lr_prediction_train = logit_reg.predict_proba(train_X)[:, 1] > 0.5
lr_prediction_valid = logit_reg.predict_proba(valid_X)[:, 1] > 0.5
# print("LR Accuracy on train is:", accuracy_score(train_y, lr_prediction_train))
# print("LR Accuracy on test is:", accuracy_score(valid_y, lr_prediction_valid))
# print("LR Precision_score train is:", precision_score(train_y, lr_prediction_train, average='micro'))
# print("LR Precision_score on test is:", precision_score(valid_y, lr_prediction_valid, average='micro'))
# print("LR Recall_score on train is:", recall_score(train_y, lr_prediction_train, average='micro'))
# print("LR Recall_score on test is:", recall_score(valid_y, lr_prediction_valid, average='micro'))
# print("LR f1_score on train is:", f1_score(train_y, lr_prediction_train, average='micro'))
# print("LR f1_score on test is:", f1_score(valid_y, lr_prediction_valid, average='micro'))

# # # Decision Tree
DecisionTree = DecisionTreeClassifier(max_depth=4)
DecisionTree.fit(train_X, train_y)

# # plotDecisionTree(DecisionTree, feature_names=train_X.columns)
# # # fig, ax = plt.subplots(figsize=(8, 10))
# # # tree.plot_tree(DecisionTree, fontsize=6)
# # # plt.show()
# # dot_data = export_graphviz(DecisionTree, filled=True, rounded=True,
# #                                     class_names=['Setosa',
# #                                                 'Versicolor',
# #                                                 'Virginica'],
# #                                     feature_names=['predictors'],
# #                                     out_file=None)
# # graph = graph_from_dot_data(dot_data)
# # graph.write_png('tree.png')
# # # graph = graphviz.Source(DecisionTree, format="png")
# # # graph.render("decision_tree_graphivz")
# #
# # # dot_data = StringIO()
# # # export_graphviz(DecisionTree, out_file=dot_data,
# # #                 filled=True, rounded=True,
# # #                 special_characters=True, feature_names = predictors, class_names=['0', '1'])
# # # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# # # graph.write_png('wine.png')
# # # Image(graph.create_png())
# #
importances = DecisionTree.feature_importances_
#
im = pd.DataFrame({'feature': train_X.columns, 'importance': importances})
im = im.sort_values('importance',ascending=False)
print(im)
# #
# #
dt_prediction_train = DecisionTree.predict(train_X)
dt_prediction_valid = DecisionTree.predict(valid_X)
#
print("DT Accuracy score on train is:", accuracy_score(train_y, dt_prediction_train))
print("DT Accuracy score on test is:", accuracy_score(valid_y, dt_prediction_valid))
print("DT Precision score on train is:", precision_score(train_y, dt_prediction_train, average='micro'))
print("DT Precision score on test is:", precision_score(valid_y, dt_prediction_valid, average='micro'))
print("DT Recall score on train is:", recall_score(train_y, dt_prediction_train, average='micro'))
print("DT Recall score on test is:", recall_score(valid_y, dt_prediction_valid, average='micro'))
print("DT F1 score on train is:", f1_score(train_y, dt_prediction_train, average='micro'))

# Naive Bayes
nb = GaussianNB()
nb.fit(train_X, train_y)
# #
# # # predict probabilities
 #
# # # #Model matrix
nb_prediction_train = nb.predict(train_X)
nb_prediction_valid = nb.predict(valid_X)

print("NB_Accuracy on train is:",accuracy_score(train_y,nb_prediction_train))
print("NB_Accuracy on test is:",accuracy_score(valid_y,nb_prediction_valid))
print("NB_Precision_score train is:",precision_score(train_y,nb_prediction_train, average='micro'))
print("NB_Precision_score on test is:",precision_score(valid_y,nb_prediction_valid, average='micro'))
print("NB_Recall_score on train is:",recall_score(train_y,nb_prediction_train, average='micro'))
print("NB_Recall_score on test is:",recall_score(valid_y,nb_prediction_valid, average='micro'))
print("NB_f1_score on train is:",f1_score(train_y,nb_prediction_train, average='micro'))
print("NB_f1_score on test is:",f1_score(valid_y,nb_prediction_valid, average='micro'))
#
# # #Random forest
rf = RandomForestClassifier(random_state=0)
cc_rf = rf.fit(train_X.values, train_y.values.ravel())
rf_prediction_train = cc_rf.predict(train_X)
rf_prediction_valid = cc_rf.predict(valid_X)
#
print("RF_Accuracy on train is:",accuracy_score(train_y,rf_prediction_train))
print("RF_Accuracy on test is:",accuracy_score(valid_y,rf_prediction_valid))
print("RF_Precision_score train is:",precision_score(train_y,rf_prediction_train, average='micro'))
print("RF_Precision_score on test is:",precision_score(valid_y,rf_prediction_valid, average='micro'))
print("RF_Recall_score on train is:",recall_score(train_y,rf_prediction_train, average='micro'))
print("RF_Recall_score on test is:",recall_score(valid_y,rf_prediction_valid, average='micro'))
print("RF_f1_score on train is:",f1_score(train_y,rf_prediction_train, average='micro'))
print("RF_f1_score on test is:",f1_score(valid_y,rf_prediction_valid, average='micro'))

# # Gradient Boosted Trees¶
gbm = GradientBoostingClassifier(random_state=0)
gbm.fit(train_X, train_y)
gbm.predict(valid_X[:2])
#
importances = list(zip(gbm.feature_importances_, wine_duplicate.columns))
# pd.DataFrame(importances, index=[x for (_,x) in importances]).sort_values(by = 0, ascending = False).plot(kind = 'bar', color = 'b', figsize=(20,8) )
gbt_prediction_train = gbm.predict(train_X)
gbt_prediction_valid = gbm.predict(valid_X)

print("GB_Accuracy on train is:", accuracy_score(train_y,gbt_prediction_train))
print("GB_Accuracy on test is:", accuracy_score(valid_y,gbt_prediction_valid))
print("GB_Precision_score train is:", precision_score(train_y,gbt_prediction_train, average='micro'))
print("GB_Precision_score on test is:", precision_score(valid_y,gbt_prediction_valid,  average='micro'))
print("GB_Recall_score on train is:", recall_score(train_y,gbt_prediction_train, average='micro'))
print("GB_Recall_score on test is:", recall_score(valid_y,gbt_prediction_valid, average='micro'))
print("GB_f1_score on train is:", f1_score(train_y,gbt_prediction_train, average='micro'))
print("GB_f1_score on test is:", f1_score(valid_y,gbt_prediction_valid, average='micro'))
# #
#
# # Neural Network

# scaler = MinMaxScaler()
# X_scale=scaler.fit_transform(X)
# print(X_scale.shape)
# # y_tr = wine_duplicate[[outcome]] # for neural network only because this model need to get y_df.shape[1]
# train_X_nn, valid_X_nn, train_y_nn, valid_y_nn = train_test_split(X_scale, y_tr, test_size=0.3, random_state=1)
#
# model2 = tf.keras.Sequential([ tf.keras.layers.Flatten(),
#                              tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
#                              tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid)])
# model2.compile(optimizer= tf.optimizers.Adam(),
#               loss = tf.losses.binary_crossentropy,
#               metrics = ['accuracy'])
# model2.fit(train_X_nn, train_y_nn, epochs = 15)
#
# model2.evaluate(valid_X_nn, valid_y_nn)
# y_pred_valid = model2.predict(valid_X_nn).ravel()
# y_pred = model2.predict(train_X_nn).ravel()
#
# Model Matrix
# predict probabilities for test set
# prediction_probs = model2.predict(valid_X_nn)
# # predict classes
# nn_prediction_train = (model2.predict(train_X_nn) > 0.5).astype("int32")
# nn_prediction_valid = (model2.predict(valid_X_nn) > 0.5).astype("int32")
#
## Baseline AUC
# LR
# fpr, tpr, thresholds = roc_curve(train_y, lr_prediction_train, pos_label="test")
# # print("LogisticRegression Train: ", str(auc(fpr, tpr)))
# fpr, tpr, thresholds = roc_curve(valid_y, lr_prediction_valid, pos_label="test")
# print("LogisticRegression Valid: ", str(auc(fpr, tpr)), "\n")
# DT
# fpr, tpr, thresholds = roc_curve(train_y, dt_prediction_train, pos_label="test")
# print("DecisionTree Train: ", str(auc(fpr, tpr)))
# fpr, tpr, thresholds = roc_curve(valid_y, dt_prediction_valid, pos_label="test"),
# print("DecisionTree Valid: ", str(auc(fpr, tpr)), "\n")
# NB
# fpr, tpr, thresholds = roc_curve(train_y,nb_prediction_train)
# print("NaiveBayes Train: ",str(auc(fpr, tpr)))
# fpr, tpr, thresholds = roc_curve(valid_y,nb_prediction_valid)
# print("NaiveBayes Valid: ",str(auc(fpr, tpr)),"\n")
# # RF
# fpr, tpr, thresholds = roc_curve(train_y,rf_prediction_train)
# print("RandomForest Train: ",str(auc(fpr, tpr)))
# fpr, tpr, thresholds = roc_curve(valid_y,rf_prediction_valid)
# print("RandomForest Valid: ",str(auc(fpr, tpr)),"\n")
# # GBT
# fpr, tpr, thresholds = roc_curve(train_y,gbt_prediction_train )
# print("GradientBoostedTree Train: ",str(auc(fpr, tpr)))
# fpr, tpr, thresholds = roc_curve(valid_y,gbt_prediction_valid)
# print("GradientBoostedTree Valid: ",str(auc(fpr, tpr)),"\n")
# # NN
# fpr, tpr, thresholds = roc_curve(train_y_nn, y_pred)
# print("NeuralNetworks Train: ",str(auc(fpr, tpr)))
# #fpr, tpr, thresholds = roc_curve(valid_y,nn_prediction_valid)
# fpr, tpr, thresholds = roc_curve(valid_y_nn, y_pred_valid)
# print("NeuralNetworkss Valid: ",str(auc(fpr, tpr)),"\n")

# ## ROC Curve Analysis
# rf_proba = cc_rf.predict_proba(valid_X)[:,1]
# rf_roc = roc_curve(valid_y, rf_proba)
# rf_roc = pd.DataFrame(rf_roc)
# gbm_proba=gbm.predict_proba(valid_X)[:,1]
# gbm_roc = roc_curve(valid_y, gbm_proba)
# gbm_roc = pd.DataFrame(gbm_roc)
# nn_proba = model2.predict(valid_X_nn)
# nn_roc = roc_curve(valid_y_nn, nn_proba)
# nn_roc = pd.DataFrame(nn_roc)
# Classifier = [logit_reg, nb, DecisionTree]
# result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
# for cls in Classifier:
#     yproba = cls.predict_proba(valid_X)[:, 1]
#     # plot_roc_curve(cls, valid_X, valid_y)
#     fpr, tpr, thresholds = roc_curve(valid_y, yproba)
#
#     auc = roc_auc_score(valid_y, yproba)
#
#     result_table = result_table.append({'classifiers': cls,
#                                         'fpr': fpr,
#                                         'tpr': tpr,
#                                         'auc': auc}, ignore_index=True)
# result_table.set_index('classifiers', inplace=True)
# # result_table.fillna(0)
# fig = plt.figure(figsize=(8, 6))
#
# print(result_table.head())
#
# for i in result_table.index:
#     plt.plot(result_table.loc[i]['fpr'],
#              result_table.loc[i]['tpr'],
#              label=i)
#     # label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
#
# plt.plot(rf_roc.loc[0, :], rf_roc.loc[1, :], label="Random Forest")
# plt.plot(gbm_roc.loc[0, :], gbm_roc.loc[1, :], label="Gradiant Boosted Tree")
# plt.plot(nn_roc.loc[0, :], nn_roc.loc[1, :], label="Neural Network")
#
# plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
#
# plt.xticks(np.arange(0.0, 1.1, step=0.1))
# plt.xlabel("False Positive Rate", fontsize=15)
#
# plt.yticks(np.arange(0.0, 1.1, step=0.1))
# plt.ylabel("True Positive Rate", fontsize=15)
#
# plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
# plt.legend(prop={'size': 8}, loc='lower right')
#
# plt.show()

##7 Choose the data mining technique to be used(regression, neural nets, hierachcal cluster and so on)

##8Use algorithm to perform the task

##9 Interpret the result of the algorithm
##10 Deploy the mode
