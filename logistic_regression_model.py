import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, f1_score, log_loss, precision_score, recall_score, roc_curve
from sklearn.pipeline import make_pipeline
import joblib
import csv
from scipy.stats import ks_2samp



# 设置全局字体和字号
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置全局字体为宋体
plt.rcParams['font.size'] = 10.5  # 设置全局字号为五号
plt.rcParams['axes.unicode_minus'] = False



# bg=datetime.datetime.now()
# # Load data
# data = pd.read_csv("./data/data.csv")
# X = data['msg_new']
# y = data['label']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(random_state=42
#                                                                ))
# param_grid = [
#         {
#             'logisticregression__penalty': ['l2'],  # 'lbfgs', 'newton-cg', 'sag', 'saga' support these
#             'logisticregression__C': [0.01, 0.1, 1, 10, 100],
#             'logisticregression__solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
#             'tfidfvectorizer__max_features': [None, 5000, 10000],
#             'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)]
#         },
#         {
#             'logisticregression__penalty': ['l1'],  # 'saga' and 'liblinear' support l1
#             'logisticregression__C': [0.01, 0.1, 1, 10, 100],
#             'logisticregression__solver': ['saga', 'liblinear'],
#             'tfidfvectorizer__max_features': [None, 5000, 10000],
#             'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)]
#         },
#         {
#             'logisticregression__penalty': ['elasticnet'],  # Only 'saga' supports elasticnet
#             'logisticregression__C': [0.01, 0.1, 1, 10, 100],
#             'logisticregression__solver': ['saga'],
#             'logisticregression__l1_ratio': [0, 0.25, 0.5, 0.75, 1],  # Add different l1_ratio values
#             'tfidfvectorizer__max_features': [None, 5000, 10000],
#             'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)]
#         }
#     ]
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# y_pred_proba = best_model.predict_proba(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall=recall_score(y_test, y_pred)
# loss = log_loss(y_test, y_pred_proba)
# f1 = f1_score(y_test, y_pred)
# joblib.dump(best_model, './logistic/logistic_regression_best_model.pkl')


# # Assuming y_pred_proba contains the probabilities of the positive class
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
# roc_auc = auc(fpr, tpr)

# positive_proba = y_pred_proba[:, 1][y_test == 1]
# negative_proba = y_pred_proba[:, 1][y_test == 0]
# ks_statistic=ks_2samp(positive_proba, negative_proba).statistic
# # print(f"KS Statistic: {ks_statistic}")


# with open('./logistic/logistic_regression.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Accuracy', 'precision','recall','f1','Log Loss','KS Statistic'])
#     writer.writerow([accuracy, precision,recall, f1, loss, ks_statistic])
#     file.close()
# train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
# ed=datetime.datetime.now()
# print(ed-bg)
test_data = pd.read_csv('./data/test.csv')
msg_test_list = test_data['msg_new'].astype(str).tolist()
logistic_model = joblib.load('./logistic/logistic_regression_best_model.pkl')
lprobabilities=[]
probs = logistic_model.predict(msg_test_list)
for prob in probs:
    lprobabilities.append(prob)
with open(f'./sv/predictlogistic.csv', 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(lprobabilities)
    f.close()

# import joblib

# # 加载模型
# model = joblib.load('./models/logistic_regression_best_model.pkl')


# logistic_regression_model = model.named_steps['logisticregression']
# C_parameter = logistic_regression_model.C
# penalty_parameter = logistic_regression_model.penalty
# #     'logisticregression__penalty': ['l1', 'l2'],
# #     'tfidfvectorizer__max_features': [None, 5000, 10000],
# #     'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)]
# # }
# print(C_parameter,penalty_parameter)

# tfidf_vectorizer = model.named_steps['tfidfvectorizer']
# max_features_parameter = tfidf_vectorizer.max_features
# print("max_features:", max_features_parameter)
# ngram_range_parameter = tfidf_vectorizer.ngram_range
# print("ngram_range:", ngram_range_parameter)