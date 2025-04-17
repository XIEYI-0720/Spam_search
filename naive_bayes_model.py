import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


print(datetime.datetime.now())
# 设置全局字体和字号
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置全局字体为宋体
plt.rcParams['font.size'] = 5  # 设置全局字号为五号
plt.rcParams['axes.unicode_minus'] = False



# 加载数据
df = pd.read_csv("./data/data.csv")
# 数据预处理
X = df['msg_new']
y = df['label']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建一个管道，包含TF-IDF向量化和朴素贝叶斯模型
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

# 参数网格
parameters = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'tfidf__ngram_range': ((1, 1), (1, 2)),  # 单个词汇和双词汇组合
    'nb__alpha': (1e-2, 1e-3, 1e-1)
}

# 网格搜索
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 最佳参数和模型评估
print("Best Parameters: ", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

# 绘制学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    return plt

plot_learning_curve(grid_search.best_estimator_, "Learning Curve for Naive Bayes", X_train, y_train, cv=5)
plt.tight_layout(pad=1.0)  # 设置单倍行距  
# plt.savefig(f'./bert_result/Learning curve.pdf')
plt.show()

model_filename = f'./models/Naive_Bayes_model.pkl'
joblib.dump(pipeline, model_filename)
