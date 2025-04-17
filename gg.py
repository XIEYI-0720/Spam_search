from sklearn.model_selection import StratifiedKFold, learning_curve, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers


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

# 特征维度为128
input_dim = 128

# 构建FastText模型
model = Sequential()
# 输入层
model.add(Dense(256, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.5))  # 添加Dropout层进行正则化
# 隐藏层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 添加Dropout层进行正则化
# 输出层，二分类任务
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 输出模型结构
model.summary()

# 准备数据
data = pd.read_csv("./data/example.csv")
X = data["message"].tolist()
y = data["label"].tolist()

# 进行交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=skf)

# 绘制学习曲线
plt.plot(train_sizes, train_scores.mean(axis=1), color='blue', label='Training score')
plt.plot(train_sizes, test_scores.mean(axis=1), color='red', label='Cross-validation score')
plt.legend()
plt.show()