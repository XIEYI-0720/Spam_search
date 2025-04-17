import csv
import datetime
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, auc, classification_report, f1_score, precision_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

begin=datetime.datetime.now()
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('svm', SVC(probability=True))  # 使用SVC并启用概率估计
])

# 调整参数网格
parameters = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'tfidf__ngram_range': ((1, 1), (1, 2)),  # 单个词汇和双词汇组合
    'svm__C': (1, 10, 100),  # SVM正则化参数
    'svm__kernel': ('linear', 'rbf'),
    'svm__gamma': ('scale', 'auto', 0.01, 0.1, 1),
    'svm__degree': (2, 3, 4),
    'svm__coef0': (0.0, 0.5, 1.0)
}

# 网格搜索
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 最佳参数和模型评估
print("Best Parameters: ", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))
# 保存模型
model_filename = './models/SVM_model.pkl'
joblib.dump(grid_search.best_estimator_, model_filename)
end=datetime.datetime.now()
print('time:',end-begin)



model=joblib.load('./models/SVM_model.pkl')
data = pd.read_csv("./data/data.csv")
msg_test_list = data['msg_new'].astype(str).tolist()
label_test_list = data['label'].tolist()
preds=model.predict(msg_test_list)
accuracy=accuracy_score(label_test_list, preds)
precision=precision_score(label_test_list, preds, average='weighted')
f1=f1_score(label_test_list, preds, average='weighted')
probabilities=[]
probs=model.predict_proba(msg_test_list)
print(probs)
for prob in probs:
    prob = prob[1]
    probabilities.append(prob)

print(probabilities)

fpr, tpr, _ = roc_curve(label_test_list, probabilities)
    # 计算AUC分数
roc_auc = auc(fpr, tpr)
    # 假设你已经计算了两个模型的FPR、TPR和AUC

with open('./results/AUC_SVM.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['fpr', 'tpr', 'auc'])
    writer.writerow([fpr, tpr, roc_auc])
    file.close()