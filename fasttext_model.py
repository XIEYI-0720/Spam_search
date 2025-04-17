import csv
import datetime
import os
import fasttext
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from transformers import GPT2Model
import seaborn as sns


# 设置全局字体和字号
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置全局字体为宋体
plt.rcParams['font.size'] = 10.5  # 设置全局字号为五号
plt.rcParams['axes.unicode_minus'] = False

begin=datetime.datetime.now()
# 加载数据
data = pd.read_csv('./data/data.csv')
X = data['msg_new']
y = data['label'].apply(lambda x: '__label__1' if x == 1 else '__label__0')

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=71)

# 准备训练和验证文件
train_file = './fasttext/ft_train.txt'
val_file = './fasttext/ft_val.txt'
with open(train_file, 'w', encoding='utf-8') as f:
    for text, label in zip(X_train, y_train):
        f.write(label + ' ' + text + '\n')

with open(val_file, 'w', encoding='utf-8') as f:
    for text, label in zip(X_val, y_val):
        f.write(label + ' ' + text + '\n')

# 网格搜索参数
lr_list = [0.01,0.02,0.05,0.1,0.2]
epoch_list = [30, 40, 50]
dim_list= [50,100,200]
word_ngrams_list = [1, 2, 3]

# 保存结果
results = []
# 初始化列表来存储每个epoch的训练和验证准确率
train_accuracies = []
val_accuracies = []
best_accuracy = 0
for lr in lr_list:
    for word_ngrams in word_ngrams_list:
        for dim in dim_list:
            for epoch in epoch_list:
                # 训练模型，包括dim参数
                model = fasttext.train_supervised(input=train_file, lr=lr, epoch=epoch, wordNgrams=word_ngrams, dim=dim)
                
                # 预测验证集
                val_preds = [model.predict(text)[0][0] for text in X_val]
                # print(val_preds)
                val_labels = y_val.apply(lambda x: x).tolist()
                val_preds = [1 if x == '__label__1' else 0 for x in val_preds]
                # print(val_preds)
                val_labels= [1 if x == '__label__1' else 0 for x in val_labels]
                # print(val_labels)
                # 计算指标
                accuracy = accuracy_score(val_labels, val_preds)
                precision = precision_score(val_labels, val_preds)
                recall=recall_score(val_labels, val_preds)
                f1 = f1_score(val_labels, val_preds)
                # print(accuracy,recall,f1)

                # 保存结果
                results.append({
                    'lr': lr,
                    'epoch': epoch,
                    'dim':dim,
                    'word_ngrams': word_ngrams,
                    'accuracy': accuracy,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1
                })
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    model.save_model(f'./fasttext/model_lr{lr}_epoch{epoch}_dim{dim}_ngrams{word_ngrams}.bin')
                    cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])
                    # 使用pandas将混淆矩阵转换为DataFrame，以便更好地显示标签
                    cm_df = pd.DataFrame(cm, index=['实际: 0', '实际: 1'], columns=['预测: 0', '预测: 1'])
                    # 使用seaborn绘制混淆矩阵热图
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
                    plt.ylabel('实际值')
                    plt.xlabel('预测值')
                    plt.tight_layout(pad=1.0)  # 设置单倍行距  
                    plt.savefig(f'./fasttext/cm_lr{lr}_epoch{epoch}_dim{dim}_ngrams{word_ngrams}.png')
                
# 保存结果到CSV
results_df = pd.DataFrame(results)
results_df.to_csv('./fasttext/fasttext.csv', index=False)
# 选择最优模型

best_model_index = results_df['accuracy'].idxmax()
best_model_params = results_df.iloc[best_model_index]

# 打印最优模型的参数
print(f"最优模型的参数：\n{best_model_params}")



endt=datetime.datetime.now()
print('time:',endt-begin)
# def index(preds,labels):    # 计算指标
#     acc = accuracy_score(labels, preds)
#     precision = precision_score(labels, preds)
#     recall=recall_score(labels, preds)
#     f1 = f1_score(labels, preds)
#     print(f'准确率 = {acc}')
#     print(f'精确度 = {precision}')
#     print(f'召回率 = {recall}')
#     print(f'F1 值 = {f1}')
#     # with open('./results/fasttext_predict_result.txt', 'a', encoding='utf-8') as f:
#     #     f.write(f'准确率 = {acc}\n')
#     #     f.write(f'精确度 = {precision}\n')
#     #     f.write(f'F1 值 = {f1}\n\n\n')
#     #     f.close()
# def fasttext_prob(texts):
#     # 获取测试集上的预测概率
#     predicted_probs = []
#     label=[]
#     for text in texts:
#         # 使用predict方法获取预测标签及其概率，k=1返回最可能的一个标签及其概率
#         predicted_label, prob = fasttext_model.predict(text, k=1)
#         prob = prob[0]  # 从列表中提取概率值

#         # 确保概率值在0到1之间
#         prob = min(max(prob, 0.0), 1.0)
#         formatted_prob = 1.0 - prob
        
#         # 确保概率值和它的补数都被转换为浮点数
#         if predicted_label[0] == '__label__1':
#             a = 1
#             label.append(a)
#             predicted_probs.append([float(formatted_prob), float(prob)])  # 假设其他标签的概率为1-prob
#         else:
#             label.append(0)
#             predicted_probs.append([float(prob), float(formatted_prob)])  # 假设其他标签的概率为1-prob

#     return label, predicted_probs

# print(datetime.datetime.now())
# test_data = pd.read_csv('./data/test.csv')

# msg_test_list = [str(msg) for msg in test_data['msg_new'].tolist()]
# label_test_list = test_data['label'].tolist()
#     # 加载模型
# fasttext_model = fasttext.load_model('./models/fasttext_model.bin')

# y, fasttext_train_preds = fasttext_prob(msg_test_list)
# final_train_voting_predictions = np.argmax(fasttext_train_preds, axis=1)

# index(y, label_test_list)
# # index(final_train_voting_predictions, label_test_list)
# print(datetime.datetime.now())
