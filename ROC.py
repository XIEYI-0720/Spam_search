import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
import fasttext
import torch
from transformers import BertForSequenceClassification, BertTokenizer


# 加载模型 
logistic_model = joblib.load('./models/logistic_regression_best_model.pkl')
fasttext_model = fasttext.load_model('./models/fasttext_model.bin')
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bert_model = BertForSequenceClassification.from_pretrained("./models/bert_best_model")
meta_model = joblib.load('meta_model.pkl')
# 加载数据
data = pd.read_csv('./data/example.csv')
X_val = [str(msg) for msg in data['msg_new'].tolist()]
bX_val=[str(msg) for msg in data['message'].tolist()]
y_val = data['label'].tolist()  # 将标签转换为二进制形式

def lget_prediction_probabilities(model, texts):
    probabilities=[]
    probs = model.predict_proba(texts)
    for prob in probs:
        prob = prob[1]
        probabilities.append(prob)
    return probabilities

def fget_prediction_probabilities(model, texts):
    probabilities = []
    # 对每个文本进行预测
    for text in texts:
        # 使用FastText模型预测，获取所有类别的概率
        labels, probs = model.predict(text, k=-1)  # 假设k=-1返回所有标签的概率
        # print(labels,probs)
        # 假设我们关注的正类标签是'__label__1'
        # 找到正类标签的概率
        prob = probs[labels.index('__label__1')] if '__label__1' in labels else 0
        # print('------',prob)
        probabilities.append(prob)
    return probabilities

def bget_prediction_probabilities(model,texts):
    model.eval()
    probabilities = []
    for text in texts:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
            probs = probs[0][1]
            probabilities.append(probs)
    return probabilities

def sget_prediction_probabilities(model,texts):
    # 准备元模型的输入特征（将预测概率作为特征）
    meta_features = np.column_stack((fval_probabilities, bval_probabilities))
    # 获取元模型的预测概率
    meta_val_probabilities = meta_model.predict_proba(meta_features)[:, 1]
    pass
# 获取验证集上的预测概率
lval_probabilities = lget_prediction_probabilities(logistic_model, X_val)
# 计算ROC曲线的FPR和TPR
lfpr, ltpr, _ = roc_curve(y_val, lval_probabilities)
# 计算AUC分数
lroc_auc = auc(lfpr, ltpr)
# print(lroc_auc)

# 获取验证集上的预测概率
fval_probabilities = fget_prediction_probabilities(fasttext_model, X_val)
# 计算ROC曲线的FPR和TPR
ffpr, ftpr, _ = roc_curve(y_val, fval_probabilities)
# 计算AUC分数
froc_auc = auc(ffpr, ftpr)
# print(roc_auc)

# 获取验证集上的预测概率
bval_probabilities = bget_prediction_probabilities(bert_model, bX_val)
# print(bval_probabilities)
# 计算ROC曲线的FPR和TPR
bfpr, btpr, _ = roc_curve(y_val, bval_probabilities)
# 计算AUC分数
broc_auc = auc(bfpr, btpr)
# print(broc_auc)

# 获取验证集上的预测概率
sval_probabilities = sget_prediction_probabilities(meta_model, X_val, bX_val)
# 计算元模型的ROC曲线的FPR和TPR
mfpr, mtpr, _ = roc_curve(y_val, sval_probabilities)
# 计算元模型的AUC分数
mroc_auc = auc(mfpr, mtpr)



# 假设你已经计算了两个模型的FPR、TPR和AUC
fpr1, tpr1, roc_auc1 = ffpr,ftpr,froc_auc# 模型1的FPR、TPR和AUC
fpr2, tpr2, roc_auc2 =bfpr,btpr,broc_auc # 模型2的FPR、TPR和AUC
fpr3, tpr3, roc_auc3 =lfpr,ltpr,lroc_auc # 模型2的FPR、TPR和AUC
plt.figure()
lw = 2
plt.plot(fpr1, tpr1, color='darkorange', lw=lw, label='模型1 ROC curve (area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='blue', lw=lw, label='模型2 ROC curve (area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='green', lw=lw, label='模型3 ROC curve (area = %0.2f)' % roc_auc3)  # Add Model 3
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic comparison')
plt.legend(loc="lower right")
plt.show()